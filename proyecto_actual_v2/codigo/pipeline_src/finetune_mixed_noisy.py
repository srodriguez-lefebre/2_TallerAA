from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_logmel_image_cache import logmel_image_cache_path  # noqa: E402
from fat2019.data import load_training_labels  # noqa: E402
from fat2019.labels import dataframe_to_multihot  # noqa: E402
from fat2019.metrics import calculate_overall_lwlrap  # noqa: E402
from fat2019.neural_helpers import (  # noqa: E402
    compute_pos_weight,
    make_train_valid_indices,
    sigmoid_numpy,
)
from fat2019.submission import (  # noqa: E402
    label_columns_from_sample,
    read_sample_submission,
    write_submission,
)
from final_config import BRANCHES, FINE_TUNE, BranchConfig  # noqa: E402
from train_logmel_cnn import (  # noqa: E402
    LogmelDataset,
    SmallLogmelCnn,
    build_optimizer,
    build_scheduler,
)


@dataclass(frozen=True)
class BranchRunResult:
    branch: str
    baseline_lwlrap: float
    best_lwlrap: float
    final_lwlrap: float
    best_epoch: int
    submission_path: Path
    best_model_path: Path
    final_model_path: Path


class IndexedLogmelDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray | None = None,
        *,
        indices: np.ndarray | None = None,
    ) -> None:
        self.images = images
        self.targets = targets
        self.indices = (
            np.arange(len(images), dtype=np.int64)
            if indices is None
            else indices.astype(np.int64, copy=False)
        )

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int):
        real_index = int(self.indices[index])
        image = torch.from_numpy(self.images[real_index].astype(np.float32, copy=False)).unsqueeze(0)
        if self.targets is None:
            return image
        target = torch.from_numpy(self.targets[real_index].astype(np.float32, copy=False))
        return image, target


class MixedCuratedNoisyDataset(Dataset):
    def __init__(
        self,
        *,
        curated_images: np.ndarray,
        curated_targets: np.ndarray,
        curated_indices: np.ndarray,
        noisy_images: np.ndarray,
        noisy_targets: np.ndarray,
        noisy_indices: np.ndarray,
        curated_weight: float,
        noisy_weight: float,
        augment: bool,
        gaussian_noise_std: float,
    ) -> None:
        self.curated_images = curated_images
        self.curated_targets = curated_targets
        self.curated_indices = curated_indices.astype(np.int64, copy=False)
        self.noisy_images = noisy_images
        self.noisy_targets = noisy_targets
        self.noisy_indices = noisy_indices.astype(np.int64, copy=False)
        self.curated_weight = float(curated_weight)
        self.noisy_weight = float(noisy_weight)
        self.augment = augment
        self.gaussian_noise_std = float(gaussian_noise_std)

    @property
    def curated_len(self) -> int:
        return int(len(self.curated_indices))

    @property
    def noisy_len(self) -> int:
        return int(len(self.noisy_indices))

    def __len__(self) -> int:
        return self.curated_len + self.noisy_len

    def __getitem__(self, index: int):
        if index < self.curated_len:
            real_index = int(self.curated_indices[index])
            image = self.curated_images[real_index]
            target = self.curated_targets[real_index]
            weight = self.curated_weight
        else:
            noisy_index = int(self.noisy_indices[index - self.curated_len])
            image = self.noisy_images[noisy_index]
            target = self.noisy_targets[noisy_index]
            weight = self.noisy_weight

        image_tensor = torch.from_numpy(image.astype(np.float32, copy=False)).unsqueeze(0)
        if self.augment:
            image_tensor = augment_logmel_image(
                image_tensor,
                gaussian_noise_std=self.gaussian_noise_std,
            )
        target_tensor = torch.from_numpy(target.astype(np.float32, copy=False))
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        return image_tensor, target_tensor, weight_tensor


class HalfCuratedHalfNoisyBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        curated_len: int,
        noisy_len: int,
        batch_size: int,
        seed: int,
        steps_per_epoch: int | None = None,
    ) -> None:
        if curated_len <= 0:
            raise ValueError("curated_len must be positive")
        if noisy_len <= 0:
            raise ValueError("noisy_len must be positive")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        self.curated_len = int(curated_len)
        self.noisy_len = int(noisy_len)
        self.batch_size = int(batch_size)
        self.curated_per_batch = self.batch_size // 2
        self.noisy_per_batch = self.batch_size - self.curated_per_batch
        self.steps_per_epoch = int(
            steps_per_epoch
            if steps_per_epoch is not None
            else math.ceil(self.curated_len / self.curated_per_batch)
        )
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        curated_needed = self.steps_per_epoch * self.curated_per_batch
        noisy_needed = self.steps_per_epoch * self.noisy_per_batch
        curated_indices = _sample_epoch_indices(rng, self.curated_len, curated_needed)
        noisy_indices = _sample_epoch_indices(rng, self.noisy_len, noisy_needed) + self.curated_len

        for step in range(self.steps_per_epoch):
            c_start = step * self.curated_per_batch
            n_start = step * self.noisy_per_batch
            batch = np.concatenate(
                [
                    curated_indices[c_start : c_start + self.curated_per_batch],
                    noisy_indices[n_start : n_start + self.noisy_per_batch],
                ]
            )
            rng.shuffle(batch)
            yield batch.astype(np.int64).tolist()


def _sample_epoch_indices(
    rng: np.random.Generator,
    population_size: int,
    needed: int,
) -> np.ndarray:
    if needed <= population_size:
        return rng.permutation(population_size)[:needed]
    repeats = needed // population_size
    remainder = needed % population_size
    chunks = [rng.permutation(population_size) for _ in range(repeats)]
    if remainder:
        chunks.append(rng.permutation(population_size)[:remainder])
    return np.concatenate(chunks)


def augment_logmel_image(
    image: torch.Tensor,
    *,
    gaussian_noise_std: float,
) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        shift = int(torch.randint(-32, 33, ()).item())
        image = torch.roll(image, shifts=shift, dims=2)
    if torch.rand(()) < 0.5:
        width = int(torch.randint(8, 48, ()).item())
        start = int(torch.randint(0, max(1, image.shape[2] - width + 1), ()).item())
        image[:, :, start : start + width] = 0.0
    if torch.rand(()) < 0.5:
        width = int(torch.randint(4, 16, ()).item())
        start = int(torch.randint(0, max(1, image.shape[1] - width + 1), ()).item())
        image[:, start : start + width, :] = 0.0
    if gaussian_noise_std > 0.0 and torch.rand(()) < 0.5:
        image = image + torch.randn_like(image) * gaussian_noise_std
    return image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune E100 FAT2019 branches with curated + noisy batches.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--work-dir", type=Path, default=Path("proyecto_actual_v2/codigo/work"))
    parser.add_argument("--branches", default="all", help="Comma-separated branch names or 'all'.")
    parser.add_argument("--epochs", type=int, default=FINE_TUNE.epochs)
    parser.add_argument("--lr", type=float, default=FINE_TUNE.lr)
    parser.add_argument("--min-lr", type=float, default=FINE_TUNE.min_lr)
    parser.add_argument("--noisy-loss-weight", type=float, default=FINE_TUNE.noisy_loss_weight)
    parser.add_argument("--curated-loss-weight", type=float, default=FINE_TUNE.curated_loss_weight)
    parser.add_argument("--gaussian-noise-std", type=float, default=FINE_TUNE.gaussian_noise_std)
    parser.add_argument("--seed", type=int, default=FINE_TUNE.seed)
    parser.add_argument("--num-workers", type=int, default=FINE_TUNE.num_workers)
    parser.add_argument("--max-curated-train", type=int, default=None)
    parser.add_argument("--max-noisy", type=int, default=None)
    parser.add_argument("--submission-checkpoint", choices=("final", "best"), default="final")
    return parser


def _torch_load(path: Path, *, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_cache(
    branch: BranchConfig,
    *,
    split: str,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if branch.frames == 1024 and branch.cache_tag is None:
        memmap_stem = data_dir / f"{split}_logmel_image_m{branch.n_mels}_f{branch.frames}_x"
        memmap_path = memmap_stem.with_suffix(".npy")
        fnames_path = data_dir / f"{memmap_stem.name}_fnames.txt"
        if memmap_path.exists() and fnames_path.exists():
            x = np.load(memmap_path, mmap_mode="r")
            fnames = np.array(
                [line.strip() for line in fnames_path.read_text().splitlines() if line.strip()],
                dtype=str,
            )
            return x, fnames

    path = logmel_image_cache_path(
        data_dir,
        split=split,
        n_mels=branch.n_mels,
        frames=branch.frames,
        tag=branch.cache_tag,
    )
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    cache = np.load(path, allow_pickle=False)
    return cache["x"], cache["fnames"].astype(str)


def _assert_order(expected: list[str], actual: np.ndarray, *, label: str) -> None:
    actual_list = actual.astype(str).tolist()
    if expected != actual_list:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(expected, actual_list))
                if left != right
            ),
            None,
        )
        raise ValueError(f"{label} cache order mismatch at row {mismatch}: expected labels order")


def _predict_scores(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            output = model(images.to(device, non_blocking=True))
            logits.append(output.detach().cpu().numpy())
    return sigmoid_numpy(np.vstack(logits))


def _weighted_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for images, targets, sample_weights in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_by_class = criterion(logits, targets)
        loss_by_sample = loss_by_class.mean(dim=1)
        weighted_losses = loss_by_sample * sample_weights
        loss = weighted_losses.sum() / sample_weights.sum().clamp_min(1e-6)
        loss.backward()
        optimizer.step()

        weighted_loss_sum += float(weighted_losses.detach().sum().cpu())
        weight_sum += float(sample_weights.detach().sum().cpu())
    return weighted_loss_sum / max(weight_sum, 1e-6)


def _model_from_checkpoint(checkpoint: dict[str, object], *, num_classes: int, device: torch.device) -> SmallLogmelCnn:
    model = SmallLogmelCnn(
        num_classes=num_classes,
        architecture=str(checkpoint.get("architecture", "standard")),
        activation=str(checkpoint.get("activation", "silu")),
        block_dropout=float(checkpoint.get("block_dropout", 0.0)),
        head_hidden=int(checkpoint.get("head_hidden", 0)),
        head_dropout=float(checkpoint.get("head_dropout", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    source_checkpoint: dict[str, object],
    epoch: int,
    valid_lwlrap: float,
    branch: BranchConfig,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(source_checkpoint)
    payload.update(
        {
            "model_state": model.state_dict(),
            "epoch": int(epoch),
            "valid_lwlrap": float(valid_lwlrap),
            "fine_tune_source_epoch": int(source_checkpoint.get("epoch", 0)),
            "fine_tune_branch": branch.name,
            "fine_tune_epochs": int(args.epochs),
            "fine_tune_lr": float(args.lr),
            "fine_tune_scheduler": "cosine",
            "fine_tune_curated_loss_weight": float(args.curated_loss_weight),
            "fine_tune_noisy_loss_weight": float(args.noisy_loss_weight),
            "fine_tune_gaussian_noise_std": float(args.gaussian_noise_std),
        }
    )
    torch.save(payload, path)


def _write_branch_metadata(
    path: Path,
    *,
    branch: BranchConfig,
    source_metadata: dict[str, object],
    result: BranchRunResult,
    args: argparse.Namespace,
    rows: dict[str, int],
    steps_per_epoch: int,
) -> None:
    lines = [
        f"# {branch.name} noisy fine-tune",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline validation lwlrap | {result.baseline_lwlrap:.6f} |",
        f"| Best validation lwlrap | {result.best_lwlrap:.6f} |",
        f"| Final validation lwlrap | {result.final_lwlrap:.6f} |",
        f"| Delta best-baseline | {result.best_lwlrap - result.baseline_lwlrap:+.6f} |",
        f"| Best epoch | {result.best_epoch} |",
        f"| Epochs requested | {args.epochs} |",
        f"| LR | {args.lr} |",
        f"| Noisy loss weight | {args.noisy_loss_weight} |",
        f"| Steps per epoch | {steps_per_epoch} |",
        "",
        "## Source config",
        "",
        "```json",
        json.dumps(source_metadata, indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "```json",
        json.dumps(rows, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_branch(branch: BranchConfig, args: argparse.Namespace) -> BranchRunResult:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    work_dir = args.work_dir
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)

    curated_labels = load_training_labels(data_dir / "train_curated.csv")
    noisy_labels = load_training_labels(data_dir / "train_noisy.csv")
    curated_y = dataframe_to_multihot(curated_labels, label_columns).astype(np.float32)
    noisy_y = dataframe_to_multihot(noisy_labels, label_columns).astype(np.float32)

    curated_x, curated_fnames = _load_cache(branch, split="curated", data_dir=data_dir)
    noisy_x, noisy_fnames = _load_cache(branch, split="noisy", data_dir=data_dir)
    _assert_order(curated_labels["fname"].astype(str).tolist(), curated_fnames, label="curated")
    _assert_order(noisy_labels["fname"].astype(str).tolist(), noisy_fnames, label="noisy")

    train_indices, valid_indices = make_train_valid_indices(
        num_rows=len(curated_x),
        test_size=FINE_TUNE.validation_test_size,
        seed=args.seed,
        full_train=False,
    )
    if args.max_curated_train is not None:
        train_indices = train_indices[: args.max_curated_train]
    noisy_indices = np.arange(len(noisy_x), dtype=np.int64)
    if args.max_noisy is not None:
        noisy_indices = noisy_indices[: args.max_noisy]
        noisy_labels = noisy_labels.head(args.max_noisy).copy()

    source_checkpoint_path = branch.source_checkpoint(work_dir)
    source_metadata_path = branch.source_metadata(work_dir)
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(
            f"missing source checkpoint for {branch.name}: {source_checkpoint_path}. "
            "Run the curated E100 stage first."
        )
    if not source_metadata_path.exists():
        raise FileNotFoundError(
            f"missing source metadata for {branch.name}: {source_metadata_path}. "
            "Run the curated E100 stage first."
        )
    source_checkpoint = _torch_load(source_checkpoint_path, map_location=device)
    source_metadata = json.loads(source_metadata_path.read_text())
    if int(source_checkpoint.get("frames", branch.frames)) != branch.frames:
        raise ValueError(f"{branch.name}: checkpoint frames do not match branch config")
    if str(source_checkpoint.get("cache_tag", branch.cache_tag)) != str(branch.cache_tag):
        raise ValueError(f"{branch.name}: checkpoint cache tag does not match branch config")

    model = _model_from_checkpoint(
        source_checkpoint,
        num_classes=len(label_columns),
        device=device,
    )

    valid_loader = DataLoader(
        IndexedLogmelDataset(curated_x, curated_y, indices=valid_indices),
        batch_size=branch.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    baseline_scores = _predict_scores(model, valid_loader, device=device)
    baseline_lwlrap = calculate_overall_lwlrap(curated_y[valid_indices], baseline_scores)

    dataset = MixedCuratedNoisyDataset(
        curated_images=curated_x,
        curated_targets=curated_y,
        curated_indices=train_indices,
        noisy_images=noisy_x,
        noisy_targets=noisy_y,
        noisy_indices=noisy_indices,
        curated_weight=args.curated_loss_weight,
        noisy_weight=args.noisy_loss_weight,
        augment=True,
        gaussian_noise_std=args.gaussian_noise_std,
    )
    steps_per_epoch = math.ceil(len(train_indices) / (branch.batch_size // 2))
    train_loader = DataLoader(
        dataset,
        batch_sampler=HalfCuratedHalfNoisyBatchSampler(
            curated_len=dataset.curated_len,
            noisy_len=dataset.noisy_len,
            batch_size=branch.batch_size,
            seed=args.seed,
            steps_per_epoch=steps_per_epoch,
        ),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pos_weight = torch.from_numpy(compute_pos_weight(curated_y[train_indices])).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=str(source_metadata.get("optimizer", "adamw")),
        learning_rate=args.lr,
        weight_decay=float(source_metadata.get("weight_decay", 1e-4)),
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name="cosine",
        epochs=args.epochs,
        plateau_patience=2,
        plateau_factor=0.5,
        lr_milestones=[],
        min_lr=args.min_lr,
    )

    branch_run_dir = branch.run_dir(work_dir)
    model_dir = branch_run_dir / "models"
    experiments_dir = branch_run_dir / "experiments"
    submissions_dir = branch_run_dir / "submissions"
    best_model_path = model_dir / "small_logmel_cnn_best.pt"
    final_model_path = model_dir / "small_logmel_cnn_final.pt"
    history: list[dict[str, float | int]] = []
    best_lwlrap = baseline_lwlrap
    best_epoch = 0
    _save_checkpoint(
        best_model_path,
        model=model,
        source_checkpoint=source_checkpoint,
        epoch=0,
        valid_lwlrap=baseline_lwlrap,
        branch=branch,
        args=args,
    )

    print(
        f"{branch.name}: baseline_lwlrap={baseline_lwlrap:.6f} "
        f"device={device} steps_per_epoch={steps_per_epoch}",
        flush=True,
    )
    final_lwlrap = baseline_lwlrap
    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _weighted_train_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        valid_scores = _predict_scores(model, valid_loader, device=device)
        valid_lwlrap = calculate_overall_lwlrap(curated_y[valid_indices], valid_scores)
        final_lwlrap = valid_lwlrap
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_lwlrap": float(valid_lwlrap),
                "learning_rate": learning_rate,
            }
        )
        print(
            f"{branch.name} epoch {epoch}: "
            f"loss={train_loss:.5f} valid_lwlrap={valid_lwlrap:.6f} lr={learning_rate:.8f}",
            flush=True,
        )
        if valid_lwlrap > best_lwlrap:
            best_lwlrap = valid_lwlrap
            best_epoch = epoch
            _save_checkpoint(
                best_model_path,
                model=model,
                source_checkpoint=source_checkpoint,
                epoch=epoch,
                valid_lwlrap=valid_lwlrap,
                branch=branch,
                args=args,
            )

    _save_checkpoint(
        final_model_path,
        model=model,
        source_checkpoint=source_checkpoint,
        epoch=args.epochs,
        valid_lwlrap=final_lwlrap,
        branch=branch,
        args=args,
    )

    best_checkpoint = _torch_load(best_model_path, map_location=device)
    best_model = _model_from_checkpoint(best_checkpoint, num_classes=len(label_columns), device=device)
    valid_scores_best = _predict_scores(best_model, valid_loader, device=device)
    final_checkpoint = _torch_load(final_model_path, map_location=device)
    final_model = _model_from_checkpoint(final_checkpoint, num_classes=len(label_columns), device=device)
    valid_scores_final = _predict_scores(final_model, valid_loader, device=device)

    del train_loader, dataset, noisy_x, noisy_y, optimizer, criterion, best_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_x, test_fnames = _load_cache(branch, split="test", data_dir=data_dir)
    _assert_order(sample_submission["fname"].astype(str).tolist(), test_fnames, label="test")
    prediction_model = final_model if args.submission_checkpoint == "final" else _model_from_checkpoint(
        best_checkpoint,
        num_classes=len(label_columns),
        device=device,
    )
    test_loader = DataLoader(
        LogmelDataset(test_x, augment=False),
        batch_size=branch.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_scores = _predict_scores(prediction_model, test_loader, device=device)
    submission = pd.DataFrame(np.clip(test_scores, 0.0, 1.0), columns=label_columns)
    submission.insert(0, "fname", sample_submission["fname"].astype(str).to_numpy())
    submission_path = submissions_dir / "small_logmel_cnn.csv"
    write_submission(submission, submission_path, label_columns)

    experiments_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(experiments_dir / "history.csv", index=False)
    np.save(experiments_dir / "valid_scores_best.npy", valid_scores_best.astype(np.float32))
    np.save(experiments_dir / "valid_scores_final.npy", valid_scores_final.astype(np.float32))
    np.save(experiments_dir / "valid_targets.npy", curated_y[valid_indices].astype(np.float32))
    pd.DataFrame({"fname": curated_labels.iloc[valid_indices]["fname"].astype(str)}).to_csv(
        experiments_dir / "valid_fnames.csv",
        index=False,
    )

    result = BranchRunResult(
        branch=branch.name,
        baseline_lwlrap=float(baseline_lwlrap),
        best_lwlrap=float(best_lwlrap),
        final_lwlrap=float(final_lwlrap),
        best_epoch=int(best_epoch),
        submission_path=submission_path,
        best_model_path=best_model_path,
        final_model_path=final_model_path,
    )
    rows = {
        "curated_total": int(len(curated_x)),
        "curated_train": int(len(train_indices)),
        "curated_valid": int(len(valid_indices)),
        "noisy_total": int(len(noisy_indices)),
        "test": int(len(test_x)),
    }
    _write_branch_metadata(
        experiments_dir / "metadata.md",
        branch=branch,
        source_metadata=source_metadata,
        result=result,
        args=args,
        rows=rows,
        steps_per_epoch=steps_per_epoch,
    )
    (experiments_dir / "metadata.json").write_text(
        json.dumps(
            {
                "branch": branch.name,
                "baseline_lwlrap": result.baseline_lwlrap,
                "best_lwlrap": result.best_lwlrap,
                "final_lwlrap": result.final_lwlrap,
                "best_epoch": result.best_epoch,
                "submission_path": str(submission_path),
                "best_model_path": str(best_model_path),
                "final_model_path": str(final_model_path),
                "rows": rows,
                "steps_per_epoch": steps_per_epoch,
                "epochs": args.epochs,
                "lr": args.lr,
                "scheduler": "cosine",
                "noisy_loss_weight": args.noisy_loss_weight,
                "curated_loss_weight": args.curated_loss_weight,
                "gaussian_noise_std": args.gaussian_noise_std,
                "submission_checkpoint": args.submission_checkpoint,
                "source_checkpoint": str(source_checkpoint_path),
                "source_metadata": str(source_metadata_path),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{branch.name}: best_lwlrap={result.best_lwlrap:.6f} "
        f"final_lwlrap={result.final_lwlrap:.6f} wrote={submission_path}",
        flush=True,
    )
    return result


def _select_branches(raw: str) -> list[BranchConfig]:
    if raw == "all":
        return list(BRANCHES)
    by_name = {branch.name: branch for branch in BRANCHES}
    selected: list[BranchConfig] = []
    for name in [part.strip() for part in raw.split(",") if part.strip()]:
        if name not in by_name:
            raise ValueError(f"unknown branch {name!r}; choices are {sorted(by_name)}")
        selected.append(by_name[name])
    if not selected:
        raise ValueError("expected at least one branch")
    return selected


def main() -> None:
    args = _build_parser().parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.min_lr <= 0.0:
        raise ValueError("--min-lr must be positive")
    if not 0.0 < args.noisy_loss_weight <= args.curated_loss_weight:
        raise ValueError("--noisy-loss-weight must be in (0, curated_loss_weight]")
    if args.gaussian_noise_std < 0.0:
        raise ValueError("--gaussian-noise-std must be non-negative")

    results = []
    for branch in _select_branches(args.branches):
        results.append(run_branch(branch, args))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("finetune_mixed_noisy_ok")
    for result in results:
        print(
            f"{result.branch}: baseline={result.baseline_lwlrap:.6f} "
            f"best={result.best_lwlrap:.6f} final={result.final_lwlrap:.6f}"
        )


if __name__ == "__main__":
    main()
