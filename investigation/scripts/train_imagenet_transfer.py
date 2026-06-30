from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_logmel_image_cache import logmel_image_cache_path
from scripts.fat2019.data import load_training_labels
from scripts.fat2019.labels import dataframe_to_multihot
from scripts.fat2019.metrics import calculate_overall_lwlrap
from scripts.fat2019.neural_helpers import (
    compute_pos_weight,
    make_train_valid_indices,
    sigmoid_numpy,
)
from scripts.fat2019.submission import (
    label_columns_from_sample,
    read_sample_submission,
    write_submission,
)
from scripts.train_logmel_cnn import LogmelDataset, parse_lr_milestones


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    valid_lwlrap: float | None
    learning_rate: float


class FrozenResNetTransfer(nn.Module):
    def __init__(self, backbone: nn.Module, *, feature_count: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(feature_count, num_classes),
        )

    def train(self, mode: bool = True) -> FrozenResNetTransfer:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.classifier(features)


def prepare_imagenet_batch(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(f"expected NCHW one-channel images, got {tuple(images.shape)}")
    scaled = images.clamp(-3.0, 3.0).add(3.0).div(6.0)
    resized = functional.interpolate(
        scaled,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    rgb = resized.repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb - mean) / std


def build_resnet50_transfer(
    *,
    num_classes: int,
    weights: ResNet50_Weights | None,
) -> FrozenResNetTransfer:
    base_model = resnet50(weights=weights)
    feature_count = int(base_model.fc.in_features)
    base_model.fc = nn.Identity()
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    return FrozenResNetTransfer(
        base_model,
        feature_count=feature_count,
        num_classes=num_classes,
    )


def build_transfer_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    full_train: bool,
    epochs: int,
    lr_milestones: list[int],
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    if not full_train:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=1,
            factor=0.5,
            min_lr=1e-5,
        )
    if lr_milestones:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_milestones,
            gamma=0.5,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen ImageNet ResNet50 head on FAT2019 log-mel images."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/imagenet_transfer"))
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("submissions/imagenet_transfer"),
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments/imagenet_transfer"),
    )
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--lr-milestones", default="")
    parser.add_argument("--time-reverse-probability", type=float, default=0.0)
    parser.add_argument("--contrast-strength", type=float, default=0.0)
    parser.add_argument("--full-train", action="store_true")
    parser.add_argument(
        "--weights",
        choices=("imagenet", "none"),
        default="imagenet",
    )
    return parser


def _predict_logits(
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
            prepared = prepare_imagenet_batch(images.to(device, non_blocking=True))
            output = model(prepared)
            logits.append(output.detach().cpu().numpy())
    return np.vstack(logits)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for images, targets in loader:
        prepared = prepare_imagenet_batch(images.to(device, non_blocking=True))
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(prepared)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        rows = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows
    return total_loss / max(1, total_rows)


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if not 0.0 <= args.time_reverse_probability <= 1.0:
        raise ValueError("time_reverse_probability must be in [0, 1]")
    if not 0.0 <= args.contrast_strength <= 1.0:
        raise ValueError("contrast_strength must be in [0, 1]")
    lr_milestones = parse_lr_milestones(args.lr_milestones)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    labels = load_training_labels(args.data_dir / "train_curated.csv")
    targets = dataframe_to_multihot(labels, label_columns).astype(np.float32)
    train_cache = np.load(
        logmel_image_cache_path(
            args.data_dir,
            split="curated",
            n_mels=args.n_mels,
            frames=args.frames,
        ),
        allow_pickle=False,
    )
    test_cache = np.load(
        logmel_image_cache_path(
            args.data_dir,
            split="test",
            n_mels=args.n_mels,
            frames=args.frames,
        ),
        allow_pickle=False,
    )
    images = train_cache["x"]
    test_images = test_cache["x"]
    if len(images) != len(targets):
        raise ValueError(f"feature/label row mismatch: {len(images)} vs {len(targets)}")

    train_indices, valid_indices = make_train_valid_indices(
        num_rows=len(images),
        test_size=args.test_size,
        seed=args.seed,
        full_train=args.full_train,
    )
    train_loader = DataLoader(
        LogmelDataset(
            images[train_indices],
            targets[train_indices],
            augment=True,
            time_reverse_probability=args.time_reverse_probability,
            contrast_strength=args.contrast_strength,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = None
    if valid_indices.size > 0:
        valid_loader = DataLoader(
            LogmelDataset(images[valid_indices], targets[valid_indices]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    test_loader = DataLoader(
        LogmelDataset(test_images),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    weights = ResNet50_Weights.DEFAULT if args.weights == "imagenet" else None
    model = build_resnet50_transfer(
        num_classes=len(label_columns),
        weights=weights,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(compute_pos_weight(targets[train_indices])).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_transfer_scheduler(
        optimizer,
        full_train=args.full_train,
        epochs=args.epochs,
        lr_milestones=lr_milestones,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.submissions_dir.mkdir(parents=True, exist_ok=True)
    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "resnet50_transfer_best.pt"
    best_lwlrap = -1.0
    best_epoch = 0
    non_improving_epochs = 0
    early_stopped = False
    history: list[EpochResult] = []

    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        if valid_loader is None:
            valid_lwlrap = None
            scheduler.step()
            improved = True
        else:
            valid_scores = sigmoid_numpy(_predict_logits(model, valid_loader, device=device))
            valid_lwlrap = calculate_overall_lwlrap(targets[valid_indices], valid_scores)
            scheduler.step(valid_lwlrap)
            improved = valid_lwlrap > best_lwlrap
        history.append(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                valid_lwlrap=valid_lwlrap,
                learning_rate=learning_rate,
            )
        )
        metric_text = "" if valid_lwlrap is None else f" valid_lwlrap={valid_lwlrap:.6f}"
        print(f"epoch {epoch}: loss={train_loss:.5f}{metric_text} lr={learning_rate:.8f}")

        if improved:
            best_epoch = epoch
            if valid_lwlrap is not None:
                best_lwlrap = valid_lwlrap
            non_improving_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "valid_lwlrap": valid_lwlrap,
                    "label_columns": label_columns,
                    "n_mels": args.n_mels,
                    "frames": args.frames,
                    "weights": args.weights,
                    "full_train": args.full_train,
                },
                model_path,
            )
        else:
            non_improving_epochs += 1
            if (
                args.early_stopping_patience > 0
                and non_improving_epochs >= args.early_stopping_patience
            ):
                early_stopped = True
                print(f"early_stopping epoch={epoch} best_epoch={best_epoch}")
                break

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_scores = sigmoid_numpy(_predict_logits(model, test_loader, device=device))
    submission = pd.DataFrame(np.clip(test_scores, 0.0, 1.0), columns=label_columns)
    submission.insert(0, "fname", sample_submission["fname"].astype(str).to_numpy())
    submission_path = args.submissions_dir / "resnet50_transfer.csv"
    write_submission(submission, submission_path, label_columns)

    pd.DataFrame([result.__dict__ for result in history]).to_csv(
        args.experiments_dir / "resnet50_transfer_history.csv",
        index=False,
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "best_lwlrap": None if args.full_train else float(best_lwlrap),
        "best_epoch": int(best_epoch),
        "device": str(device),
        "rows": int(len(images)),
        "train_rows": int(len(train_indices)),
        "valid_rows": int(len(valid_indices)),
        "labels": len(label_columns),
        "seed": args.seed,
        "full_train": args.full_train,
        "weights": args.weights,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "time_reverse_probability": args.time_reverse_probability,
        "contrast_strength": args.contrast_strength,
        "lr_milestones": lr_milestones,
        "early_stopped": early_stopped,
        "model_path": str(model_path),
        "submission_path": str(submission_path),
    }
    (args.experiments_dir / "resnet50_transfer_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if args.full_train:
        print(f"full_train_epoch={checkpoint['epoch']}")
    else:
        print(f"best_lwlrap={best_lwlrap:.6f} epoch={checkpoint['epoch']}")
    print(f"wrote {submission_path}")


if __name__ == "__main__":
    main()
