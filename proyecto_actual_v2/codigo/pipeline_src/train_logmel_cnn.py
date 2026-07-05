from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_logmel_image_cache import logmel_image_cache_path
from fat2019.data import load_training_labels
from fat2019.labels import dataframe_to_multihot
from fat2019.metrics import calculate_overall_lwlrap
from fat2019.neural_helpers import (
    compute_pos_weight,
    make_train_valid_indices,
    sigmoid_numpy,
)
from fat2019.submission import (
    label_columns_from_sample,
    read_sample_submission,
    write_submission,
)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_loss: float
    valid_lwlrap: float | None
    learning_rate: float


class LogmelDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray | None = None,
        *,
        augment: bool = False,
        time_reverse_probability: float = 0.0,
        contrast_strength: float = 0.0,
    ) -> None:
        self._images = images
        self._targets = targets
        self._augment = augment
        self._time_reverse_probability = time_reverse_probability
        self._contrast_strength = contrast_strength

    def __len__(self) -> int:
        return int(len(self._images))

    def __getitem__(self, index: int):
        image = torch.from_numpy(self._images[index].astype(np.float32, copy=False)).unsqueeze(0)
        if self._augment:
            image = _augment_image(
                image,
                time_reverse_probability=self._time_reverse_probability,
                contrast_strength=self._contrast_strength,
            )
        if self._targets is None:
            return image
        target = torch.from_numpy(self._targets[index].astype(np.float32, copy=False))
        return image, target


def _scale_contrast(image: torch.Tensor, *, factor: float) -> torch.Tensor:
    mean = image.mean()
    return (image - mean) * factor + mean


def _augment_image(
    image: torch.Tensor,
    *,
    apply_spec_augment: bool = True,
    time_reverse_probability: float = 0.0,
    contrast_strength: float = 0.0,
) -> torch.Tensor:
    if time_reverse_probability > 0.0 and torch.rand(()) < time_reverse_probability:
        image = image.flip(dims=(2,))
    if contrast_strength > 0.0:
        factor = float(torch.empty(()).uniform_(1.0 - contrast_strength, 1.0 + contrast_strength))
        image = _scale_contrast(image, factor=factor)
    if apply_spec_augment:
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
    return image


class HiddenLinear(nn.Linear):
    pass


class TemporalBigruHead(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        hidden_size: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_channels,
            hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected CNN features with shape (batch, channels, freq, time), got {x.shape}")
        sequence = x.mean(dim=2).transpose(1, 2)
        outputs, _hidden = self.gru(sequence)
        pooled = torch.cat([outputs.mean(dim=1), outputs.amax(dim=1)], dim=1)
        return self.classifier(self.dropout(pooled))


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: str,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _activation_layer(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SeparableResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: str,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, activation=activation),
            DepthwiseSeparableConv(out_channels, out_channels, activation=activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = _activation_layer(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.residual(x))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.layers(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class SmallLogmelCnn(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        architecture: str = "standard",
        activation: str = "silu",
        block_dropout: float = 0.0,
        head_hidden: int = 0,
        head_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        temporal_head = False
        if architecture == "standard":
            feature_channels = 192
            self.features = nn.Sequential(
                _conv_block(1, 32, activation=activation, dropout=block_dropout),
                nn.MaxPool2d(2),
                _conv_block(32, 64, activation=activation, dropout=block_dropout),
                nn.MaxPool2d(2),
                _conv_block(64, 128, activation=activation, dropout=block_dropout),
                nn.MaxPool2d(2),
                _conv_block(128, 192, activation=activation, dropout=block_dropout),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif architecture == "separable_residual":
            feature_channels = 512
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                _activation_layer(activation),
                SeparableResidualBlock(64, 128, activation=activation),
                SeparableResidualBlock(128, 256, activation=activation),
                SeparableResidualBlock(256, 384, activation=activation),
                DepthwiseSeparableConv(384, feature_channels, activation=activation),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif architecture == "separable_residual_se":
            feature_channels = 512
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                _activation_layer(activation),
                SeparableResidualBlock(64, 128, activation=activation),
                SqueezeExcitation(128),
                SeparableResidualBlock(128, 256, activation=activation),
                SqueezeExcitation(256),
                SeparableResidualBlock(256, 384, activation=activation),
                SqueezeExcitation(384),
                DepthwiseSeparableConv(384, feature_channels, activation=activation),
                SqueezeExcitation(feature_channels),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif architecture == "separable_temporal_bigru":
            feature_channels = 512
            temporal_head = True
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                _activation_layer(activation),
                SeparableResidualBlock(64, 128, activation=activation),
                SeparableResidualBlock(128, 256, activation=activation),
                SeparableResidualBlock(256, 384, activation=activation),
                DepthwiseSeparableConv(384, feature_channels, activation=activation),
            )
        elif architecture == "temporal_bigru":
            feature_channels = 128
            temporal_head = True
            self.features = nn.Sequential(
                _conv_block(1, 32, activation=activation, dropout=block_dropout),
                nn.MaxPool2d(kernel_size=(2, 2)),
                _conv_block(32, 64, activation=activation, dropout=block_dropout),
                nn.MaxPool2d(kernel_size=(2, 2)),
                _conv_block(64, feature_channels, activation=activation, dropout=block_dropout),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
        else:
            raise ValueError(f"unknown architecture: {architecture}")

        if temporal_head:
            self.classifier = TemporalBigruHead(
                input_channels=feature_channels,
                hidden_size=64,
                num_classes=num_classes,
                dropout=head_dropout,
            )
        elif head_hidden > 0:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                HiddenLinear(feature_channels, head_hidden, bias=False),
                nn.BatchNorm1d(head_hidden),
                _activation_layer(activation),
                nn.Dropout(p=head_dropout),
                nn.Linear(head_hidden, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=head_dropout),
                nn.Linear(feature_channels, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def apply_he_initialization(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, HiddenLinear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _activation_layer(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"unknown activation: {name}")


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    activation: str,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        _activation_layer(activation),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        _activation_layer(activation),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout2d(p=dropout))
    return nn.Sequential(*layers)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_name: str,
    epochs: int,
    plateau_patience: int,
    plateau_factor: float,
    lr_milestones: list[int],
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )
    if scheduler_name == "multistep":
        if not lr_milestones:
            raise ValueError("multistep scheduler requires at least one LR milestone")
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=lr_milestones,
            gamma=plateau_factor,
        )
    raise ValueError(f"unknown scheduler: {scheduler_name}")


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    raise ValueError(f"unknown optimizer: {optimizer_name}")


def parse_lr_milestones(raw_milestones: str) -> list[int]:
    if not raw_milestones.strip():
        return []
    milestones = [int(value.strip()) for value in raw_milestones.split(",") if value.strip()]
    if any(milestone <= 0 for milestone in milestones):
        raise ValueError("LR milestones must be positive epoch numbers")
    if milestones != sorted(set(milestones)):
        raise ValueError("LR milestones must be unique and sorted")
    return milestones


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a small CNN on fixed log-mel images.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/logmel_cnn"))
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions/logmel_cnn"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments/logmel_cnn"))
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--cache-tag", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--initializer", choices=("default", "he_normal"), default="default")
    parser.add_argument(
        "--architecture",
        choices=(
            "standard",
            "separable_residual",
            "separable_residual_se",
            "temporal_bigru",
            "separable_temporal_bigru",
        ),
        default="standard",
    )
    parser.add_argument("--activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--block-dropout", type=float, default=0.0)
    parser.add_argument("--head-hidden", type=int, default=0)
    parser.add_argument("--head-dropout", type=float, default=0.35)
    parser.add_argument("--time-reverse-probability", type=float, default=0.0)
    parser.add_argument("--contrast-strength", type=float, default=0.0)
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "plateau", "multistep"),
        default="cosine",
    )
    parser.add_argument("--lr-milestones", default="")
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving validation epochs; zero disables it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="Train on all curated rows and skip holdout validation.",
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
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            output = model(images.to(device, non_blocking=True))
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
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    labels = load_training_labels(args.data_dir / "train_curated.csv")
    y = dataframe_to_multihot(labels, label_columns).astype(np.float32)

    train_cache = np.load(
        logmel_image_cache_path(
            args.data_dir,
            split="curated",
            n_mels=args.n_mels,
            frames=args.frames,
            tag=args.cache_tag,
        ),
        allow_pickle=False,
    )
    test_cache = np.load(
        logmel_image_cache_path(
            args.data_dir,
            split="test",
            n_mels=args.n_mels,
            frames=args.frames,
            tag=args.cache_tag,
        ),
        allow_pickle=False,
    )
    x = train_cache["x"]
    x_test = test_cache["x"]

    if args.max_train is not None:
        x = x[: args.max_train]
        y = y[: args.max_train]
        labels = labels.head(args.max_train).copy()

    if len(x) != len(y):
        raise ValueError(f"feature/label row mismatch: {len(x)} vs {len(y)}")

    train_indices, valid_indices = make_train_valid_indices(
        num_rows=len(x),
        test_size=args.test_size,
        seed=args.seed,
        full_train=args.full_train,
    )

    train_loader = DataLoader(
        LogmelDataset(
            x[train_indices],
            y[train_indices],
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
            LogmelDataset(x[valid_indices], y[valid_indices], augment=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    test_loader = DataLoader(
        LogmelDataset(x_test, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if not 0.0 <= args.block_dropout < 1.0:
        raise ValueError("block_dropout must be in [0, 1)")
    if not 0.0 <= args.head_dropout < 1.0:
        raise ValueError("head_dropout must be in [0, 1)")
    if args.head_hidden < 0:
        raise ValueError("head_hidden must be non-negative")
    if not 0.0 <= args.time_reverse_probability <= 1.0:
        raise ValueError("time_reverse_probability must be in [0, 1]")
    if not 0.0 <= args.contrast_strength <= 1.0:
        raise ValueError("contrast_strength must be in [0, 1]")
    lr_milestones = parse_lr_milestones(args.lr_milestones)

    model = SmallLogmelCnn(
        num_classes=len(label_columns),
        architecture=args.architecture,
        activation=args.activation,
        block_dropout=args.block_dropout,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    )
    if args.initializer == "he_normal":
        model.apply(apply_he_initialization)
    model = model.to(device)
    pos_weight = torch.from_numpy(compute_pos_weight(y[train_indices])).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        lr_milestones=lr_milestones,
        min_lr=args.min_lr,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.submissions_dir.mkdir(parents=True, exist_ok=True)
    args.experiments_dir.mkdir(parents=True, exist_ok=True)

    best_lwlrap = -1.0
    best_epoch = 0
    best_model_path = args.models_dir / "small_logmel_cnn_best.pt"
    history: list[EpochResult] = []
    non_improving_epochs = 0
    early_stopped = False
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
            history.append(
                EpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    valid_lwlrap=None,
                    learning_rate=learning_rate,
                )
            )
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(-train_loss)
            else:
                scheduler.step()
            print(f"epoch {epoch}: loss={train_loss:.5f} lr={learning_rate:.8f}")
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "valid_lwlrap": None,
                    "label_columns": label_columns,
                    "n_mels": args.n_mels,
                    "frames": args.frames,
                    "cache_tag": args.cache_tag,
                    "full_train": True,
                    "initializer": args.initializer,
                    "architecture": args.architecture,
                    "activation": args.activation,
                    "block_dropout": args.block_dropout,
                    "head_hidden": args.head_hidden,
                    "head_dropout": args.head_dropout,
                    "time_reverse_probability": args.time_reverse_probability,
                    "contrast_strength": args.contrast_strength,
                },
                best_model_path,
            )
            continue

        valid_logits = _predict_logits(model, valid_loader, device=device)
        valid_scores = sigmoid_numpy(valid_logits)
        valid_lwlrap = calculate_overall_lwlrap(y[valid_indices], valid_scores)
        history.append(
            EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                valid_lwlrap=valid_lwlrap,
                learning_rate=learning_rate,
            )
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_lwlrap)
        else:
            scheduler.step()
        print(
            f"epoch {epoch}: loss={train_loss:.5f} "
            f"valid_lwlrap={valid_lwlrap:.6f} lr={learning_rate:.8f}"
        )
        if valid_lwlrap > best_lwlrap:
            best_lwlrap = valid_lwlrap
            best_epoch = epoch
            non_improving_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "valid_lwlrap": valid_lwlrap,
                    "label_columns": label_columns,
                    "n_mels": args.n_mels,
                    "frames": args.frames,
                    "cache_tag": args.cache_tag,
                    "full_train": False,
                    "initializer": args.initializer,
                    "architecture": args.architecture,
                    "activation": args.activation,
                    "block_dropout": args.block_dropout,
                    "head_hidden": args.head_hidden,
                    "head_dropout": args.head_dropout,
                    "time_reverse_probability": args.time_reverse_probability,
                    "contrast_strength": args.contrast_strength,
                },
                best_model_path,
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

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_scores = sigmoid_numpy(_predict_logits(model, test_loader, device=device))
    submission = pd.DataFrame(np.clip(test_scores, 0.0, 1.0), columns=label_columns)
    submission.insert(0, "fname", sample_submission["fname"].astype(str).to_numpy())
    submission_path = args.submissions_dir / "small_logmel_cnn.csv"
    write_submission(submission, submission_path, label_columns)

    history_path = args.experiments_dir / "small_logmel_cnn_history.csv"
    pd.DataFrame([result.__dict__ for result in history]).to_csv(history_path, index=False)
    metadata_path = args.experiments_dir / "small_logmel_cnn_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "best_lwlrap": None if args.full_train else float(best_lwlrap),
                "best_epoch": int(best_epoch),
                "device": str(device),
                "rows": int(len(x)),
                "train_rows": int(len(train_indices)),
                "valid_rows": int(len(valid_indices)),
                "labels": len(label_columns),
                "n_mels": args.n_mels,
                "frames": args.frames,
                "cache_tag": args.cache_tag,
                "seed": args.seed,
                "test_size": args.test_size,
                "full_train": args.full_train,
                "initializer": args.initializer,
                "architecture": args.architecture,
                "activation": args.activation,
                "block_dropout": args.block_dropout,
                "head_hidden": args.head_hidden,
                "head_dropout": args.head_dropout,
                "time_reverse_probability": args.time_reverse_probability,
                "contrast_strength": args.contrast_strength,
                "optimizer": args.optimizer,
                "scheduler": args.scheduler,
                "lr_milestones": lr_milestones,
                "plateau_patience": args.plateau_patience,
                "plateau_factor": args.plateau_factor,
                "min_lr": args.min_lr,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopped": early_stopped,
                "submission_path": str(submission_path),
                "model_path": str(best_model_path),
            },
            indent=2,
        )
        + "\n"
    )
    if args.full_train:
        print(f"full_train_epoch={checkpoint['epoch']}")
    else:
        print(f"best_lwlrap={best_lwlrap:.6f} epoch={checkpoint['epoch']}")
    print(f"wrote {submission_path}")


if __name__ == "__main__":
    main()
