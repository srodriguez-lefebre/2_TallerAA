from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BranchConfig:
    name: str
    source_run: str
    ensemble_weight: float
    n_mels: int
    frames: int
    cache_tag: str | None
    batch_size: int
    base_epochs: int
    base_lr: float
    base_weight_decay: float
    architecture: str
    activation: str
    initializer: str
    optimizer: str
    scheduler: str
    lr_milestones: tuple[int, ...]
    head_hidden: int
    head_dropout: float
    block_dropout: float = 0.0
    time_reverse_probability: float = 0.0
    contrast_strength: float = 0.0

    def source_checkpoint(self, work_dir: Path) -> Path:
        return work_dir / "models" / self.source_run / "small_logmel_cnn_best.pt"

    def source_metadata(self, work_dir: Path) -> Path:
        return work_dir / "experiments" / self.source_run / "small_logmel_cnn_metadata.json"

    def run_dir(self, work_dir: Path) -> Path:
        return work_dir / "runs" / self.name


BRANCHES: tuple[BranchConfig, ...] = (
    BranchConfig(
        name="separable_headsep",
        source_run="separable_headsep_e100_seed42",
        ensemble_weight=0.25,
        n_mels=128,
        frames=512,
        cache_tag=None,
        batch_size=24,
        base_epochs=100,
        base_lr=1e-3,
        base_weight_decay=1e-4,
        architecture="separable_residual",
        activation="relu",
        initializer="he_normal",
        optimizer="adam",
        scheduler="multistep",
        lr_milestones=(27, 36, 43, 49, 52),
        head_hidden=256,
        head_dropout=0.30,
    ),
    BranchConfig(
        name="globalmel_sep_temporal",
        source_run="globalmel_sep_temporal_e100_seed42",
        ensemble_weight=0.375,
        n_mels=128,
        frames=512,
        cache_tag="globalmel",
        batch_size=24,
        base_epochs=100,
        base_lr=1e-3,
        base_weight_decay=1e-4,
        architecture="separable_temporal_bigru",
        activation="silu",
        initializer="he_normal",
        optimizer="adamw",
        scheduler="multistep",
        lr_milestones=(25, 39),
        head_hidden=0,
        head_dropout=0.30,
    ),
    BranchConfig(
        name="sep_temporal_f1024",
        source_run="sep_temporal_f1024_e100_seed42",
        ensemble_weight=0.375,
        n_mels=128,
        frames=1024,
        cache_tag=None,
        batch_size=12,
        base_epochs=100,
        base_lr=1e-3,
        base_weight_decay=1e-4,
        architecture="separable_temporal_bigru",
        activation="silu",
        initializer="he_normal",
        optimizer="adamw",
        scheduler="multistep",
        lr_milestones=(19, 25),
        head_hidden=0,
        head_dropout=0.30,
    ),
)


@dataclass(frozen=True)
class FineTuneConfig:
    epochs: int = 30
    lr: float = 1e-4
    min_lr: float = 1e-6
    noisy_loss_weight: float = 0.30
    curated_loss_weight: float = 1.00
    validation_test_size: float = 0.20
    seed: int = 42
    gaussian_noise_std: float = 0.015
    num_workers: int = 2
    submission_checkpoint: str = "final"
    augmentations: tuple[str, ...] = field(
        default=("time shift", "time mask", "frequency mask", "gaussian noise")
    )


FINE_TUNE = FineTuneConfig()

