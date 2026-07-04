from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
INVESTIGATION_DIR = REPO_ROOT / "investigation"
EXPERIMENT_DIR = INVESTIGATION_DIR / "noisy_finetune_20260704"
SOURCE_STATES_DIR = EXPERIMENT_DIR / "source_states"
RUNS_DIR = EXPERIMENT_DIR / "runs"
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class BranchConfig:
    name: str
    source_run: str
    ensemble_weight: float
    n_mels: int
    frames: int
    cache_tag: str | None
    batch_size: int

    @property
    def source_checkpoint(self) -> Path:
        return (
            INVESTIGATION_DIR
            / "models"
            / self.source_run
            / "small_logmel_cnn_best.pt"
        )

    @property
    def source_metadata(self) -> Path:
        return (
            INVESTIGATION_DIR
            / "experiments"
            / self.source_run
            / "small_logmel_cnn_metadata.json"
        )

    @property
    def source_submission(self) -> Path:
        return (
            INVESTIGATION_DIR
            / "submissions"
            / self.source_run
            / "small_logmel_cnn.csv"
        )

    @property
    def copied_checkpoint(self) -> Path:
        return SOURCE_STATES_DIR / self.name / "small_logmel_cnn_best.pt"

    @property
    def copied_metadata(self) -> Path:
        return SOURCE_STATES_DIR / self.name / "small_logmel_cnn_metadata.json"

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.name


BRANCHES: tuple[BranchConfig, ...] = (
    BranchConfig(
        name="separable_headsep",
        source_run="parallel100_20260702_separable_headsep_e100_seed42",
        ensemble_weight=0.25,
        n_mels=128,
        frames=512,
        cache_tag=None,
        batch_size=24,
    ),
    BranchConfig(
        name="globalmel_sep_temporal",
        source_run="parallel100_20260702_globalmel_sep_temporal_e100_seed42",
        ensemble_weight=0.375,
        n_mels=128,
        frames=512,
        cache_tag="globalmel",
        batch_size=24,
    ),
    BranchConfig(
        name="sep_temporal_f1024",
        source_run="parallel100_20260702_sep_temporal_f1024_e100_seed42",
        ensemble_weight=0.375,
        n_mels=128,
        frames=1024,
        cache_tag=None,
        batch_size=12,
    ),
)


FINETUNE_EPOCHS = 30
FINETUNE_LR = 1e-4
MIN_LR = 1e-6
NOISY_LOSS_WEIGHT = 0.30
CURATED_LOSS_WEIGHT = 1.00
VALIDATION_TEST_SIZE = 0.20
SEED = 42
GAUSSIAN_NOISE_STD = 0.015
NUM_WORKERS = 2

