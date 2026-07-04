from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_config import BRANCHES, DATA_DIR, EXPERIMENT_DIR, RUNS_DIR

sys.path.insert(0, str(DATA_DIR.parent / "investigation"))

from scripts.build_logmel_image_cache import logmel_image_cache_path  # noqa: E402
from scripts.fat2019.data import load_training_labels  # noqa: E402
from scripts.fat2019.submission import (  # noqa: E402
    label_columns_from_sample,
    read_sample_submission,
    validate_submission,
)


SCRIPT_NAMES = [
    "prepare_experiment.py",
    "build_noisy_caches.py",
    "build_f1024_memmap_caches.py",
    "finetune_mixed_noisy.py",
    "evaluate_and_blend.py",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate noisy fine-tune experiment artifacts.")
    parser.add_argument("--static", action="store_true")
    return parser


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _validate_script_imports() -> None:
    scripts_dir = Path(__file__).resolve().parent
    for script_name in SCRIPT_NAMES:
        path = scripts_dir / script_name
        _assert_exists(path)
        module_name = f"_validate_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def validate_static() -> None:
    _assert_exists(EXPERIMENT_DIR / "README.md")
    _assert_exists(EXPERIMENT_DIR / "configs.md")
    for branch in BRANCHES:
        _assert_exists(branch.source_checkpoint)
        _assert_exists(branch.source_metadata)
        _assert_exists(branch.source_submission)
        _assert_exists(
            logmel_image_cache_path(
                DATA_DIR,
                split="curated",
                n_mels=branch.n_mels,
                frames=branch.frames,
                tag=branch.cache_tag,
            )
        )
        _assert_exists(
            logmel_image_cache_path(
                DATA_DIR,
                split="test",
                n_mels=branch.n_mels,
                frames=branch.frames,
                tag=branch.cache_tag,
            )
        )
    _validate_script_imports()
    print("static_validation_ok")


def validate_caches() -> None:
    noisy_rows = len(load_training_labels(DATA_DIR / "train_noisy.csv"))
    for branch in BRANCHES:
        if branch.frames == 1024 and branch.cache_tag is None:
            cache_path = DATA_DIR / f"noisy_logmel_image_m{branch.n_mels}_f{branch.frames}_x.npy"
            _assert_exists(cache_path)
            cache = np.load(cache_path, mmap_mode="r")
            if cache.shape != (noisy_rows, branch.n_mels, branch.frames):
                raise ValueError(f"unexpected cache shape for {cache_path}: {cache.shape}")
            continue
        cache_path = logmel_image_cache_path(
            DATA_DIR,
            split="noisy",
            n_mels=branch.n_mels,
            frames=branch.frames,
            tag=branch.cache_tag,
        )
        _assert_exists(cache_path)
        cache = np.load(cache_path, allow_pickle=False)
        if cache["x"].shape != (noisy_rows, branch.n_mels, branch.frames):
            raise ValueError(f"unexpected cache shape for {cache_path}: {cache['x'].shape}")


def validate_runs() -> None:
    sample = read_sample_submission(DATA_DIR / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample)
    for branch in BRANCHES:
        for path in (
            branch.run_dir / "models" / "small_logmel_cnn_best.pt",
            branch.run_dir / "models" / "small_logmel_cnn_final.pt",
            branch.run_dir / "experiments" / "history.csv",
            branch.run_dir / "experiments" / "metadata.md",
            branch.run_dir / "experiments" / "metadata.json",
            branch.run_dir / "experiments" / "valid_scores_best.npy",
            branch.run_dir / "experiments" / "valid_scores_final.npy",
            branch.run_dir / "experiments" / "valid_targets.npy",
            branch.run_dir / "submissions" / "small_logmel_cnn.csv",
        ):
            _assert_exists(path)
        submission = pd.read_csv(branch.run_dir / "submissions" / "small_logmel_cnn.csv")
        validate_submission(submission, label_columns, expected_rows=len(sample))

    ensemble_submission = RUNS_DIR / "ensemble" / "submission.csv"
    _assert_exists(ensemble_submission)
    validate_submission(pd.read_csv(ensemble_submission), label_columns, expected_rows=len(sample))
    _assert_exists(EXPERIMENT_DIR / "results.md")


def main() -> None:
    args = _build_parser().parse_args()
    validate_static()
    if args.static:
        return
    validate_caches()
    validate_runs()
    print("noisy_finetune_experiment_validation_ok")


if __name__ == "__main__":
    main()
