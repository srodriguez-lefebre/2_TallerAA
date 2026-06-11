from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fat2019.features import extract_log_mel_stats, read_wav_mono
from scripts.fat2019.data import class_priors_from_training_labels, load_training_labels
from scripts.fat2019.submission import (
    build_model_submission,
    build_prior_submission,
    label_columns_from_sample,
    read_sample_submission,
    validate_submission,
    write_submission,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Freesound Audio Tagging 2019 submission.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    parser.add_argument(
        "--strategy",
        choices=("model", "priors"),
        default=None,
        help="Submission strategy. 'priors' is a reusable smoke-test baseline.",
    )
    return parser


def create_prior_submission(data_dir: Path) -> pd.DataFrame:
    sample = read_sample_submission(data_dir / "sample_submission.csv")
    curated = load_training_labels(data_dir / "train_curated.csv")
    noisy = load_training_labels(data_dir / "train_noisy.csv")
    training_labels = pd.concat([curated, noisy], ignore_index=True)
    priors = class_priors_from_training_labels(training_labels)
    return build_prior_submission(sample, priors)


def _test_feature_cache_path(data_dir: Path) -> Path:
    return data_dir / "test_logmel_stats.npz"


def _build_test_features(data_dir: Path, sample: pd.DataFrame) -> np.ndarray:
    cache_path = _test_feature_cache_path(data_dir)
    if cache_path.exists():
        return np.load(cache_path)["x"].astype(np.float32)

    features: list[np.ndarray] = []
    for row_index, fname in enumerate(sample["fname"], start=1):
        sample_rate, waveform = read_wav_mono(str(data_dir / str(fname)))
        features.append(
            extract_log_mel_stats(
                waveform,
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=80,
                fmin=20.0,
                fmax=sample_rate / 2,
            )
        )
        if row_index % 250 == 0:
            print(f"test features {row_index}/{len(sample)}")

    x = np.vstack(features).astype(np.float32)
    np.savez_compressed(cache_path, x=x)
    return x


def create_model_submission(data_dir: Path, model_path: Path) -> pd.DataFrame:
    sample = read_sample_submission(data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample)
    with model_path.open("rb") as model_file:
        payload = pickle.load(model_file)
    model = payload["model"]
    trained_label_columns = payload["label_columns"]
    if trained_label_columns != label_columns:
        raise ValueError("model label columns do not match sample_submission.csv")
    features = _build_test_features(data_dir, sample)
    return build_model_submission(sample, label_columns, model, features)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    strategy = args.strategy or ("model" if args.model is not None else "priors")

    if strategy == "model":
        if args.model is None:
            raise ValueError("--strategy model requires --model")
        submission = create_model_submission(args.data_dir, args.model)
    elif strategy == "priors":
        submission = create_prior_submission(args.data_dir)
    else:
        raise ValueError(f"unsupported strategy: {strategy}")

    sample = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample)
    validate_submission(submission, label_columns, expected_rows=len(sample))
    write_submission(submission, args.output, label_columns)
    print(f"wrote {args.output} with {len(submission)} rows and {len(label_columns)} labels")


if __name__ == "__main__":
    main()
