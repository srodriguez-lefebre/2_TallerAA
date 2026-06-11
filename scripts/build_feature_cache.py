from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fat2019.data import load_training_labels
from scripts.fat2019.features import (
    extract_log_mel_stats,
    extract_log_mel_stats_extended,
    read_wav_mono,
)
from scripts.fat2019.submission import read_sample_submission


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cached audio feature matrices.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feature-set",
        choices=("basic", "extended"),
        default="extended",
    )
    parser.add_argument(
        "--splits",
        default="curated,test",
        help="Comma-separated cache splits to build: curated,noisy,test.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def parse_splits(raw_splits: str) -> list[str]:
    splits = [split.strip() for split in raw_splits.split(",") if split.strip()]
    valid_splits = {"curated", "noisy", "test"}
    unknown_splits = sorted(set(splits) - valid_splits)
    if unknown_splits:
        raise ValueError(f"unknown cache split(s): {unknown_splits}")
    if not splits:
        raise ValueError("expected at least one cache split")
    return splits


def cache_path(data_dir: Path, *, split: str, feature_set: str) -> Path:
    if split not in {"curated", "noisy", "test"}:
        raise ValueError(f"unknown cache split: {split}")
    if feature_set not in {"basic", "extended"}:
        raise ValueError(f"unknown feature set: {feature_set}")

    if split == "test":
        suffix = "" if feature_set == "basic" else f"_{feature_set}"
        return data_dir / f"test_logmel_stats{suffix}.npz"

    suffix = "all" if feature_set == "basic" else feature_set
    return data_dir / f"{split}_logmel_stats_{suffix}.npz"


def _extract_features(path: Path, feature_set: str) -> np.ndarray:
    sample_rate, waveform = read_wav_mono(str(path))
    extractor = extract_log_mel_stats_extended if feature_set == "extended" else extract_log_mel_stats
    return extractor(
        waveform,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        fmin=20.0,
        fmax=sample_rate / 2,
    )


def build_matrix(
    data_dir: Path,
    fnames: list[str],
    *,
    feature_set: str,
    output_path: Path,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        print(f"skip existing {output_path}")
        return

    features: list[np.ndarray] = []
    for row_index, fname in enumerate(fnames, start=1):
        features.append(_extract_features(data_dir / fname, feature_set))
        if row_index % 250 == 0:
            print(f"{output_path.name}: {row_index}/{len(fnames)}")
    x = np.vstack(features).astype(np.float32)
    np.savez_compressed(output_path, x=x, fnames=np.array(fnames))
    print(f"wrote {output_path} {x.shape}")


def main() -> None:
    args = _build_parser().parse_args()
    splits = parse_splits(args.splits)

    if "curated" in splits:
        train = load_training_labels(args.data_dir / "train_curated.csv")
        build_matrix(
            args.data_dir,
            train["fname"].astype(str).tolist(),
            feature_set=args.feature_set,
            output_path=cache_path(args.data_dir, split="curated", feature_set=args.feature_set),
            force=args.force,
        )
    if "noisy" in splits:
        noisy = load_training_labels(args.data_dir / "train_noisy.csv")
        build_matrix(
            args.data_dir,
            noisy["fname"].astype(str).tolist(),
            feature_set=args.feature_set,
            output_path=cache_path(args.data_dir, split="noisy", feature_set=args.feature_set),
            force=args.force,
        )
    if "test" in splits:
        sample = read_sample_submission(args.data_dir / "sample_submission.csv")
        build_matrix(
            args.data_dir,
            sample["fname"].astype(str).tolist(),
            feature_set=args.feature_set,
            output_path=cache_path(args.data_dir, split="test", feature_set=args.feature_set),
            force=args.force,
        )


if __name__ == "__main__":
    main()
