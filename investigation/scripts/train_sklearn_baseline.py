from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fat2019.data import load_training_labels
from scripts.fat2019.features import extract_log_mel_stats, read_wav_mono
from scripts.fat2019.labels import dataframe_to_multihot
from scripts.fat2019.metrics import calculate_overall_lwlrap
from scripts.fat2019.submission import label_columns_from_sample, read_sample_submission


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fast sklearn baseline on curated log-mel statistics.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _feature_cache_path(data_dir: Path, max_files: int | None) -> Path:
    suffix = "all" if max_files is None else str(max_files)
    return data_dir / f"curated_logmel_stats_{suffix}.npz"


def build_feature_matrix(
    labels: pd.DataFrame,
    *,
    data_dir: Path,
    max_files: int | None,
) -> tuple[np.ndarray, list[str]]:
    if max_files is not None:
        labels = labels.head(max_files).copy()

    cache_path = _feature_cache_path(data_dir, max_files)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return cached["x"].astype(np.float32), cached["fnames"].astype(str).tolist()

    features: list[np.ndarray] = []
    fnames: list[str] = []
    for row_index, fname in enumerate(labels["fname"], start=1):
        wav_path = data_dir / str(fname)
        sample_rate, waveform = read_wav_mono(str(wav_path))
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
        fnames.append(str(fname))
        if row_index % 250 == 0:
            print(f"features {row_index}/{len(labels)}")

    x = np.vstack(features).astype(np.float32)
    np.savez_compressed(cache_path, x=x, fnames=np.array(fnames))
    return x, fnames


def main() -> None:
    args = _build_parser().parse_args()

    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    labels = load_training_labels(args.data_dir / "train_curated.csv")
    if args.max_files is not None:
        labels = labels.head(args.max_files).copy()

    x, _ = build_feature_matrix(labels, data_dir=args.data_dir, max_files=args.max_files)
    y = dataframe_to_multihot(labels, label_columns)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
    )
    classifier = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=500,
                solver="liblinear",
            ),
            n_jobs=-1,
        ),
    )
    classifier.fit(x_train, y_train)
    valid_scores = classifier.predict_proba(x_valid)
    valid_lwlrap = calculate_overall_lwlrap(y_valid, valid_scores)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "sklearn_logmel_stats.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(
            {
                "model": classifier,
                "label_columns": label_columns,
                "valid_lwlrap": valid_lwlrap,
            },
            model_file,
        )

    print(f"valid_lwlrap={valid_lwlrap:.6f}")
    print(f"saved {model_path}")


if __name__ == "__main__":
    main()
