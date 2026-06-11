from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


CORRUPT_OR_BAD_LABEL_FILES = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",
}


def split_labels(labels: str) -> list[str]:
    if not isinstance(labels, str) or not labels.strip():
        return []
    return [label.strip() for label in labels.split(",") if label.strip()]


def load_training_labels(path: Path, *, drop_known_bad: bool = True) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required_columns = {"fname", "labels"}
    missing = required_columns - set(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if drop_known_bad:
        labels = labels[~labels["fname"].isin(CORRUPT_OR_BAD_LABEL_FILES)].copy()
    return labels


def class_priors_from_training_labels(labels: pd.DataFrame) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row_labels in labels["labels"]:
        counts.update(split_labels(row_labels))

    total_rows = len(labels)
    if total_rows == 0:
        raise ValueError("cannot calculate priors from an empty labels dataframe")
    return {label: count / total_rows for label, count in counts.items()}
