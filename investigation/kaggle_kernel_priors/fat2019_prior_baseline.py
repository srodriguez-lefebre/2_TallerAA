from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")
BAD_FILES = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",
}


def find_input_dir() -> Path:
    candidates = sorted(INPUT_ROOT.rglob("sample_submission.csv"))
    if candidates:
        input_dir = candidates[0].parent
        print(f"using input dir: {input_dir}")
        return input_dir

    print("sample_submission.csv was not found under /kaggle/input")
    for path in sorted(INPUT_ROOT.rglob("*"))[:200]:
        print(path)
    raise FileNotFoundError("could not locate Freesound 2019 competition files")


def split_labels(labels: str) -> list[str]:
    if not isinstance(labels, str) or not labels.strip():
        return []
    return [label.strip() for label in labels.split(",") if label.strip()]


def class_priors(labels: pd.DataFrame, label_columns: list[str]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row_labels in labels["labels"]:
        counts.update(split_labels(row_labels))
    return {label: counts[label] / len(labels) for label in label_columns}


def main() -> None:
    input_dir = find_input_dir()
    sample = pd.read_csv(input_dir / "sample_submission.csv")
    label_columns = [column for column in sample.columns if column != "fname"]

    curated = pd.read_csv(input_dir / "train_curated.csv")
    noisy = pd.read_csv(input_dir / "train_noisy.csv")
    labels = pd.concat([curated, noisy], ignore_index=True)
    labels = labels[~labels["fname"].isin(BAD_FILES)].copy()

    priors = class_priors(labels, label_columns)
    submission = pd.DataFrame({"fname": sample["fname"].astype(str)})
    for label in label_columns:
        submission[label] = float(priors.get(label, 0.0))

    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH} with shape {submission.shape}")


if __name__ == "__main__":
    main()
