from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class ProbabilityModel(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for feature rows."""


def label_columns_from_sample(sample_submission: pd.DataFrame) -> list[str]:
    if "fname" not in sample_submission.columns:
        raise ValueError("sample submission must include fname")
    return [column for column in sample_submission.columns if column != "fname"]


def validate_submission(
    submission: pd.DataFrame,
    label_columns: list[str],
    *,
    expected_rows: int | None = None,
) -> None:
    expected_columns = ["fname", *label_columns]
    if list(submission.columns) != expected_columns:
        raise ValueError(
            "submission columns do not match expected columns/order: "
            f"expected {expected_columns[:4]}..., got {list(submission.columns)[:4]}..."
        )
    if expected_rows is not None and len(submission) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(submission)}")
    if submission["fname"].isna().any():
        raise ValueError("submission contains empty fname values")

    probabilities = submission[label_columns]
    if probabilities.isna().any().any():
        raise ValueError("submission probabilities contain NaN values")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any().any():
        raise ValueError("submission probabilities must be in [0, 1]")


def build_prior_submission(
    sample_submission: pd.DataFrame,
    class_priors: dict[str, float],
) -> pd.DataFrame:
    label_columns = label_columns_from_sample(sample_submission)
    submission = pd.DataFrame({"fname": sample_submission["fname"].astype(str)})
    for label in label_columns:
        submission[label] = float(class_priors.get(label, 0.0))
    validate_submission(submission, label_columns, expected_rows=len(sample_submission))
    return submission


def build_model_submission(
    sample_submission: pd.DataFrame,
    label_columns: list[str],
    model: ProbabilityModel,
    features: np.ndarray,
) -> pd.DataFrame:
    scores = np.asarray(model.predict_proba(features), dtype=np.float64)
    expected_shape = (len(sample_submission), len(label_columns))
    if scores.shape != expected_shape:
        raise ValueError(f"model scores must have shape {expected_shape}, got {scores.shape}")

    scores = np.clip(scores, 0.0, 1.0)
    submission = pd.DataFrame(scores, columns=label_columns)
    submission.insert(0, "fname", sample_submission["fname"].astype(str).to_numpy())
    validate_submission(submission, label_columns, expected_rows=len(sample_submission))
    return submission


def read_sample_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_submission(
    submission: pd.DataFrame,
    output_path: Path,
    label_columns: list[str],
) -> None:
    validate_submission(submission, label_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
