from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def compute_pos_weight(targets: np.ndarray) -> np.ndarray:
    if targets.ndim != 2:
        raise ValueError(f"expected 2D targets, got shape {targets.shape}")

    positives = targets.sum(axis=0).astype(np.float32)
    negatives = (targets.shape[0] - positives).astype(np.float32)
    weights = np.ones_like(positives, dtype=np.float32)
    non_empty = positives > 0.0
    weights[non_empty] = negatives[non_empty] / positives[non_empty]
    return np.clip(weights, 1.0, 20.0).astype(np.float32)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64, copy=False), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def make_train_valid_indices(
    *,
    num_rows: int,
    test_size: float,
    seed: int,
    full_train: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    indices = np.arange(num_rows)
    if full_train:
        return indices, np.array([], dtype=np.int64)
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
    )
    return train_indices, valid_indices
