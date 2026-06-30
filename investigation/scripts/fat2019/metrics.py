from __future__ import annotations

import numpy as np


def _one_sample_positive_class_precisions(
    scores: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive_class_indices = np.flatnonzero(truth > 0)
    if len(positive_class_indices) == 0:
        return positive_class_indices, np.zeros(0, dtype=np.float64)

    num_classes = scores.shape[0]
    retrieved_classes = np.argsort(scores)[::-1]
    class_rankings = np.zeros(num_classes, dtype=np.int32)
    class_rankings[retrieved_classes] = np.arange(num_classes)

    retrieved_class_true = np.zeros(num_classes, dtype=np.bool_)
    retrieved_class_true[class_rankings[positive_class_indices]] = True
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)

    precision_at_hits = (
        retrieved_cumulative_hits[class_rankings[positive_class_indices]]
        / (1 + class_rankings[positive_class_indices].astype(np.float64))
    )
    return positive_class_indices, precision_at_hits


def calculate_per_class_lwlrap(
    truth: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if truth.shape != scores.shape:
        raise ValueError(
            f"truth and scores must have the same shape, got {truth.shape} and {scores.shape}"
        )

    num_samples, num_classes = scores.shape
    precisions_for_samples_by_classes = np.zeros(
        (num_samples, num_classes),
        dtype=np.float64,
    )

    for sample_index in range(num_samples):
        positive_class_indices, precision_at_hits = _one_sample_positive_class_precisions(
            scores[sample_index, :],
            truth[sample_index, :],
        )
        precisions_for_samples_by_classes[sample_index, positive_class_indices] = (
            precision_at_hits
        )

    labels_per_class = np.sum(truth > 0, axis=0)
    total_labels = np.sum(labels_per_class)
    if total_labels == 0:
        raise ValueError("truth must contain at least one positive label")

    weight_per_class = labels_per_class / float(total_labels)
    per_class_lwlrap = np.sum(precisions_for_samples_by_classes, axis=0) / np.maximum(
        1,
        labels_per_class,
    )
    return per_class_lwlrap, weight_per_class


def calculate_overall_lwlrap(truth: np.ndarray, scores: np.ndarray) -> float:
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(truth, scores)
    return float(np.sum(per_class_lwlrap * weight_per_class))
