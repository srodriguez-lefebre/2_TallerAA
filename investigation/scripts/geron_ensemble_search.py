from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fat2019.data import load_training_labels
from scripts.fat2019.labels import dataframe_to_multihot
from scripts.fat2019.metrics import calculate_overall_lwlrap
from scripts.fat2019.submission import label_columns_from_sample, read_sample_submission
from scripts.train_sklearn_variants import predict_scores


GERON_SOURCE = "Geron ch7: soft voting, random subspace/bagging intuition, stacking"
COURSE_SOURCE = "TAA hyperparameter/model selection: holdout, CV, random search"


@dataclass(frozen=True)
class HoldoutPredictions:
    valid_indices: np.ndarray
    truth: np.ndarray
    matrices: dict[str, np.ndarray]


def _positive_weight_items(weights: Mapping[str, float]) -> list[tuple[str, float]]:
    if not weights:
        raise ValueError("expected at least one positive weight")

    positive_items: list[tuple[str, float]] = []
    for name, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError("weights must be non-negative")
        if weight > 0.0:
            positive_items.append((name, weight))

    if not positive_items:
        raise ValueError("expected at least one positive weight")
    return positive_items


def _validate_same_shape(arrays: Sequence[np.ndarray]) -> tuple[int, int]:
    if not arrays:
        raise ValueError("expected at least one score matrix")
    first_shape = arrays[0].shape
    if len(first_shape) != 2:
        raise ValueError(f"score matrices must be 2-D, got {first_shape}")
    if any(scores.shape != first_shape for scores in arrays):
        raise ValueError("all score matrices must have the same shape")
    return int(first_shape[0]), int(first_shape[1])


def weighted_average_score_matrices(
    matrices: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    positive_items = _positive_weight_items(weights)
    unknown_names = sorted(name for name, _ in positive_items if name not in matrices)
    if unknown_names:
        raise ValueError(f"unknown score matrix names: {unknown_names}")

    selected_arrays = [np.asarray(matrices[name], dtype=np.float64) for name, _ in positive_items]
    _validate_same_shape(selected_arrays)

    total_weight = float(sum(weight for _, weight in positive_items))
    weighted = np.zeros_like(selected_arrays[0], dtype=np.float64)
    for scores, (_, weight) in zip(selected_arrays, positive_items):
        weighted += scores * weight
    return weighted / total_weight


def _validate_compatible_score_shapes(*arrays: np.ndarray) -> None:
    _validate_same_shape([np.asarray(array, dtype=np.float64) for array in arrays])


def build_fashion_system_scores(
    sklearn_scores: np.ndarray,
    head_scores: np.ndarray,
    relu_scores: np.ndarray,
    literal_scores: np.ndarray,
) -> np.ndarray:
    _validate_compatible_score_shapes(sklearn_scores, head_scores, relu_scores, literal_scores)
    return 0.15 * sklearn_scores + 0.85 * (
        0.575 * head_scores + 0.30 * relu_scores + 0.125 * literal_scores
    )


def build_current_system_scores(
    fashion_scores: np.ndarray,
    separable_residual_scores: np.ndarray,
    resnet_scores: np.ndarray,
    headsep_scores: np.ndarray,
) -> np.ndarray:
    _validate_compatible_score_shapes(
        fashion_scores,
        separable_residual_scores,
        resnet_scores,
        headsep_scores,
    )
    return (
        0.575 * fashion_scores
        + 0.10 * separable_residual_scores
        + 0.175 * resnet_scores
        + 0.15 * headsep_scores
    )


def blend_score_matrices(
    base_scores: np.ndarray,
    branch_scores: np.ndarray,
    *,
    branch_weight: float,
) -> np.ndarray:
    if branch_weight < 0.0 or branch_weight > 1.0:
        raise ValueError("branch_weight must be between 0 and 1")
    _validate_compatible_score_shapes(base_scores, branch_scores)
    return (1.0 - branch_weight) * base_scores + branch_weight * branch_scores


def rowwise_zscore_scores(scores: np.ndarray) -> np.ndarray:
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.ndim != 2:
        raise ValueError("scores must be 2-D")
    mean = scores_array.mean(axis=1, keepdims=True)
    std = scores_array.std(axis=1, keepdims=True)
    return (scores_array - mean) / np.maximum(std, 1e-6)


def rowwise_minmax_scores(scores: np.ndarray) -> np.ndarray:
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.ndim != 2:
        raise ValueError("scores must be 2-D")
    row_min = scores_array.min(axis=1, keepdims=True)
    row_max = scores_array.max(axis=1, keepdims=True)
    return (scores_array - row_min) / np.maximum(row_max - row_min, 1e-6)


def rowwise_rank_scores(scores: np.ndarray) -> np.ndarray:
    scores_array = np.asarray(scores, dtype=np.float64)
    if scores_array.ndim != 2:
        raise ValueError("scores must be 2-D")
    order = np.argsort(np.argsort(scores_array, axis=1), axis=1)
    return order / max(1, scores_array.shape[1] - 1)


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))


def blend_score_matrices_with_transform(
    base_scores: np.ndarray,
    branch_scores: np.ndarray,
    *,
    branch_weight: float,
    method: str,
) -> np.ndarray:
    if branch_weight < 0.0 or branch_weight > 1.0:
        raise ValueError("branch_weight must be between 0 and 1")
    _validate_compatible_score_shapes(base_scores, branch_scores)

    base_array = np.asarray(base_scores, dtype=np.float64)
    branch_array = np.asarray(branch_scores, dtype=np.float64)
    if method == "prob":
        return blend_score_matrices(base_array, branch_array, branch_weight=branch_weight)
    if method == "logit_sigmoid":
        return _sigmoid(
            (1.0 - branch_weight) * _logit(base_array) + branch_weight * _logit(branch_array)
        )
    if method == "geom":
        return np.exp(
            (1.0 - branch_weight) * np.log(np.clip(base_array, 1e-6, 1.0))
            + branch_weight * np.log(np.clip(branch_array, 1e-6, 1.0))
        )
    if method == "row_rank":
        return (1.0 - branch_weight) * rowwise_rank_scores(
            base_array
        ) + branch_weight * rowwise_rank_scores(branch_array)
    if method == "row_z":
        return (1.0 - branch_weight) * rowwise_zscore_scores(
            base_array
        ) + branch_weight * rowwise_zscore_scores(branch_array)
    raise ValueError(f"unknown two-way blend method: {method}")


def _zscore_from_fit(scores: np.ndarray, fit_scores: np.ndarray) -> np.ndarray:
    mean = fit_scores.mean(axis=0, keepdims=True)
    std = fit_scores.std(axis=0, keepdims=True)
    return (scores - mean) / np.maximum(std, 1e-6)


def _robust_zscore_from_fit(scores: np.ndarray, fit_scores: np.ndarray) -> np.ndarray:
    median = np.median(fit_scores, axis=0, keepdims=True)
    q75 = np.percentile(fit_scores, 75, axis=0, keepdims=True)
    q25 = np.percentile(fit_scores, 25, axis=0, keepdims=True)
    return (scores - median) / np.maximum(q75 - q25, 1e-6)


def _minmax_from_fit(scores: np.ndarray, fit_scores: np.ndarray) -> np.ndarray:
    low = fit_scores.min(axis=0, keepdims=True)
    high = fit_scores.max(axis=0, keepdims=True)
    return (scores - low) / np.maximum(high - low, 1e-6)


def _rank_columns(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(scores, axis=0), axis=0)
    return order / max(1, scores.shape[0] - 1)


def average_transformed_score_matrices(
    matrices: Mapping[str, np.ndarray],
    names: Sequence[str],
    *,
    method: str,
    fit_matrices: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    if not names:
        raise ValueError("expected at least one matrix name")
    unknown_names = sorted(name for name in names if name not in matrices)
    if unknown_names:
        raise ValueError(f"unknown score matrix names: {unknown_names}")

    fit_source = matrices if fit_matrices is None else fit_matrices
    missing_fit_names = sorted(name for name in names if name not in fit_source)
    if missing_fit_names:
        raise ValueError(f"missing fit matrices: {missing_fit_names}")

    selected = [np.asarray(matrices[name], dtype=np.float64) for name in names]
    _validate_same_shape(selected)

    transformed: list[np.ndarray] = []
    for name, scores in zip(names, selected):
        fit_scores = np.asarray(fit_source[name], dtype=np.float64)
        if fit_scores.ndim != 2 or fit_scores.shape[1] != scores.shape[1]:
            raise ValueError("fit matrices must have the same number of classes")

        if method == "prob_avg":
            transformed.append(scores)
        elif method == "logit_avg":
            transformed.append(_logit(scores))
        elif method == "z_prob_avg":
            transformed.append(_zscore_from_fit(scores, fit_scores))
        elif method == "z_logit_avg":
            transformed.append(_zscore_from_fit(_logit(scores), _logit(fit_scores)))
        elif method == "robust_z_prob_avg":
            transformed.append(_robust_zscore_from_fit(scores, fit_scores))
        elif method == "minmax_avg":
            transformed.append(_minmax_from_fit(scores, fit_scores))
        elif method == "rank_avg":
            transformed.append(_rank_columns(scores))
        else:
            raise ValueError(f"unknown transform method: {method}")
    return np.mean(np.stack(transformed, axis=0), axis=0)


def generate_dirichlet_weight_candidates(
    *,
    model_count: int,
    seed: int,
    trials_per_alpha: int,
    alpha_scales: Sequence[float],
) -> np.ndarray:
    if model_count <= 0:
        raise ValueError("model_count must be positive")
    if trials_per_alpha < 0:
        raise ValueError("trials_per_alpha must be non-negative")
    if not alpha_scales:
        raise ValueError("expected at least one alpha scale")

    rng = np.random.default_rng(seed)
    candidates = [np.full(model_count, 1.0 / model_count, dtype=np.float64)]
    for alpha_scale in alpha_scales:
        if alpha_scale <= 0.0:
            raise ValueError("alpha scales must be positive")
        alpha = np.full(model_count, float(alpha_scale), dtype=np.float64)
        for _ in range(trials_per_alpha):
            candidates.append(rng.dirichlet(alpha))
    return np.vstack(candidates)


def fit_predict_classwise_logistic_oof(
    matrices: Mapping[str, np.ndarray],
    names: Sequence[str],
    truth: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    c: float,
) -> np.ndarray:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if c <= 0.0:
        raise ValueError("c must be positive")

    selected = [np.asarray(matrices[name], dtype=np.float64) for name in names]
    sample_count, class_count = _validate_same_shape(selected)
    if truth.shape != (sample_count, class_count):
        raise ValueError(
            f"truth shape must match score matrices, got {truth.shape} and "
            f"{(sample_count, class_count)}"
        )

    stacked_scores = np.stack(selected, axis=2)
    oof_scores = np.zeros_like(truth, dtype=np.float64)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_indices, valid_indices in splitter.split(np.arange(sample_count)):
        for class_index in range(class_count):
            class_truth = truth[train_indices, class_index]
            if class_truth.min() == class_truth.max():
                oof_scores[valid_indices, class_index] = float(class_truth[0])
                continue

            classifier = LogisticRegression(
                C=c,
                class_weight="balanced",
                solver="liblinear",
                max_iter=200,
                random_state=seed,
            )
            classifier.fit(stacked_scores[train_indices, class_index, :], class_truth)
            oof_scores[valid_indices, class_index] = classifier.predict_proba(
                stacked_scores[valid_indices, class_index, :]
            )[:, 1]

    return oof_scores


def _weights_to_json(names: Sequence[str], weights: np.ndarray) -> str:
    payload = {
        name: round(float(weight), 8)
        for name, weight in zip(names, weights)
        if float(weight) > 0.0
    }
    return json.dumps(payload, sort_keys=True)


def _append_result(
    rows: list[dict[str, object]],
    *,
    run_name: str,
    source_idea: str,
    model_names: Sequence[str],
    method: str,
    validation_strategy: str,
    valid_lwlrap: float,
    weights_json: str = "",
    notes: str = "",
) -> None:
    rows.append(
        {
            "run_name": run_name,
            "source_idea": source_idea,
            "model_names": "+".join(model_names),
            "method": method,
            "validation_strategy": validation_strategy,
            "valid_lwlrap": float(valid_lwlrap),
            "weights_json": weights_json,
            "notes": notes,
        }
    )


def load_default_holdout_predictions(data_dir: Path, experiments_dir: Path) -> HoldoutPredictions:
    sample_submission = read_sample_submission(data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    truth_all = dataframe_to_multihot(load_training_labels(data_dir / "train_curated.csv"), label_columns)

    base_path = experiments_dir / "catsdogs_holdout_predictions.npz"
    base = np.load(base_path)
    valid_indices = base["valid_indices"]
    matrices = {
        name: base[name].astype(np.float64)
        for name in ("head", "relu", "literal", "sepres", "resnet")
    }

    extra_sources = (
        (experiments_dir / "catsdogs_headsep_holdout.npz", "headsep"),
        (experiments_dir / "logmel_cnn_temporal_bigru" / "holdout_predictions.npz", "temporal_bigru"),
        (
            experiments_dir / "logmel_cnn_separable_temporal_bigru" / "holdout_predictions.npz",
            "separable_temporal_bigru",
        ),
    )
    for path, key in extra_sources:
        loaded = np.load(path)
        if not np.array_equal(valid_indices, loaded["valid_indices"]):
            raise ValueError(f"{path} does not use the same valid_indices as {base_path}")
        matrices[key] = loaded[key].astype(np.float64)

    return HoldoutPredictions(
        valid_indices=valid_indices,
        truth=truth_all[valid_indices],
        matrices=matrices,
    )


def load_sklearn_full_model_scores(
    data_dir: Path,
    model_path: Path,
    valid_indices: np.ndarray,
) -> np.ndarray:
    with model_path.open("rb") as model_file:
        payload = pickle.load(model_file)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    features = np.load(data_dir / "curated_logmel_stats_all.npz", allow_pickle=False)["x"].astype(
        np.float32
    )
    return predict_scores(model, features[valid_indices])


def with_reconstructed_current_system(
    predictions: HoldoutPredictions,
    *,
    data_dir: Path,
    sklearn_model_path: Path,
) -> HoldoutPredictions:
    required_names = ("head", "relu", "literal", "sepres", "resnet", "headsep")
    missing_names = sorted(name for name in required_names if name not in predictions.matrices)
    if missing_names:
        raise ValueError(f"missing matrices needed to reconstruct current: {missing_names}")

    matrices = dict(predictions.matrices)
    matrices["sklearn"] = load_sklearn_full_model_scores(
        data_dir,
        sklearn_model_path,
        predictions.valid_indices,
    )
    matrices["fashion"] = build_fashion_system_scores(
        matrices["sklearn"],
        matrices["head"],
        matrices["relu"],
        matrices["literal"],
    )
    matrices["current"] = build_current_system_scores(
        matrices["fashion"],
        matrices["sepres"],
        matrices["resnet"],
        matrices["headsep"],
    )
    return HoldoutPredictions(
        valid_indices=predictions.valid_indices,
        truth=predictions.truth,
        matrices=matrices,
    )


def evaluate_individuals(predictions: HoldoutPredictions) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, scores in predictions.matrices.items():
        _append_result(
            rows,
            run_name=f"individual_{name}",
            source_idea="baseline component",
            model_names=[name],
            method="individual",
            validation_strategy="holdout seed42",
            valid_lwlrap=calculate_overall_lwlrap(predictions.truth, scores),
        )
    return rows


def evaluate_equal_average_combinations(
    predictions: HoldoutPredictions,
    *,
    max_subset_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    names = list(predictions.matrices)
    upper_size = min(max_subset_size, len(names))
    for subset_size in range(2, upper_size + 1):
        for subset in itertools.combinations(names, subset_size):
            weights = {name: 1.0 for name in subset}
            scores = weighted_average_score_matrices(predictions.matrices, weights)
            _append_result(
                rows,
                run_name=f"equal_average_{subset_size}",
                source_idea=GERON_SOURCE,
                model_names=subset,
                method="equal_soft_voting",
                validation_strategy="holdout seed42",
                valid_lwlrap=calculate_overall_lwlrap(predictions.truth, scores),
                weights_json=_weights_to_json(subset, np.full(len(subset), 1.0 / len(subset))),
            )
    return rows


def evaluate_transformed_averages(predictions: HoldoutPredictions) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subsets = {
        "best_equal_prev": (
            "literal",
            "sepres",
            "resnet",
            "headsep",
            "separable_temporal_bigru",
        ),
        "selected6": (
            "head",
            "literal",
            "sepres",
            "resnet",
            "headsep",
            "separable_temporal_bigru",
        ),
        "no_temporal": ("head", "literal", "sepres", "resnet", "headsep"),
        "all8": tuple(predictions.matrices),
    }
    methods = (
        "prob_avg",
        "logit_avg",
        "z_prob_avg",
        "z_logit_avg",
        "robust_z_prob_avg",
        "minmax_avg",
        "rank_avg",
    )

    for subset_name, names in subsets.items():
        for method in methods:
            scores = average_transformed_score_matrices(
                predictions.matrices,
                names,
                method=method,
            )
            _append_result(
                rows,
                run_name=f"transformed_{subset_name}_{method}",
                source_idea=f"{GERON_SOURCE}; feature scaling/calibration sanity check",
                model_names=names,
                method=method,
                validation_strategy="holdout seed42",
                valid_lwlrap=calculate_overall_lwlrap(predictions.truth, scores),
                notes=subset_name,
            )
    return rows


def evaluate_reconstructed_current_system(
    predictions: HoldoutPredictions,
    *,
    max_branch_weight: float,
    branch_step: float,
) -> list[dict[str, object]]:
    if max_branch_weight < 0.0 or max_branch_weight > 1.0:
        raise ValueError("max_branch_weight must be between 0 and 1")
    if branch_step <= 0.0:
        raise ValueError("branch_step must be positive")

    rows: list[dict[str, object]] = []
    reconstruction_note = (
        "current reconstructed from stored sklearn final model plus documented "
        "fashion/catsdogs holdout formula"
    )
    for name in ("sklearn", "fashion", "current"):
        _append_result(
            rows,
            run_name=f"reconstructed_{name}",
            source_idea="historical baseline reconstruction",
            model_names=[name],
            method="reconstructed_system",
            validation_strategy="holdout seed42 local analog",
            valid_lwlrap=calculate_overall_lwlrap(predictions.truth, predictions.matrices[name]),
            notes=reconstruction_note,
        )

    branch_name = "separable_temporal_bigru"
    weights = np.arange(0.0, max_branch_weight + branch_step / 2.0, branch_step)
    for branch_weight in weights:
        scores = blend_score_matrices(
            predictions.matrices["current"],
            predictions.matrices[branch_name],
            branch_weight=float(branch_weight),
        )
        _append_result(
            rows,
            run_name="current_sep_temporal_sweep",
            source_idea=f"{GERON_SOURCE}; documented current reconstruction",
            model_names=("current", branch_name),
            method="two_way_soft_voting_sweep",
            validation_strategy="holdout seed42 local analog",
            valid_lwlrap=calculate_overall_lwlrap(predictions.truth, scores),
            weights_json=json.dumps(
                {
                    "current": round(float(1.0 - branch_weight), 8),
                    branch_name: round(float(branch_weight), 8),
                },
                sort_keys=True,
            ),
            notes=reconstruction_note,
        )
    return rows


def evaluate_reconstructed_current_transform_sweeps(
    predictions: HoldoutPredictions,
    *,
    max_branch_weight: float,
    branch_step: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    branch_name = "separable_temporal_bigru"
    weights = np.arange(0.0, max_branch_weight + branch_step / 2.0, branch_step)
    methods = ("prob", "logit_sigmoid", "geom", "row_rank", "row_z")

    for method in methods:
        for branch_weight in weights:
            scores = blend_score_matrices_with_transform(
                predictions.matrices["current"],
                predictions.matrices[branch_name],
                branch_weight=float(branch_weight),
                method=method,
            )
            _append_result(
                rows,
                run_name="current_sep_temporal_transform_sweep",
                source_idea=f"{GERON_SOURCE}; rank-preserving score post-processing",
                model_names=("current", branch_name),
                method=method,
                validation_strategy="holdout seed42 local analog",
                valid_lwlrap=calculate_overall_lwlrap(predictions.truth, scores),
                weights_json=json.dumps(
                    {
                        "current": round(float(1.0 - branch_weight), 8),
                        branch_name: round(float(branch_weight), 8),
                    },
                    sort_keys=True,
                ),
                notes="transformed two-way blend over reconstructed current",
            )
    return rows


def evaluate_weight_search(
    predictions: HoldoutPredictions,
    *,
    names: Sequence[str],
    seed: int,
    n_splits: int,
    trials_per_alpha: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    random_candidates = generate_dirichlet_weight_candidates(
        model_count=len(names),
        seed=seed,
        trials_per_alpha=trials_per_alpha,
        alpha_scales=(0.2, 0.5, 1.0, 2.0, 5.0),
    )
    hand_candidates = np.array(
        [
            [0.575, 0.0, 0.10, 0.175, 0.15, 0.0],
            [0.50, 0.0, 0.10, 0.15, 0.15, 0.10],
            [0.45, 0.0, 0.10, 0.15, 0.15, 0.15],
        ],
        dtype=np.float64,
    )
    hand_candidates = hand_candidates / hand_candidates.sum(axis=1, keepdims=True)
    candidates = np.vstack([random_candidates, hand_candidates])
    arrays = np.stack([predictions.matrices[name] for name in names], axis=0)

    best_full_score = -1.0
    best_full_weights = candidates[0]
    for weights in candidates:
        scores = np.tensordot(weights, arrays, axes=(0, 0))
        valid_lwlrap = calculate_overall_lwlrap(predictions.truth, scores)
        if valid_lwlrap > best_full_score:
            best_full_score = valid_lwlrap
            best_full_weights = weights

    _append_result(
        rows,
        run_name="dirichlet_weight_search_full_holdout",
        source_idea=f"{GERON_SOURCE}; {COURSE_SOURCE}",
        model_names=names,
        method="random_dirichlet_weight_search",
        validation_strategy="full holdout search, optimistic",
        valid_lwlrap=best_full_score,
        weights_json=_weights_to_json(names, best_full_weights),
        notes=f"{len(candidates)} candidates",
    )

    oof_scores = np.zeros_like(predictions.truth, dtype=np.float64)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_index, (train_indices, valid_indices) in enumerate(
        splitter.split(np.arange(predictions.truth.shape[0])),
        start=1,
    ):
        best_fold_score = -1.0
        best_fold_weights = candidates[0]
        for weights in candidates:
            train_scores = np.tensordot(weights, arrays[:, train_indices], axes=(0, 0))
            valid_lwlrap = calculate_overall_lwlrap(predictions.truth[train_indices], train_scores)
            if valid_lwlrap > best_fold_score:
                best_fold_score = valid_lwlrap
                best_fold_weights = weights

        oof_scores[valid_indices] = np.tensordot(
            best_fold_weights,
            arrays[:, valid_indices],
            axes=(0, 0),
        )
        fold_lwlrap = calculate_overall_lwlrap(
            predictions.truth[valid_indices],
            oof_scores[valid_indices],
        )
        _append_result(
            rows,
            run_name=f"dirichlet_weight_search_oof_fold{fold_index}",
            source_idea=f"{GERON_SOURCE}; {COURSE_SOURCE}",
            model_names=names,
            method="random_dirichlet_weight_search",
            validation_strategy=f"{n_splits}-fold oof fold",
            valid_lwlrap=fold_lwlrap,
            weights_json=_weights_to_json(names, best_fold_weights),
            notes=f"train_best={best_fold_score:.6f}; {len(candidates)} candidates",
        )

    _append_result(
        rows,
        run_name="dirichlet_weight_search_oof",
        source_idea=f"{GERON_SOURCE}; {COURSE_SOURCE}",
        model_names=names,
        method="random_dirichlet_weight_search",
        validation_strategy=f"{n_splits}-fold oof on holdout",
        valid_lwlrap=calculate_overall_lwlrap(predictions.truth, oof_scores),
        notes=f"{len(candidates)} candidates",
    )
    return rows, oof_scores


def evaluate_logistic_stacking(
    predictions: HoldoutPredictions,
    *,
    names: Sequence[str],
    seed: int,
    n_splits: int,
    c: float,
) -> tuple[list[dict[str, object]], np.ndarray]:
    oof_scores = fit_predict_classwise_logistic_oof(
        predictions.matrices,
        names,
        predictions.truth,
        n_splits=n_splits,
        seed=seed,
        c=c,
    )
    rows: list[dict[str, object]] = []
    _append_result(
        rows,
        run_name="classwise_logistic_stacking_oof",
        source_idea=f"{GERON_SOURCE}; {COURSE_SOURCE}",
        model_names=names,
        method="classwise_logistic_stacking",
        validation_strategy=f"{n_splits}-fold oof on holdout",
        valid_lwlrap=calculate_overall_lwlrap(predictions.truth, oof_scores),
        notes=f"C={c}",
    )
    return rows, oof_scores


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Geron-inspired soft-voting and stacking on FAT2019 holdout predictions.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("investigation/experiments"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--trials-per-alpha", type=int, default=100)
    parser.add_argument("--max-equal-subset-size", type=int, default=6)
    parser.add_argument("--stacking-c", type=float, default=0.1)
    parser.add_argument(
        "--sklearn-model-path",
        type=Path,
        default=Path("investigation/models/variants_regsearch_lowc/logreg_c001_logmel_stats.pkl"),
    )
    parser.add_argument("--current-sweep-max", type=float, default=0.30)
    parser.add_argument("--current-sweep-step", type=float, default=0.005)
    parser.add_argument("--skip-reconstructed-current", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir or args.experiments_dir / "geron_ensemble_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_default_holdout_predictions(args.data_dir, args.experiments_dir)
    selected_names = (
        "head",
        "literal",
        "sepres",
        "resnet",
        "headsep",
        "separable_temporal_bigru",
    )

    rows: list[dict[str, object]] = []
    rows.extend(evaluate_individuals(predictions))
    rows.extend(
        evaluate_equal_average_combinations(
            predictions,
            max_subset_size=args.max_equal_subset_size,
        )
    )
    rows.extend(evaluate_transformed_averages(predictions))

    if not args.skip_reconstructed_current:
        reconstructed_predictions = with_reconstructed_current_system(
            predictions,
            data_dir=args.data_dir,
            sklearn_model_path=args.sklearn_model_path,
        )
        rows.extend(
            evaluate_reconstructed_current_system(
                reconstructed_predictions,
                max_branch_weight=args.current_sweep_max,
                branch_step=args.current_sweep_step,
            )
        )
        rows.extend(
            evaluate_reconstructed_current_transform_sweeps(
                reconstructed_predictions,
                max_branch_weight=args.current_sweep_max,
                branch_step=args.current_sweep_step,
            )
        )

    weight_rows, weight_oof_scores = evaluate_weight_search(
        predictions,
        names=selected_names,
        seed=args.seed,
        n_splits=args.n_splits,
        trials_per_alpha=args.trials_per_alpha,
    )
    rows.extend(weight_rows)

    stacking_rows, stacking_oof_scores = evaluate_logistic_stacking(
        predictions,
        names=selected_names,
        seed=args.seed,
        n_splits=args.n_splits,
        c=args.stacking_c,
    )
    rows.extend(stacking_rows)

    results = pd.DataFrame(rows).sort_values("valid_lwlrap", ascending=False)
    results_path = output_dir / "geron_ensemble_results.csv"
    results.to_csv(results_path, index=False)

    np.savez_compressed(
        output_dir / "geron_ensemble_oof_predictions.npz",
        valid_indices=predictions.valid_indices,
        weight_search=weight_oof_scores,
        classwise_logistic_stacking=stacking_oof_scores,
    )
    metadata_path = output_dir / "geron_ensemble_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_splits": args.n_splits,
                "trials_per_alpha": args.trials_per_alpha,
                "stacking_c": args.stacking_c,
                "sklearn_model_path": str(args.sklearn_model_path),
                "current_sweep_max": args.current_sweep_max,
                "current_sweep_step": args.current_sweep_step,
                "valid_rows": int(predictions.truth.shape[0]),
                "classes": int(predictions.truth.shape[1]),
                "sources": [GERON_SOURCE, COURSE_SOURCE],
                "selected_names": list(selected_names),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(results.head(20).to_string(index=False))
    print(f"wrote {results_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
