from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_feature_cache import cache_path
from scripts.fat2019.data import load_training_labels
from scripts.fat2019.labels import dataframe_to_multihot
from scripts.fat2019.metrics import calculate_overall_lwlrap
from scripts.fat2019.submission import (
    build_model_submission,
    label_columns_from_sample,
    read_sample_submission,
    write_submission,
)


@dataclass(frozen=True)
class FeatureMatrices:
    curated: np.ndarray
    test: np.ndarray
    noisy: np.ndarray | None = None


@dataclass(frozen=True)
class VariantResult:
    name: str
    valid_lwlrap: float
    model_path: Path
    submission_path: Path


LOGREG_C_VALUES = {
    "logreg_c0003": 0.003,
    "logreg_c0005": 0.005,
    "logreg_c0007": 0.007,
    "logreg_c0008": 0.008,
    "logreg_c001": 0.01,
    "logreg_c0012": 0.012,
    "logreg_c0015": 0.015,
    "logreg_c002": 0.02,
    "logreg_c003": 0.03,
    "logreg_c005": 0.05,
    "logreg_c01": 0.1,
    "logreg_c02": 0.2,
    "logreg_c03": 0.3,
    "logreg_c05": 0.5,
    "logreg_c3": 3.0,
}


def parse_model_names(raw_names: str) -> list[str]:
    names = [name.strip() for name in raw_names.split(",") if name.strip()]
    if not names:
        raise ValueError("expected at least one model name")
    return names


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sklearn variants on cached log-mel statistics.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--models",
        default="logreg_c01,logreg_c02,logreg_c03,logreg_c05,logreg_c3,extra_trees,random_forest,mlp_small,xgb_hist",
        help="Comma-separated variant names.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument(
        "--include-noisy",
        action="store_true",
        help="Add train_noisy rows to model fitting while validating on curated only.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("basic", "extended"),
        default="basic",
    )
    return parser


def load_cached_features(data_dir: Path, feature_set: str) -> tuple[np.ndarray, np.ndarray]:
    matrices = load_feature_matrices(data_dir, feature_set, include_noisy=False)
    return matrices.curated, matrices.test


def load_feature_matrices(
    data_dir: Path,
    feature_set: str,
    *,
    include_noisy: bool = False,
) -> FeatureMatrices:
    curated_cache = np.load(
        cache_path(data_dir, split="curated", feature_set=feature_set),
        allow_pickle=False,
    )
    test_cache = np.load(
        cache_path(data_dir, split="test", feature_set=feature_set),
        allow_pickle=False,
    )
    noisy = None
    if include_noisy:
        noisy_cache = np.load(
            cache_path(data_dir, split="noisy", feature_set=feature_set),
            allow_pickle=False,
        )
        noisy = noisy_cache["x"].astype(np.float32)
    return FeatureMatrices(
        curated=curated_cache["x"].astype(np.float32),
        test=test_cache["x"].astype(np.float32),
        noisy=noisy,
    )


def make_model(name: str, *, seed: int):
    if name in LOGREG_C_VALUES:
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=LOGREG_C_VALUES[name],
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c005":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.05,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c001":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.01,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c0003":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.003,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c0005":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.005,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c0007":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.007,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c002":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.02,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c003":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.03,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c01":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c02":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.2,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c03":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c05":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "logreg_c3":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=3.0,
                    class_weight="balanced",
                    max_iter=700,
                    solver="liblinear",
                ),
                n_jobs=-1,
            ),
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "mlp_small":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(384, 128),
                alpha=1e-3,
                batch_size=128,
                early_stopping=True,
                learning_rate_init=5e-4,
                max_iter=120,
                random_state=seed,
            ),
        )
    if name == "sgd_log_alpha1e4":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=1e-4,
                    class_weight="balanced",
                    max_iter=3000,
                    tol=1e-4,
                    random_state=seed,
                ),
                n_jobs=-1,
            ),
        )
    if name == "sgd_log_alpha3e5":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=3e-5,
                    class_weight="balanced",
                    max_iter=3000,
                    tol=1e-4,
                    random_state=seed,
                ),
                n_jobs=-1,
            ),
        )
    if name == "sgd_log_elastic":
        return make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    alpha=1e-4,
                    l1_ratio=0.15,
                    class_weight="balanced",
                    max_iter=3000,
                    tol=1e-4,
                    random_state=seed,
                ),
                n_jobs=-1,
            ),
        )
    if name == "xgb_hist":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError("xgb_hist requires xgboost") from exc
        return OneVsRestClassifier(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_estimators=160,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                random_state=seed,
                n_jobs=1,
            ),
            n_jobs=2,
        )
    raise ValueError(f"unknown model variant: {name}")


def predict_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x)
        if isinstance(scores, list):
            scores = np.column_stack([column[:, 1] for column in scores])
        return np.asarray(scores, dtype=np.float64)
    if hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=np.float64)
        # Scale each class independently to [0, 1] for submission compatibility.
        min_values = decision.min(axis=0, keepdims=True)
        max_values = decision.max(axis=0, keepdims=True)
        return (decision - min_values) / np.maximum(1e-12, max_values - min_values)
    raise TypeError("model must expose predict_proba or decision_function")


def nearest_label_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    k: int = 35,
) -> np.ndarray:
    distances = pairwise_distances(x_query, x_train, metric="cosine")
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    weights = 1.0 / np.maximum(1e-6, distances[np.arange(len(x_query))[:, None], nearest])
    weighted_labels = y_train[nearest] * weights[:, :, None]
    return weighted_labels.sum(axis=1) / weights.sum(axis=1, keepdims=True)


def train_variant(
    name: str,
    *,
    x: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    sample_submission: pd.DataFrame,
    label_columns: list[str],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    x_extra: np.ndarray | None,
    y_extra: np.ndarray | None,
    seed: int,
    models_dir: Path,
    submissions_dir: Path,
) -> VariantResult:
    if (x_extra is None) != (y_extra is None):
        raise ValueError("x_extra and y_extra must be provided together")

    if name == "knn_cosine":
        x_valid_train = x[train_indices]
        y_valid_train = y[train_indices]
        x_full_train = x
        y_full_train = y
        if x_extra is not None and y_extra is not None:
            x_valid_train = np.vstack([x_valid_train, x_extra])
            y_valid_train = np.vstack([y_valid_train, y_extra])
            x_full_train = np.vstack([x_full_train, x_extra])
            y_full_train = np.vstack([y_full_train, y_extra])
        valid_scores = nearest_label_scores(x_valid_train, y_valid_train, x[valid_indices])
        valid_lwlrap = calculate_overall_lwlrap(y[valid_indices], valid_scores)
        test_scores = nearest_label_scores(x_full_train, y_full_train, x_test)
        submission = pd.DataFrame(np.clip(test_scores, 0.0, 1.0), columns=label_columns)
        submission.insert(0, "fname", sample_submission["fname"].astype(str).to_numpy())
        model_payload = {"name": name, "valid_lwlrap": valid_lwlrap, "k": 35}
    else:
        x_valid_train = x[train_indices]
        y_valid_train = y[train_indices]
        x_full_train = x
        y_full_train = y
        if x_extra is not None and y_extra is not None:
            x_valid_train = np.vstack([x_valid_train, x_extra])
            y_valid_train = np.vstack([y_valid_train, y_extra])
            x_full_train = np.vstack([x_full_train, x_extra])
            y_full_train = np.vstack([y_full_train, y_extra])

        valid_model = make_model(name, seed=seed)
        valid_model.fit(x_valid_train, y_valid_train)
        valid_scores = predict_scores(valid_model, x[valid_indices])
        valid_lwlrap = calculate_overall_lwlrap(y[valid_indices], valid_scores)

        full_model = make_model(name, seed=seed)
        full_model.fit(x_full_train, y_full_train)
        submission = build_model_submission(
            sample_submission,
            label_columns,
            _PredictWrapper(full_model),
            x_test,
        )
        model_payload = {
            "name": name,
            "valid_lwlrap": valid_lwlrap,
            "model": full_model,
            "label_columns": label_columns,
        }

    models_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}_logmel_stats.pkl"
    submission_path = submissions_dir / f"{name}_logmel_stats.csv"
    with model_path.open("wb") as model_file:
        pickle.dump(model_payload, model_file)
    write_submission(submission, submission_path, label_columns)
    return VariantResult(
        name=name,
        valid_lwlrap=float(valid_lwlrap),
        model_path=model_path,
        submission_path=submission_path,
    )


class _PredictWrapper:
    def __init__(self, model) -> None:
        self._model = model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return predict_scores(self._model, x)


def main() -> None:
    args = _build_parser().parse_args()
    model_names = parse_model_names(args.models)

    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    labels = load_training_labels(args.data_dir / "train_curated.csv")
    matrices = load_feature_matrices(
        args.data_dir,
        args.feature_set,
        include_noisy=args.include_noisy,
    )
    x = matrices.curated
    x_test = matrices.test
    y = dataframe_to_multihot(labels, label_columns)
    x_extra = None
    y_extra = None
    noisy_rows = 0
    if args.include_noisy:
        if matrices.noisy is None:
            raise ValueError("include_noisy was set but no noisy matrix was loaded")
        noisy_labels = load_training_labels(args.data_dir / "train_noisy.csv")
        if len(noisy_labels) != len(matrices.noisy):
            raise ValueError(
                f"noisy labels/features length mismatch: {len(noisy_labels)} vs {len(matrices.noisy)}"
            )
        x_extra = matrices.noisy
        y_extra = dataframe_to_multihot(noisy_labels, label_columns)
        noisy_rows = int(len(x_extra))

    if args.max_train is not None:
        labels = labels.head(args.max_train).copy()
        x = x[: args.max_train]
        y = y[: args.max_train]

    indices = np.arange(len(x))
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
    )

    results = []
    for name in model_names:
        print(f"training {name}")
        result = train_variant(
            name,
            x=x,
            y=y,
            x_test=x_test,
            sample_submission=sample_submission,
            label_columns=label_columns,
            train_indices=train_indices,
            valid_indices=valid_indices,
            x_extra=x_extra,
            y_extra=y_extra,
            seed=args.seed,
            models_dir=args.models_dir,
            submissions_dir=args.submissions_dir,
        )
        print(f"{name}: valid_lwlrap={result.valid_lwlrap:.6f}")
        results.append(result)

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(
        [
            {
                "name": result.name,
                "valid_lwlrap": result.valid_lwlrap,
                "model_path": str(result.model_path),
                "submission_path": str(result.submission_path),
            }
            for result in results
        ]
    ).sort_values("valid_lwlrap", ascending=False)
    output_path = args.experiments_dir / "sklearn_variant_results.csv"
    results_df.to_csv(output_path, index=False)
    metadata_path = args.experiments_dir / "sklearn_variant_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "test_size": args.test_size,
                "models": model_names,
                "feature_set": args.feature_set,
                "include_noisy": args.include_noisy,
                "rows": int(len(x)),
                "noisy_rows": noisy_rows,
                "labels": len(label_columns),
            },
            indent=2,
        )
        + "\n"
    )
    print(results_df.to_string(index=False))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
