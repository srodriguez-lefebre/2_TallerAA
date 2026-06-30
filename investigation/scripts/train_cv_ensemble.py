from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fat2019.data import load_training_labels
from scripts.fat2019.labels import dataframe_to_multihot
from scripts.fat2019.metrics import calculate_overall_lwlrap
from scripts.fat2019.submission import (
    label_columns_from_sample,
    read_sample_submission,
    write_submission,
)
from scripts.train_sklearn_variants import (
    load_cached_features,
    make_model,
    parse_model_names,
    predict_scores,
)


def average_score_matrices(score_matrices: list[np.ndarray]) -> np.ndarray:
    if not score_matrices:
        raise ValueError("expected at least one score matrix")
    first_shape = score_matrices[0].shape
    if any(scores.shape != first_shape for scores in score_matrices):
        raise ValueError("all score matrices must have the same shape")
    return np.mean(np.stack(score_matrices, axis=0), axis=0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train K-fold sklearn ensembles on cached curated features.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions/cv_ensemble"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments/cv_ensemble"))
    parser.add_argument("--models", default="logreg_c001")
    parser.add_argument("--feature-set", choices=("basic", "extended"), default="basic")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _build_submission(
    sample_submission: pd.DataFrame,
    label_columns: list[str],
    scores: np.ndarray,
) -> pd.DataFrame:
    submission = pd.DataFrame(np.clip(scores, 0.0, 1.0), columns=label_columns)
    submission.insert(0, "fname", sample_submission["fname"].astype(str).to_numpy())
    return submission


def main() -> None:
    args = _build_parser().parse_args()
    model_names = parse_model_names(args.models)
    if args.n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    labels = load_training_labels(args.data_dir / "train_curated.csv")
    x, x_test = load_cached_features(args.data_dir, args.feature_set)
    y = dataframe_to_multihot(labels, label_columns)

    args.submissions_dir.mkdir(parents=True, exist_ok=True)
    args.experiments_dir.mkdir(parents=True, exist_ok=True)

    splitter = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.arange(len(x))))
    results: list[dict[str, object]] = []

    for model_name in model_names:
        print(f"training {model_name}")
        oof_scores = np.zeros_like(y, dtype=np.float64)
        test_scores_by_fold: list[np.ndarray] = []
        fold_lwlraps: list[float] = []

        for fold_index, (train_indices, valid_indices) in enumerate(splits, start=1):
            model = make_model(model_name, seed=args.seed + fold_index)
            model.fit(x[train_indices], y[train_indices])

            valid_scores = predict_scores(model, x[valid_indices])
            oof_scores[valid_indices] = valid_scores
            fold_lwlrap = calculate_overall_lwlrap(y[valid_indices], valid_scores)
            fold_lwlraps.append(fold_lwlrap)
            print(f"{model_name} fold {fold_index}: valid_lwlrap={fold_lwlrap:.6f}")

            test_scores_by_fold.append(predict_scores(model, x_test))

        test_scores = average_score_matrices(test_scores_by_fold)
        oof_lwlrap = calculate_overall_lwlrap(y, oof_scores)
        submission = _build_submission(sample_submission, label_columns, test_scores)
        submission_path = args.submissions_dir / f"{model_name}_{args.n_splits}fold.csv"
        write_submission(submission, submission_path, label_columns)

        scores_path = args.experiments_dir / f"{model_name}_{args.n_splits}fold_scores.npz"
        np.savez_compressed(scores_path, oof=oof_scores, test=test_scores)
        result = {
            "name": model_name,
            "oof_lwlrap": float(oof_lwlrap),
            "mean_fold_lwlrap": float(np.mean(fold_lwlraps)),
            "std_fold_lwlrap": float(np.std(fold_lwlraps)),
            "submission_path": str(submission_path),
            "scores_path": str(scores_path),
        }
        results.append(result)
        print(f"{model_name}: oof_lwlrap={oof_lwlrap:.6f}")

    results_df = pd.DataFrame(results).sort_values("oof_lwlrap", ascending=False)
    results_path = args.experiments_dir / "cv_ensemble_results.csv"
    results_df.to_csv(results_path, index=False)
    metadata_path = args.experiments_dir / "cv_ensemble_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "n_splits": args.n_splits,
                "models": model_names,
                "feature_set": args.feature_set,
                "rows": int(len(x)),
                "labels": len(label_columns),
            },
            indent=2,
        )
        + "\n"
    )
    print(results_df.to_string(index=False))
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
