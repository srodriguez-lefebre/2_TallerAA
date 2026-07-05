from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fat2019.metrics import calculate_overall_lwlrap  # noqa: E402
from fat2019.submission import (  # noqa: E402
    blend_submissions,
    label_columns_from_sample,
    read_sample_submission,
    validate_submission,
    write_submission,
)
from final_config import BRANCHES  # noqa: E402


COMPETITION = "freesound-audio-tagging-2019"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blend noisy fine-tuned branches and summarize metrics.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--work-dir", type=Path, default=Path("proyecto_actual_v2/codigo/work"))
    parser.add_argument("--output-dir", type=Path, default=Path("proyecto_actual_v2/codigo"))
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_branch_metadata(work_dir: Path, branch_name: str) -> dict[str, object]:
    path = work_dir / "runs" / branch_name / "experiments" / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing branch metadata: {path}")
    return json.loads(path.read_text())


def _blend_validation_scores(work_dir: Path) -> tuple[float, list[dict[str, object]]]:
    weighted_scores: np.ndarray | None = None
    targets: np.ndarray | None = None
    branch_rows: list[dict[str, object]] = []
    for branch in BRANCHES:
        experiments_dir = branch.run_dir(work_dir) / "experiments"
        scores = np.load(experiments_dir / "valid_scores_final.npy")
        branch_targets = np.load(experiments_dir / "valid_targets.npy")
        if targets is None:
            targets = branch_targets
        elif targets.shape != branch_targets.shape or not np.array_equal(targets, branch_targets):
            raise ValueError(f"validation targets differ for branch {branch.name}")
        if weighted_scores is None:
            weighted_scores = np.zeros_like(scores, dtype=np.float64)
        weighted_scores += branch.ensemble_weight * scores.astype(np.float64)
        metadata = _load_branch_metadata(work_dir, branch.name)
        branch_rows.append(
            {
                "branch": branch.name,
                "weight": branch.ensemble_weight,
                "baseline_lwlrap": metadata["baseline_lwlrap"],
                "best_lwlrap": metadata["best_lwlrap"],
                "final_lwlrap": metadata["final_lwlrap"],
                "best_epoch": metadata["best_epoch"],
                "submission_checkpoint": metadata.get("submission_checkpoint", "unknown"),
            }
        )
    if weighted_scores is None or targets is None:
        raise ValueError("no branch validation scores found")
    local_lwlrap = calculate_overall_lwlrap(targets, weighted_scores)
    return float(local_lwlrap), branch_rows


def main() -> None:
    args = _build_parser().parse_args()
    sample = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample)

    submissions = []
    for branch in BRANCHES:
        path = branch.run_dir(args.work_dir) / "submissions" / "small_logmel_cnn.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing branch submission: {path}")
        submission = pd.read_csv(path)
        validate_submission(submission, label_columns, expected_rows=len(sample))
        submissions.append(submission)

    ensemble_dir = args.work_dir / "runs" / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    blended = blend_submissions(
        submissions,
        weights=[branch.ensemble_weight for branch in BRANCHES],
        label_columns=label_columns,
    )
    ensemble_submission_path = ensemble_dir / "submission.csv"
    write_submission(blended, ensemble_submission_path, label_columns)
    submission_hash = sha256(ensemble_submission_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    delivery_submission_path = args.output_dir / "submission.csv"
    write_submission(blended, delivery_submission_path, label_columns)

    local_lwlrap, branch_rows = _blend_validation_scores(args.work_dir)

    results_path = args.output_dir / "pipeline_final_taller_2_v2_results.md"
    lines = [
        "# Noisy fine-tune results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Local validation",
        "",
        "| Branch | Weight | Baseline lwlrap | Best lwlrap | Final lwlrap | Best epoch | Submission ckpt |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in branch_rows:
        lines.append(
            "| {branch} | {weight:.3f} | {baseline_lwlrap:.6f} | {best_lwlrap:.6f} | {final_lwlrap:.6f} | {best_epoch} | {submission_checkpoint} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Ensemble validation lwlrap: `{local_lwlrap:.6f}`",
            "",
            "## Submission",
            "",
            f"- Path: `{ensemble_submission_path}`",
            f"- Delivery path: `{delivery_submission_path}`",
            f"- SHA256: `{submission_hash}`",
            f"- Rows: `{len(blended)}`",
            f"- Labels: `{len(label_columns)}`",
            "",
            "Kaggle is intentionally not submitted from this delivery script.",
            "",
        ]
    )
    results_path.write_text("\n".join(lines))
    print(f"wrote {ensemble_submission_path}")
    print(f"wrote {results_path}")
    print(f"ensemble_valid_lwlrap={local_lwlrap:.6f}")
    print(f"submission_sha256={submission_hash}")
    print("evaluate_and_blend_ok")


if __name__ == "__main__":
    main()
