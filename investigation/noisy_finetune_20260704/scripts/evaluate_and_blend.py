from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_config import BRANCHES, DATA_DIR, EXPERIMENT_DIR, RUNS_DIR

sys.path.insert(0, str(DATA_DIR.parent / "investigation"))

from scripts.fat2019.metrics import calculate_overall_lwlrap  # noqa: E402
from scripts.fat2019.submission import (  # noqa: E402
    blend_submissions,
    label_columns_from_sample,
    read_sample_submission,
    validate_submission,
    write_submission,
)


COMPETITION = "freesound-audio-tagging-2019"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blend noisy fine-tuned branches and summarize metrics.")
    parser.add_argument("--submit-kaggle", action="store_true")
    parser.add_argument("--message", default="noisy_finetune_20260704 3-way same weights")
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_branch_metadata(branch_name: str) -> dict[str, object]:
    path = RUNS_DIR / branch_name / "experiments" / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"missing branch metadata: {path}")
    return json.loads(path.read_text())


def _blend_validation_scores() -> tuple[float, list[dict[str, object]]]:
    weighted_scores: np.ndarray | None = None
    targets: np.ndarray | None = None
    branch_rows: list[dict[str, object]] = []
    for branch in BRANCHES:
        experiments_dir = branch.run_dir / "experiments"
        scores = np.load(experiments_dir / "valid_scores_final.npy")
        branch_targets = np.load(experiments_dir / "valid_targets.npy")
        if targets is None:
            targets = branch_targets
        elif targets.shape != branch_targets.shape or not np.array_equal(targets, branch_targets):
            raise ValueError(f"validation targets differ for branch {branch.name}")
        if weighted_scores is None:
            weighted_scores = np.zeros_like(scores, dtype=np.float64)
        weighted_scores += branch.ensemble_weight * scores.astype(np.float64)
        metadata = _load_branch_metadata(branch.name)
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


def _run_kaggle_submission(submission_path: Path, *, message: str) -> str:
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        return "Kaggle CLI not found; submission not sent."

    submit_command = [
        kaggle_bin,
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-f",
        str(submission_path),
        "-m",
        message,
    ]
    submissions_command = [
        kaggle_bin,
        "competitions",
        "submissions",
        "-c",
        COMPETITION,
        "--csv",
    ]
    try:
        submit = subprocess.run(
            submit_command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        listing = subprocess.run(
            submissions_command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        return f"Kaggle command timed out: {exc}"

    return "\n".join(
        [
            "$ " + " ".join(submit_command),
            submit.stdout.strip(),
            "$ " + " ".join(submissions_command),
            "\n".join(listing.stdout.strip().splitlines()[:6]),
        ]
    ).strip()


def main() -> None:
    args = _build_parser().parse_args()
    sample = read_sample_submission(DATA_DIR / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample)

    submissions = []
    for branch in BRANCHES:
        path = branch.run_dir / "submissions" / "small_logmel_cnn.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing branch submission: {path}")
        submission = pd.read_csv(path)
        validate_submission(submission, label_columns, expected_rows=len(sample))
        submissions.append(submission)

    ensemble_dir = RUNS_DIR / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    blended = blend_submissions(
        submissions,
        weights=[branch.ensemble_weight for branch in BRANCHES],
        label_columns=label_columns,
    )
    ensemble_submission_path = ensemble_dir / "submission.csv"
    write_submission(blended, ensemble_submission_path, label_columns)
    submission_hash = sha256(ensemble_submission_path)

    local_lwlrap, branch_rows = _blend_validation_scores()
    kaggle_status = "Not submitted. Run with --submit-kaggle to submit to Freesound Audio Tagging 2019."
    if args.submit_kaggle:
        kaggle_status = _run_kaggle_submission(ensemble_submission_path, message=args.message)

    results_path = EXPERIMENT_DIR / "results.md"
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
            f"- SHA256: `{submission_hash}`",
            f"- Rows: `{len(blended)}`",
            f"- Labels: `{len(label_columns)}`",
            "",
            "## Kaggle",
            "",
            "```text",
            kaggle_status,
            "```",
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
