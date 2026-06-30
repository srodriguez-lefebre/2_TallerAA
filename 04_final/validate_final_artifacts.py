from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "sample_submission.csv").exists() and (
            candidate / "04_final" / "final_pipeline_manifest.csv"
        ).exists():
            return candidate
    raise FileNotFoundError("Could not find project root")


def expected_sha256(caveat: str) -> str | None:
    match = re.search(r"SHA256\s+([0-9a-f]{64})", str(caveat))
    return match.group(1) if match else None


def validate_submission(path: Path, sample: pd.DataFrame) -> dict[str, object]:
    submission = pd.read_csv(path)
    label_columns = list(sample.columns[1:])
    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"{path} columns do not match sample_submission.csv")
    if len(submission) != len(sample):
        raise ValueError(f"{path} row count does not match sample_submission.csv")
    probabilities = submission[label_columns]
    if not probabilities.ge(0.0).all().all() or not probabilities.le(1.0).all().all():
        raise ValueError(f"{path} has probabilities outside [0, 1]")
    return {
        "rows": int(len(submission)),
        "columns": int(len(submission.columns)),
    }


def should_validate_as_submission(row: dict[str, object], artifact: Path) -> bool:
    if artifact.suffix.lower() != ".csv":
        return False
    searchable = " ".join(
        str(row.get(key, "")) for key in ["component", "role", "status"]
    ).lower()
    return "submission" in searchable


def validate(root: Path) -> dict[str, object]:
    manifest_path = root / "04_final" / "final_pipeline_manifest.csv"
    sample = pd.read_csv(root / "data" / "sample_submission.csv")
    manifest = pd.read_csv(manifest_path)

    rows: list[dict[str, object]] = []
    for row in manifest.to_dict(orient="records"):
        artifact = root / str(row["artifact"])
        artifact_info: dict[str, object] = {
            "component": row["component"],
            "status": row["status"],
            "artifact": row["artifact"],
            "exists": artifact.exists(),
        }
        if not artifact.exists():
            rows.append(artifact_info)
            continue

        if artifact.is_file():
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            artifact_info["sha256"] = digest
            expected_digest = expected_sha256(str(row.get("caveat", "")))
            if expected_digest is not None:
                artifact_info["sha256_matches_manifest"] = digest == expected_digest
                if digest != expected_digest:
                    raise ValueError(f"{artifact} sha256 mismatch")
            if should_validate_as_submission(row, artifact):
                artifact_info.update(validate_submission(artifact, sample))
        rows.append(artifact_info)

    missing = [row for row in rows if not row["exists"]]
    if missing:
        raise FileNotFoundError(f"Missing manifest artifacts: {missing}")
    return {
        "manifest": str(manifest_path),
        "components": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final Project 2 artifacts.")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    root = find_project_root(args.root)
    result = validate(root)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
