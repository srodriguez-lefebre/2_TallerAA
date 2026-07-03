#!/usr/bin/env python3
"""Validate the selected final submission for the delivery narrative."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab"
EXPECTED_PRIVATE_LB = 0.67126

FINAL_PATHS = {
    "07_final/submission.csv": ROOT / "07_final" / "submission.csv",
    "100. Entregable/submission.csv": ROOT / "100. Entregable" / "submission.csv",
    "source_parallel100": ROOT
    / "investigation"
    / "results"
    / "submissions"
    / "parallel100_20260702"
    / "e100_headsep25_globalmel375_f1024_375.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_submission(path: Path, sample: pd.DataFrame) -> dict[str, object]:
    df = pd.read_csv(path)
    label_columns = list(sample.columns[1:])
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "sha256": sha256(path),
        "sha256_matches_expected": sha256(path) == EXPECTED_SHA256,
        "rows": len(df),
        "columns": len(df.columns),
        "columns_match_sample": list(df.columns) == list(sample.columns),
        "fname_order_matches_sample": df["fname"].equals(sample["fname"]),
        "min_probability": float(df[label_columns].min().min()),
        "max_probability": float(df[label_columns].max().max()),
        "probabilities_in_range": bool(
            df[label_columns].ge(0.0).all().all() and df[label_columns].le(1.0).all().all()
        ),
    }


def main() -> int:
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    results = []

    for path in FINAL_PATHS.values():
        if not path.exists():
            results.append({"path": str(path.relative_to(ROOT)), "exists": False})
            continue
        results.append(validate_submission(path, sample))

    hashes = {item.get("sha256") for item in results if item.get("exists")}
    ok = (
        len(hashes) == 1
        and hashes == {EXPECTED_SHA256}
        and all(item.get("rows") == 3361 for item in results)
        and all(item.get("columns") == 81 for item in results)
        and all(item.get("columns_match_sample") for item in results)
        and all(item.get("fname_order_matches_sample") for item in results)
        and all(item.get("probabilities_in_range") for item in results)
    )

    payload = {
        "selected_final": {
            "formula": "0.25*separable_headsep_e100_seed42 + 0.375*globalmel_sep_temporal_e100_seed42 + 0.375*sep_temporal_f1024_e100_seed42",
            "private_lb": EXPECTED_PRIVATE_LB,
            "expected_sha256": EXPECTED_SHA256,
        },
        "artifacts": results,
    }
    print(json.dumps(payload, indent=2))
    if ok:
        print("final_artifacts_validation_ok")
        return 0
    print("final_artifacts_validation_failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
