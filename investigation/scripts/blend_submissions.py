from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fat2019.submission import (
    blend_submissions,
    label_columns_from_sample,
    read_sample_submission,
    write_submission,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blend FAT2019 submission CSV files.")
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Submission CSV to include. Pass once per input.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=float,
        required=True,
        help="Weight for the corresponding --input. Pass once per input.",
    )
    parser.add_argument("--sample", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if len(args.input) != len(args.weight):
        raise ValueError("--input and --weight must be passed the same number of times")

    sample = read_sample_submission(args.sample)
    label_columns = label_columns_from_sample(sample)
    submissions = [pd.read_csv(path) for path in args.input]
    blended = blend_submissions(
        submissions,
        weights=args.weight,
        label_columns=label_columns,
    )
    write_submission(blended, args.output, label_columns)
    print(f"wrote {args.output} from {len(submissions)} submissions")


if __name__ == "__main__":
    main()
