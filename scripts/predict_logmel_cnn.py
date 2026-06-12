from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_logmel_image_cache import logmel_image_cache_path
from scripts.fat2019.neural_helpers import sigmoid_numpy
from scripts.fat2019.submission import (
    label_columns_from_sample,
    read_sample_submission,
    write_submission,
)


def resolve_submission_path(
    *,
    submissions_dir: Path,
    checkpoint_path: Path,
    output_path: Path | None,
) -> Path:
    if output_path is not None:
        return output_path
    return submissions_dir / f"{checkpoint_path.stem}.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict FAT2019 submissions from a CNN checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions/logmel_cnn_predict"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments/logmel_cnn_predict"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--split", choices=("test", "curated"), default="test")
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    import torch
    from torch.utils.data import DataLoader

    from scripts.train_logmel_cnn import LogmelDataset, SmallLogmelCnn, _predict_logits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    cache = np.load(
        logmel_image_cache_path(
            args.data_dir,
            split=args.split,
            n_mels=args.n_mels,
            frames=args.frames,
        ),
        allow_pickle=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_labels = list(checkpoint["label_columns"])
    if checkpoint_labels != label_columns:
        raise ValueError("checkpoint label columns do not match sample submission")

    model = SmallLogmelCnn(
        num_classes=len(label_columns),
        architecture=str(checkpoint.get("architecture", "standard")),
        activation=str(checkpoint.get("activation", "silu")),
        block_dropout=float(checkpoint.get("block_dropout", 0.0)),
        head_hidden=int(checkpoint.get("head_hidden", 0)),
        head_dropout=float(checkpoint.get("head_dropout", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(
        LogmelDataset(cache["x"], augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    scores = sigmoid_numpy(_predict_logits(model, loader, device=device))
    submission = pd.DataFrame(np.clip(scores, 0.0, 1.0), columns=label_columns)
    if args.split == "test":
        submission.insert(0, "fname", sample_submission["fname"].astype(str).to_numpy())
    else:
        submission.insert(0, "fname", cache["fnames"].astype(str))

    output_path = resolve_submission_path(
        submissions_dir=args.submissions_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output_path,
    )
    write_submission(submission, output_path, label_columns)

    args.experiments_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.experiments_dir / f"{output_path.stem}_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "output_path": str(output_path),
                "split": args.split,
                "n_mels": args.n_mels,
                "frames": args.frames,
                "rows": int(len(submission)),
                "labels": len(label_columns),
                "device": str(device),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
