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


def parse_tta_views(raw_views: str) -> list[str]:
    valid_views = {"start", "center", "end"}
    views: list[str] = []
    for raw_view in raw_views.split(","):
        view = raw_view.strip()
        if not view:
            continue
        if view not in valid_views:
            raise ValueError(f"unknown TTA view: {view}")
        if view not in views:
            views.append(view)
    if not views:
        raise ValueError("expected at least one TTA view")
    return views


def select_frame_view(images: np.ndarray, *, frames: int, view: str) -> np.ndarray:
    if images.ndim != 3:
        raise ValueError(f"expected images with shape (rows, mels, frames), got {images.shape}")
    if frames <= 0:
        raise ValueError("frames must be positive")
    total_frames = images.shape[2]
    if total_frames == frames:
        return images
    if total_frames < frames:
        output = np.zeros((images.shape[0], images.shape[1], frames), dtype=images.dtype)
        output[:, :, :total_frames] = images
        return output
    if view == "start":
        start = 0
    elif view == "center":
        start = (total_frames - frames) // 2
    elif view == "end":
        start = total_frames - frames
    else:
        raise ValueError(f"unknown TTA view: {view}")
    return images[:, :, start : start + frames]


def enable_dropout_modules(model: "torch.nn.Module") -> None:
    import torch

    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


def _predict_logits_preserve_mode(
    model: "torch.nn.Module",
    loader: "DataLoader",
    *,
    device: "torch.device",
) -> np.ndarray:
    import torch

    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            output = model(images.to(device, non_blocking=True))
            logits.append(output.detach().cpu().numpy())
    return np.vstack(logits)


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
    parser.add_argument("--cache-tag", default=None)
    parser.add_argument("--checkpoint-frames", type=int, default=None)
    parser.add_argument("--tta-views", default="center")
    parser.add_argument("--mc-dropout-passes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    import torch
    from torch.utils.data import DataLoader

    from scripts.train_logmel_cnn import LogmelDataset, SmallLogmelCnn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_submission = read_sample_submission(args.data_dir / "sample_submission.csv")
    label_columns = label_columns_from_sample(sample_submission)
    cache = np.load(
        logmel_image_cache_path(
            args.data_dir,
            split=args.split,
            n_mels=args.n_mels,
            frames=args.frames,
            tag=args.cache_tag,
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
    if args.mc_dropout_passes < 0:
        raise ValueError("mc_dropout_passes must be non-negative")

    checkpoint_frames = (
        int(args.checkpoint_frames)
        if args.checkpoint_frames is not None
        else int(checkpoint.get("frames", args.frames))
    )
    tta_views = parse_tta_views(args.tta_views)
    score_views: list[np.ndarray] = []
    for view in tta_views:
        view_images = select_frame_view(cache["x"], frames=checkpoint_frames, view=view)
        loader = DataLoader(
            LogmelDataset(view_images, augment=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if args.mc_dropout_passes > 0:
            mc_scores = np.zeros((len(view_images), len(label_columns)), dtype=np.float64)
            for _pass_index in range(args.mc_dropout_passes):
                model.eval()
                enable_dropout_modules(model)
                logits = _predict_logits_preserve_mode(model, loader, device=device)
                mc_scores += sigmoid_numpy(logits)
            score_views.append(mc_scores / args.mc_dropout_passes)
        else:
            model.eval()
            logits = _predict_logits_preserve_mode(model, loader, device=device)
            score_views.append(sigmoid_numpy(logits))
    scores = np.mean(score_views, axis=0)
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
                "cache_tag": args.cache_tag,
                "checkpoint_frames": checkpoint_frames,
                "tta_views": tta_views,
                "mc_dropout_passes": args.mc_dropout_passes,
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
