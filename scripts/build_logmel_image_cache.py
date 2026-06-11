from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fat2019.data import load_training_labels
from scripts.fat2019.spectrogram_images import crop_or_pad_frames, normalize_logmel_image
from scripts.fat2019.submission import read_sample_submission


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed-size log-mel image caches.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--splits", default="curated,test")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser


def parse_splits(raw_splits: str) -> list[str]:
    splits = [split.strip() for split in raw_splits.split(",") if split.strip()]
    valid_splits = {"curated", "test"}
    unknown_splits = sorted(set(splits) - valid_splits)
    if unknown_splits:
        raise ValueError(f"unknown cache split(s): {unknown_splits}")
    if not splits:
        raise ValueError("expected at least one cache split")
    return splits


def logmel_image_cache_path(
    data_dir: Path,
    *,
    split: str,
    n_mels: int,
    frames: int,
) -> Path:
    return data_dir / f"{split}_logmel_image_m{n_mels}_f{frames}.npz"


def _read_waveform(path: Path, *, sample_rate: int) -> Any:
    # Third-party tensor type is intentionally lazy so unit tests do not require torch.
    import torchaudio

    waveform, original_sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_sample_rate, sample_rate)
    return waveform


def _extract_image(
    path: Path,
    *,
    mel_transform: Any,
    amplitude_to_db: Any,
    sample_rate: int,
    frames: int,
) -> np.ndarray:
    import torch

    waveform = _read_waveform(path, sample_rate=sample_rate)
    with torch.no_grad():
        mel = mel_transform(waveform)
        logmel = amplitude_to_db(mel).squeeze(0).cpu().numpy()
    image = crop_or_pad_frames(logmel, frames=frames)
    return normalize_logmel_image(image).astype(np.float16)


def build_cache(
    data_dir: Path,
    fnames: list[str],
    *,
    split: str,
    sample_rate: int,
    n_mels: int,
    frames: int,
    n_fft: int,
    hop_length: int,
    force: bool,
) -> None:
    output_path = logmel_image_cache_path(data_dir, split=split, n_mels=n_mels, frames=frames)
    if output_path.exists() and not force:
        print(f"skip existing {output_path}")
        return

    import torchaudio

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=20.0,
        f_max=sample_rate / 2,
        power=2.0,
        normalized=False,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    images = np.empty((len(fnames), n_mels, frames), dtype=np.float16)
    for row_index, fname in enumerate(fnames, start=1):
        images[row_index - 1] = _extract_image(
            data_dir / fname,
            mel_transform=mel_transform,
            amplitude_to_db=amplitude_to_db,
            sample_rate=sample_rate,
            frames=frames,
        )
        if row_index % 250 == 0:
            print(f"{output_path.name}: {row_index}/{len(fnames)}", flush=True)

    np.savez_compressed(output_path, x=images, fnames=np.array(fnames))
    print(f"wrote {output_path} {images.shape}")


def main() -> None:
    args = _build_parser().parse_args()
    splits = parse_splits(args.splits)

    if "curated" in splits:
        labels = load_training_labels(args.data_dir / "train_curated.csv")
        build_cache(
            args.data_dir,
            labels["fname"].astype(str).tolist(),
            split="curated",
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            frames=args.frames,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            force=args.force,
        )
    if "test" in splits:
        sample = read_sample_submission(args.data_dir / "sample_submission.csv")
        build_cache(
            args.data_dir,
            sample["fname"].astype(str).tolist(),
            split="test",
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            frames=args.frames,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            force=args.force,
        )


if __name__ == "__main__":
    main()
