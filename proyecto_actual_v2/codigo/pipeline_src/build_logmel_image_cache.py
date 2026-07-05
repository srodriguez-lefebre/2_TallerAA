from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fat2019.data import load_training_labels
from fat2019.spectrogram_images import (
    crop_or_pad_frames,
    normalize_logmel_image,
    normalize_logmel_image_with_stats,
)
from fat2019.submission import read_sample_submission


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed-size log-mel image caches.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--splits", default="curated,test")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--normalization", choices=("clip", "global-mel"), default="clip")
    parser.add_argument("--cache-tag", default=None)
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
    tag: str | None = None,
) -> Path:
    suffix = f"_{tag}" if tag else ""
    return data_dir / f"{split}_logmel_image_m{n_mels}_f{frames}{suffix}.npz"


def _read_waveform(path: Path, *, sample_rate: int) -> Any:
    # Third-party tensor type is intentionally lazy so unit tests do not require torch.
    import torchaudio

    waveform, original_sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_sample_rate, sample_rate)
    return waveform


def _extract_logmel_image(
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
    return crop_or_pad_frames(logmel, frames=frames)


def _extract_image(
    path: Path,
    *,
    mel_transform: Any,
    amplitude_to_db: Any,
    sample_rate: int,
    frames: int,
    normalization: str,
    global_mean: np.ndarray | None,
    global_std: np.ndarray | None,
) -> np.ndarray:
    image = _extract_logmel_image(
        path,
        mel_transform=mel_transform,
        amplitude_to_db=amplitude_to_db,
        sample_rate=sample_rate,
        frames=frames,
    )
    if normalization == "clip":
        normalized = normalize_logmel_image(image)
    elif normalization == "global-mel":
        if global_mean is None or global_std is None:
            raise ValueError("global-mel normalization requires global mean and std")
        normalized = normalize_logmel_image_with_stats(image, mean=global_mean, std=global_std)
    else:
        raise ValueError(f"unknown normalization: {normalization}")
    return normalized.astype(np.float16)


def compute_global_mel_stats(
    data_dir: Path,
    fnames: list[str],
    *,
    mel_transform: Any,
    amplitude_to_db: Any,
    sample_rate: int,
    frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = None
    total_squared = None
    count = 0
    for row_index, fname in enumerate(fnames, start=1):
        image = _extract_logmel_image(
            data_dir / fname,
            mel_transform=mel_transform,
            amplitude_to_db=amplitude_to_db,
            sample_rate=sample_rate,
            frames=frames,
        ).astype(np.float64, copy=False)
        band_sum = image.sum(axis=1)
        band_squared_sum = np.square(image).sum(axis=1)
        total = band_sum if total is None else total + band_sum
        total_squared = (
            band_squared_sum if total_squared is None else total_squared + band_squared_sum
        )
        count += image.shape[1]
        if row_index % 250 == 0:
            print(f"global stats: {row_index}/{len(fnames)}", flush=True)

    if total is None or total_squared is None or count == 0:
        raise ValueError("cannot compute global stats from an empty split")

    mean = total / count
    variance = np.maximum(total_squared / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


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
    normalization: str,
    cache_tag: str | None,
    global_mean: np.ndarray | None,
    global_std: np.ndarray | None,
    force: bool,
) -> None:
    output_path = logmel_image_cache_path(
        data_dir,
        split=split,
        n_mels=n_mels,
        frames=frames,
        tag=cache_tag,
    )
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
            normalization=normalization,
            global_mean=global_mean,
            global_std=global_std,
        )
        if row_index % 250 == 0:
            print(f"{output_path.name}: {row_index}/{len(fnames)}", flush=True)

    metadata: dict[str, Any] = {"normalization": normalization}
    if global_mean is not None and global_std is not None:
        metadata["global_mean"] = global_mean
        metadata["global_std"] = global_std
    np.savez_compressed(output_path, x=images, fnames=np.array(fnames), **metadata)
    print(f"wrote {output_path} {images.shape}")


def main() -> None:
    args = _build_parser().parse_args()
    splits = parse_splits(args.splits)
    cache_tag = args.cache_tag
    if cache_tag is None and args.normalization == "global-mel":
        cache_tag = "globalmel"

    global_mean = None
    global_std = None
    if args.normalization == "global-mel":
        import torchaudio

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            f_min=20.0,
            f_max=args.sample_rate / 2,
            power=2.0,
            normalized=False,
        )
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        labels = load_training_labels(args.data_dir / "train_curated.csv")
        global_mean, global_std = compute_global_mel_stats(
            args.data_dir,
            labels["fname"].astype(str).tolist(),
            mel_transform=mel_transform,
            amplitude_to_db=amplitude_to_db,
            sample_rate=args.sample_rate,
            frames=args.frames,
        )

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
            normalization=args.normalization,
            cache_tag=cache_tag,
            global_mean=global_mean,
            global_std=global_std,
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
            normalization=args.normalization,
            cache_tag=cache_tag,
            global_mean=global_mean,
            global_std=global_std,
            force=args.force,
        )


if __name__ == "__main__":
    main()
