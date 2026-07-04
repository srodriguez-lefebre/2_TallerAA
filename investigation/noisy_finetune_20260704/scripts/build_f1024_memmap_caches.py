from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_config import DATA_DIR

sys.path.insert(0, str(DATA_DIR.parent / "investigation"))

from scripts.build_logmel_image_cache import _extract_image  # noqa: E402
from scripts.fat2019.data import load_training_labels  # noqa: E402
from scripts.fat2019.submission import read_sample_submission  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build f1024 clip-normalized memmap caches.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--frames", type=int, default=1024)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    return parser


def memmap_cache_paths(*, split: str, n_mels: int, frames: int) -> tuple[Path, Path]:
    stem = f"{split}_logmel_image_m{n_mels}_f{frames}_x"
    return DATA_DIR / f"{stem}.npy", DATA_DIR / f"{stem}_fnames.txt"


def _split_fnames(split: str) -> list[str]:
    if split == "curated":
        return load_training_labels(DATA_DIR / "train_curated.csv")["fname"].astype(str).tolist()
    if split == "noisy":
        return load_training_labels(DATA_DIR / "train_noisy.csv")["fname"].astype(str).tolist()
    if split == "test":
        return read_sample_submission(DATA_DIR / "sample_submission.csv")["fname"].astype(str).tolist()
    raise ValueError(f"unknown split: {split}")


def _build_split(
    split: str,
    *,
    sample_rate: int,
    n_mels: int,
    frames: int,
    n_fft: int,
    hop_length: int,
    force: bool,
) -> None:
    x_path, fnames_path = memmap_cache_paths(split=split, n_mels=n_mels, frames=frames)
    fnames = _split_fnames(split)
    if x_path.exists() and fnames_path.exists() and not force:
        existing = np.load(x_path, mmap_mode="r")
        if existing.shape == (len(fnames), n_mels, frames):
            print(f"skip existing memmap {x_path} {existing.shape}")
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

    x_path.parent.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(fnames), n_mels, frames),
    )
    for row_index, fname in enumerate(fnames, start=1):
        images[row_index - 1] = _extract_image(
            DATA_DIR / fname,
            mel_transform=mel_transform,
            amplitude_to_db=amplitude_to_db,
            sample_rate=sample_rate,
            frames=frames,
            normalization="clip",
            global_mean=None,
            global_std=None,
        )
        if row_index % 250 == 0:
            print(f"{x_path.name}: {row_index}/{len(fnames)}", flush=True)
    images.flush()
    del images
    fnames_path.write_text("\n".join(fnames) + "\n")
    print(f"wrote memmap {x_path} rows={len(fnames)}")


def main() -> None:
    args = _build_parser().parse_args()
    for split in ("curated", "noisy", "test"):
        _build_split(
            split,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            frames=args.frames,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            force=args.force,
        )
    print("f1024_memmap_caches_ok")


if __name__ == "__main__":
    main()

