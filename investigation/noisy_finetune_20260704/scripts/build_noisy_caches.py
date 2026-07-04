from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_config import BRANCHES, DATA_DIR

sys.path.insert(0, str(DATA_DIR.parent / "investigation"))

from scripts.build_logmel_image_cache import (  # noqa: E402
    build_cache,
    compute_global_mel_stats,
    logmel_image_cache_path,
)
from scripts.fat2019.data import load_training_labels  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build noisy log-mel image caches for the fine-tuning experiment.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=512)
    return parser


def _global_stats_from_curated_cache(
    *,
    n_mels: int,
    frames: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    curated_cache_path = logmel_image_cache_path(
        DATA_DIR,
        split="curated",
        n_mels=n_mels,
        frames=frames,
        tag="globalmel",
    )
    if not curated_cache_path.exists():
        return None, None
    cache = np.load(curated_cache_path, allow_pickle=False)
    if "global_mean" not in cache or "global_std" not in cache:
        return None, None
    return cache["global_mean"].astype(np.float32), cache["global_std"].astype(np.float32)


def _compute_global_stats(
    *,
    sample_rate: int,
    n_mels: int,
    frames: int,
    n_fft: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torchaudio

    labels = load_training_labels(DATA_DIR / "train_curated.csv")
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
    return compute_global_mel_stats(
        DATA_DIR,
        labels["fname"].astype(str).tolist(),
        mel_transform=mel_transform,
        amplitude_to_db=amplitude_to_db,
        sample_rate=sample_rate,
        frames=frames,
    )


def main() -> None:
    args = _build_parser().parse_args()
    noisy_labels = load_training_labels(DATA_DIR / "train_noisy.csv")
    noisy_fnames = noisy_labels["fname"].astype(str).tolist()
    unique_configs = sorted(
        {(branch.n_mels, branch.frames, branch.cache_tag) for branch in BRANCHES},
        key=lambda item: (item[0], item[1], item[2] or ""),
    )

    for n_mels, frames, cache_tag in unique_configs:
        normalization = "global-mel" if cache_tag == "globalmel" else "clip"
        global_mean = None
        global_std = None
        if normalization == "global-mel":
            global_mean, global_std = _global_stats_from_curated_cache(
                n_mels=n_mels,
                frames=frames,
            )
            if global_mean is None or global_std is None:
                global_mean, global_std = _compute_global_stats(
                    sample_rate=args.sample_rate,
                    n_mels=n_mels,
                    frames=frames,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                )

        output_path = logmel_image_cache_path(
            DATA_DIR,
            split="noisy",
            n_mels=n_mels,
            frames=frames,
            tag=cache_tag,
        )
        build_cache(
            DATA_DIR,
            noisy_fnames,
            split="noisy",
            sample_rate=args.sample_rate,
            n_mels=n_mels,
            frames=frames,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            normalization=normalization,
            cache_tag=cache_tag,
            global_mean=global_mean,
            global_std=global_std,
            force=args.force,
        )
        cache = np.load(output_path, allow_pickle=False)
        if len(cache["x"]) != len(noisy_fnames):
            raise ValueError(f"{output_path} row mismatch: {len(cache['x'])} vs {len(noisy_fnames)}")
        print(f"cache_ok {output_path} rows={len(cache['x'])}")

    print("noisy_caches_ok")


if __name__ == "__main__":
    main()

