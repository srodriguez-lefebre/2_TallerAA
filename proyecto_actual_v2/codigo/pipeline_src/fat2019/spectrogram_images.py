from __future__ import annotations

import numpy as np


def crop_or_pad_frames(image: np.ndarray, *, frames: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")
    if frames <= 0:
        raise ValueError("frames must be positive")

    current_frames = image.shape[1]
    if current_frames == frames:
        return image.astype(np.float32, copy=False)
    if current_frames > frames:
        start = (current_frames - frames) // 2
        return image[:, start : start + frames].astype(np.float32, copy=False)

    output = np.zeros((image.shape[0], frames), dtype=np.float32)
    output[:, :current_frames] = image.astype(np.float32, copy=False)
    return output


def normalize_logmel_image(image: np.ndarray) -> np.ndarray:
    normalized = image.astype(np.float32, copy=False)
    mean = float(normalized.mean())
    std = float(normalized.std())
    return (normalized - mean) / max(std, 1e-6)


def normalize_logmel_image_with_stats(
    image: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {image.shape}")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (image.shape[0],):
        raise ValueError(f"mean must have shape ({image.shape[0]},), got {mean.shape}")
    if std.shape != (image.shape[0],):
        raise ValueError(f"std must have shape ({image.shape[0]},), got {std.shape}")

    safe_std = np.maximum(std, 1e-6)
    return ((image.astype(np.float32, copy=False) - mean[:, None]) / safe_std[:, None]).astype(
        np.float32,
        copy=False,
    )
