from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.io import wavfile


def _hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    if fmax is None:
        fmax = sample_rate / 2
    if not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError("expected 0 <= fmin < fmax <= sample_rate / 2")

    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    fft_frequencies = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)

    filters = np.zeros((len(fft_frequencies), n_mels), dtype=np.float32)
    for mel_index in range(n_mels):
        lower = hz_points[mel_index]
        center = hz_points[mel_index + 1]
        upper = hz_points[mel_index + 2]

        left_slope = (fft_frequencies - lower) / (center - lower)
        right_slope = (upper - fft_frequencies) / (upper - center)
        filters[:, mel_index] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    return filters


def log_mel_spectrogram(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float | None = 4000.0,
    eps: float = 1e-6,
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be mono with shape (samples,), got {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("waveform must not be empty")

    _, _, stft = signal.stft(
        waveform,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft).astype(np.float32)
    filters = mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = filters.T @ magnitude
    return np.log(mel + eps).astype(np.float32)


def extract_log_mel_stats(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    log_mel = log_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    stats = [
        np.mean(log_mel, axis=1),
        np.std(log_mel, axis=1),
        np.max(log_mel, axis=1),
        np.percentile(log_mel, 75, axis=1),
    ]
    waveform_float = np.asarray(waveform, dtype=np.float32)
    duration = np.array([waveform_float.size / sample_rate], dtype=np.float32)
    rms = np.array([np.sqrt(np.mean(np.square(waveform_float)))], dtype=np.float32)
    zero_crossing_rate = np.array(
        [np.mean(np.diff(np.signbit(waveform_float)).astype(np.float32))],
        dtype=np.float32,
    )
    return np.concatenate([*stats, duration, rms, zero_crossing_rate]).astype(np.float32)


def extract_log_mel_stats_extended(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 80,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> np.ndarray:
    log_mel = log_mel_spectrogram(
        waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    stats = [
        np.mean(log_mel, axis=1),
        np.std(log_mel, axis=1),
        np.min(log_mel, axis=1),
        np.percentile(log_mel, 25, axis=1),
        np.percentile(log_mel, 50, axis=1),
        np.percentile(log_mel, 75, axis=1),
        np.max(log_mel, axis=1),
    ]
    waveform_float = np.asarray(waveform, dtype=np.float32)
    duration = np.array([waveform_float.size / sample_rate], dtype=np.float32)
    rms = np.array([np.sqrt(np.mean(np.square(waveform_float)))], dtype=np.float32)
    zero_crossing_rate = np.array(
        [np.mean(np.diff(np.signbit(waveform_float)).astype(np.float32))],
        dtype=np.float32,
    )
    return np.concatenate([*stats, duration, rms, zero_crossing_rate]).astype(np.float32)


def read_wav_mono(path: str) -> tuple[int, np.ndarray]:
    sample_rate, waveform = wavfile.read(path)
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    if np.issubdtype(waveform.dtype, np.integer):
        max_value = np.iinfo(waveform.dtype).max
        waveform = waveform.astype(np.float32) / max_value
    else:
        waveform = waveform.astype(np.float32)
    return int(sample_rate), waveform
