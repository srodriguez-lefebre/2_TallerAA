from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")
BAD_FILES = {
    "f76181c4.wav",
    "77b925c2.wav",
    "6a1f682a.wav",
    "c7db12aa.wav",
    "7752cc8a.wav",
    "1d44b0bd.wav",
}


def split_labels(labels: str) -> list[str]:
    if not isinstance(labels, str) or not labels.strip():
        return []
    return [label.strip() for label in labels.split(",") if label.strip()]


def find_input_dir() -> Path:
    candidates = sorted(KAGGLE_INPUT_ROOT.rglob("sample_submission.csv"))
    if candidates:
        input_dir = candidates[0].parent
        print(f"using input dir: {input_dir}")
        return input_dir

    print("sample_submission.csv was not found under /kaggle/input")
    for path in sorted(KAGGLE_INPUT_ROOT.rglob("*"))[:200]:
        print(path)
    raise FileNotFoundError("could not locate Freesound 2019 competition files")


def dataframe_to_multihot(labels: pd.DataFrame, label_columns: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(label_columns)}
    y = np.zeros((len(labels), len(label_columns)), dtype=np.float32)
    for row_index, row_labels in enumerate(labels["labels"]):
        for label in split_labels(row_labels):
            if label in label_to_index:
                y[row_index, label_to_index[label]] = 1.0
    return y


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(mels: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmin = 20.0
    fmax = sample_rate / 2
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
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


def log_mel_spectrogram(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    n_fft = 1024
    hop_length = 512
    _, _, stft = signal.stft(
        waveform.astype(np.float32),
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(stft).astype(np.float32)
    return np.log(mel_filterbank(sample_rate, n_fft, 80).T @ magnitude + 1e-6)


def extract_log_mel_stats(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    log_mel = log_mel_spectrogram(waveform, sample_rate)
    waveform = waveform.astype(np.float32)
    features = [
        np.mean(log_mel, axis=1),
        np.std(log_mel, axis=1),
        np.max(log_mel, axis=1),
        np.percentile(log_mel, 75, axis=1),
        np.array([waveform.size / sample_rate], dtype=np.float32),
        np.array([np.sqrt(np.mean(np.square(waveform)))], dtype=np.float32),
        np.array([np.mean(np.diff(np.signbit(waveform)).astype(np.float32))], dtype=np.float32),
    ]
    return np.concatenate(features).astype(np.float32)


def read_wav_from_zip(zip_file: zipfile.ZipFile, fname: str) -> tuple[int, np.ndarray]:
    with zip_file.open(fname) as wav_handle:
        sample_rate, waveform = wavfile.read(BytesIO(wav_handle.read()))
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1)
    if np.issubdtype(waveform.dtype, np.integer):
        waveform = waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    else:
        waveform = waveform.astype(np.float32)
    return int(sample_rate), waveform


def build_features(zip_path: Path, fnames: list[str]) -> np.ndarray:
    features = []
    with zipfile.ZipFile(zip_path) as zip_file:
        for row_index, fname in enumerate(fnames, start=1):
            sample_rate, waveform = read_wav_from_zip(zip_file, fname)
            features.append(extract_log_mel_stats(waveform, sample_rate))
            if row_index % 500 == 0:
                print(f"features {row_index}/{len(fnames)}")
    return np.vstack(features).astype(np.float32)


def class_priors(labels: pd.DataFrame, label_columns: list[str]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row_labels in labels["labels"]:
        counts.update(split_labels(row_labels))
    return {label: counts[label] / len(labels) for label in label_columns}


def main() -> None:
    input_dir = find_input_dir()
    sample = pd.read_csv(input_dir / "sample_submission.csv")
    label_columns = [column for column in sample.columns if column != "fname"]
    curated = pd.read_csv(input_dir / "train_curated.csv")
    curated = curated[~curated["fname"].isin(BAD_FILES)].copy()

    x_train = build_features(input_dir / "train_curated.zip", curated["fname"].astype(str).tolist())
    y_train = dataframe_to_multihot(curated, label_columns)
    x_test = build_features(input_dir / "test.zip", sample["fname"].astype(str).tolist())

    model = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=500,
                solver="liblinear",
            ),
            n_jobs=-1,
        ),
    )
    model.fit(x_train, y_train)
    scores = np.clip(model.predict_proba(x_test), 0.0, 1.0)

    submission = pd.DataFrame(scores, columns=label_columns)
    submission.insert(0, "fname", sample["fname"].astype(str).to_numpy())
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH} with shape {submission.shape}")


if __name__ == "__main__":
    main()
