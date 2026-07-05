from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
INVESTIGATION_DIR = REPO_ROOT / "investigation"
if str(INVESTIGATION_DIR) not in sys.path:
    sys.path.insert(0, str(INVESTIGATION_DIR))

from scripts.build_logmel_image_cache import _extract_image  # noqa: E402
from scripts.fat2019.neural_helpers import sigmoid_numpy  # noqa: E402
from scripts.fat2019.submission import label_columns_from_sample, validate_submission  # noqa: E402
from scripts.train_logmel_cnn import LogmelDataset, SmallLogmelCnn  # noqa: E402


WORK_DIR = INVESTIGATION_DIR / "kaggle_taa_2026_check"
EXPERIMENT_DIR = INVESTIGATION_DIR / "noisy_finetune_20260704"
SAMPLE_PATH = WORK_DIR / "sample_submission_v24.csv"
AUDIO_DIR = WORK_DIR / "audio_missing"
OUTPUT_DIR = WORK_DIR / "generated_v24"
OUTPUT_PATH = OUTPUT_DIR / "Grupo_A_sub_1.csv"


@dataclass(frozen=True)
class Branch:
    name: str
    weight: float
    frames: int
    normalization: str
    old_submission: Path
    checkpoint: Path


BRANCHES = (
    Branch(
        name="separable_headsep",
        weight=0.25,
        frames=512,
        normalization="clip",
        old_submission=EXPERIMENT_DIR / "runs/separable_headsep/submissions/small_logmel_cnn.csv",
        checkpoint=EXPERIMENT_DIR / "runs/separable_headsep/models/small_logmel_cnn_final.pt",
    ),
    Branch(
        name="globalmel_sep_temporal",
        weight=0.375,
        frames=512,
        normalization="global-mel",
        old_submission=EXPERIMENT_DIR / "runs/globalmel_sep_temporal/submissions/small_logmel_cnn.csv",
        checkpoint=EXPERIMENT_DIR / "runs/globalmel_sep_temporal/models/small_logmel_cnn_final.pt",
    ),
    Branch(
        name="sep_temporal_f1024",
        weight=0.375,
        frames=1024,
        normalization="clip",
        old_submission=EXPERIMENT_DIR / "runs/sep_temporal_f1024/submissions/small_logmel_cnn.csv",
        checkpoint=EXPERIMENT_DIR / "runs/sep_temporal_f1024/models/small_logmel_cnn_final.pt",
    ),
)


def torch_load(path: Path, *, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_global_mel_stats() -> tuple[np.ndarray, np.ndarray]:
    cache = np.load(REPO_ROOT / "data/test_logmel_image_m128_f512_globalmel.npz", allow_pickle=False)
    return cache["global_mean"], cache["global_std"]


def missing_fnames(sample: pd.DataFrame, old_submission: pd.DataFrame) -> list[str]:
    old_names = set(old_submission["fname"].astype(str))
    return [fname for fname in sample["fname"].astype(str) if fname not in old_names]


def build_missing_images(
    fnames: list[str],
    *,
    frames: int,
    normalization: str,
) -> np.ndarray:
    if normalization == "global-mel":
        global_mean, global_std = load_global_mel_stats()
    else:
        global_mean = None
        global_std = None

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        f_min=20.0,
        f_max=44100 / 2,
        power=2.0,
        normalized=False,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    images = np.empty((len(fnames), 128, frames), dtype=np.float16)
    for row_index, fname in enumerate(fnames, start=1):
        audio_path = AUDIO_DIR / fname
        if not audio_path.exists():
            raise FileNotFoundError(f"missing audio for v24 row: {audio_path}")
        images[row_index - 1] = _extract_image(
            audio_path,
            mel_transform=mel_transform,
            amplitude_to_db=amplitude_to_db,
            sample_rate=44100,
            frames=frames,
            normalization=normalization,
            global_mean=global_mean,
            global_std=global_std,
        )
        if row_index % 100 == 0:
            print(f"built {normalization} f{frames}: {row_index}/{len(fnames)}", flush=True)
    return images


def predict_missing_scores(
    branch: Branch,
    fnames: list[str],
    label_columns: list[str],
    *,
    device: torch.device,
) -> pd.DataFrame:
    checkpoint = torch_load(branch.checkpoint, device=device)
    checkpoint_labels = list(checkpoint["label_columns"])
    if checkpoint_labels != label_columns:
        raise ValueError(f"{branch.name}: checkpoint labels do not match sample")

    model = SmallLogmelCnn(
        num_classes=len(label_columns),
        architecture=str(checkpoint.get("architecture", "standard")),
        activation=str(checkpoint.get("activation", "silu")),
        block_dropout=float(checkpoint.get("block_dropout", 0.0)),
        head_hidden=int(checkpoint.get("head_hidden", 0)),
        head_dropout=float(checkpoint.get("head_dropout", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    images = build_missing_images(
        fnames,
        frames=branch.frames,
        normalization=branch.normalization,
    )
    loader = DataLoader(
        LogmelDataset(images, augment=False),
        batch_size=12 if branch.frames == 1024 else 24,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_images = batch[0] if isinstance(batch, (list, tuple)) else batch
            output = model(batch_images.to(device, non_blocking=True))
            logits.append(output.detach().cpu().numpy())
    scores = sigmoid_numpy(np.vstack(logits))
    output = pd.DataFrame(np.clip(scores, 0.0, 1.0), columns=label_columns)
    output.insert(0, "fname", np.array(fnames, dtype=str))
    validate_submission(output, label_columns, expected_rows=len(fnames))
    return output


def build_branch_submission(
    branch: Branch,
    sample: pd.DataFrame,
    label_columns: list[str],
    *,
    device: torch.device,
) -> pd.DataFrame:
    old_submission = pd.read_csv(branch.old_submission)
    validate_submission(old_submission, label_columns)
    missing = missing_fnames(sample, old_submission)
    print(f"{branch.name}: old_rows={len(old_submission)} missing_rows={len(missing)}", flush=True)
    missing_submission = predict_missing_scores(branch, missing, label_columns, device=device)

    combined = pd.concat([old_submission, missing_submission], ignore_index=True)
    combined = combined.set_index("fname").loc[sample["fname"].astype(str)].reset_index()
    validate_submission(combined, label_columns, expected_rows=len(sample))
    branch_path = OUTPUT_DIR / f"{branch.name}_v24.csv"
    combined.to_csv(branch_path, index=False)
    print(f"wrote {branch_path}", flush=True)
    return combined


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(SAMPLE_PATH)
    label_columns = label_columns_from_sample(sample)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    branch_submissions: list[pd.DataFrame] = []
    weights: list[float] = []
    for branch in BRANCHES:
        branch_submissions.append(
            build_branch_submission(branch, sample, label_columns, device=device)
        )
        weights.append(branch.weight)

    weight_sum = sum(weights)
    scores = np.zeros((len(sample), len(label_columns)), dtype=np.float64)
    for branch, submission in zip(BRANCHES, branch_submissions):
        if not submission["fname"].equals(sample["fname"]):
            raise ValueError(f"{branch.name}: fname order mismatch")
        scores += (branch.weight / weight_sum) * submission[label_columns].to_numpy(dtype=np.float64)

    blended = pd.DataFrame(np.clip(scores, 0.0, 1.0), columns=label_columns)
    blended.insert(0, "fname", sample["fname"].astype(str).to_numpy())
    validate_submission(blended, label_columns, expected_rows=len(sample))
    blended.to_csv(OUTPUT_PATH, index=False)
    metadata = {
        "output_path": str(OUTPUT_PATH),
        "sample_path": str(SAMPLE_PATH),
        "rows": int(len(blended)),
        "labels": int(len(label_columns)),
        "branches": [
            {"name": branch.name, "weight": branch.weight, "frames": branch.frames}
            for branch in BRANCHES
        ],
    }
    (OUTPUT_DIR / "Grupo_A_sub_1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
