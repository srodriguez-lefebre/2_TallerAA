from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_config import BRANCHES, EXPERIMENT_DIR, SOURCE_STATES_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(source) == sha256(destination):
        return
    shutil.copy2(source, destination)


def metadata_summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    keys = [
        "architecture",
        "activation",
        "initializer",
        "head_hidden",
        "head_dropout",
        "optimizer",
        "scheduler",
        "full_train",
        "seed",
        "n_mels",
        "frames",
        "cache_tag",
        "best_epoch",
        "best_lwlrap",
    ]
    return {key: data.get(key) for key in keys if key in data}


def main() -> None:
    missing: list[Path] = []
    for branch in BRANCHES:
        for path in (
            branch.source_checkpoint,
            branch.source_metadata,
            branch.source_submission,
        ):
            if not path.exists():
                missing.append(path)
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"missing source artifacts:\n{formatted}")

    lines = [
        "# Source state manifest",
        "",
        "Estados fuente usados para el fine-tuning noisy.",
        "",
        "| Branch | Weight | Checkpoint SHA256 | Metadata SHA256 | Submission SHA256 | Config |",
        "|---|---:|---|---|---|---|",
    ]

    for branch in BRANCHES:
        branch_dir = SOURCE_STATES_DIR / branch.name
        copied_checkpoint = branch_dir / "small_logmel_cnn_best.pt"
        copied_metadata = branch_dir / "small_logmel_cnn_metadata.json"
        copied_submission = branch_dir / "small_logmel_cnn.csv"
        copy_artifact(branch.source_checkpoint, copied_checkpoint)
        copy_artifact(branch.source_metadata, copied_metadata)
        copy_artifact(branch.source_submission, copied_submission)

        summary = metadata_summary(branch.source_metadata)
        config_bits = ", ".join(f"{key}={value}" for key, value in summary.items())
        lines.append(
            "| {name} | {weight:.3f} | `{ck}` | `{meta}` | `{sub}` | {config} |".format(
                name=branch.name,
                weight=branch.ensemble_weight,
                ck=sha256(copied_checkpoint)[:16],
                meta=sha256(copied_metadata)[:16],
                sub=sha256(copied_submission)[:16],
                config=config_bits,
            )
        )

    SOURCE_STATES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SOURCE_STATES_DIR / "manifest.md"
    manifest.write_text("\n".join(lines) + "\n")
    print(f"wrote {manifest.relative_to(EXPERIMENT_DIR.parent.parent)}")
    print("source_state_manifest_ok")


if __name__ == "__main__":
    main()

