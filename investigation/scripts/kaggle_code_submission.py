from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreparedKaggleSubmission:
    dataset_id: str
    kernel_id: str
    dataset_dir: Path
    kernel_dir: Path
    code_file: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("slug must contain at least one alphanumeric character")
    return slug


def _copy_kernel_code() -> str:
    return '''from __future__ import annotations

from pathlib import Path
import shutil


INPUT_ROOT = Path("/kaggle/input")
OUTPUT_PATH = Path("/kaggle/working/submission.csv")


def main() -> None:
    candidates = sorted(INPUT_ROOT.rglob("submission.csv"))
    if not candidates:
        print("submission.csv not found")
        for path in sorted(INPUT_ROOT.rglob("*"))[:200]:
            print(path)
        raise FileNotFoundError("submission.csv not found in kernel inputs")

    source = candidates[0]
    print(f"copying {source} to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH} with {OUTPUT_PATH.stat().st_size} bytes")


if __name__ == "__main__":
    main()
'''


def prepare_copy_kernel_submission(
    *,
    csv_path: Path,
    slug: str,
    title: str,
    owner: str,
    competition: str,
    work_dir: Path,
) -> PreparedKaggleSubmission:
    slug = slugify(slug)
    dataset_id = f"{owner}/{slug}"
    kernel_id = f"{owner}/{slug}-copy"
    dataset_dir = work_dir / f"kaggle_dataset_{slug.replace('-', '_')}"
    kernel_dir = work_dir / f"kaggle_kernel_{slug.replace('-', '_')}"
    code_file = kernel_dir / f"{slug.replace('-', '_')}_copy_submission.py"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_path, dataset_dir / "submission.csv")

    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": title,
                "id": dataset_id,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
        + "\n"
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_id,
                "title": f"{title} copy",
                "code_file": code_file.name,
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": False,
                "dataset_sources": [dataset_id],
                "competition_sources": [competition],
                "kernel_sources": [],
            },
            indent=2,
        )
        + "\n"
    )
    code_file.write_text(_copy_kernel_code())
    return PreparedKaggleSubmission(
        dataset_id=dataset_id,
        kernel_id=kernel_id,
        dataset_dir=dataset_dir,
        kernel_dir=kernel_dir,
        code_file=code_file,
    )


def _run_kaggle(args: list[str], *, kaggle_config_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)
    return subprocess.run(
        ["kaggle", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )


def wait_dataset_ready(
    dataset_id: str,
    *,
    kaggle_config_dir: Path,
    poll_seconds: int,
    max_polls: int,
) -> None:
    for poll_index in range(max_polls):
        command_failed = False
        try:
            status = _run_kaggle(
                ["datasets", "status", dataset_id],
                kaggle_config_dir=kaggle_config_dir,
            ).stdout
        except subprocess.CalledProcessError as exc:
            command_failed = True
            status = exc.stdout or str(exc)
        print(status.strip())
        if "ready" in status.lower():
            return
        if not command_failed and ("error" in status.lower() or "failed" in status.lower()):
            raise RuntimeError(f"Kaggle dataset failed: {status}")
        if poll_index == max_polls - 1:
            raise TimeoutError(f"Kaggle dataset did not become ready after {max_polls} polls")
        time.sleep(poll_seconds)


def submit_prepared(
    prepared: PreparedKaggleSubmission,
    *,
    competition: str,
    message: str,
    kaggle_config_dir: Path,
    poll_seconds: int,
    max_polls: int,
) -> None:
    try:
        print(
            _run_kaggle(
                ["datasets", "create", "-p", str(prepared.dataset_dir), "--dir-mode", "zip"],
                kaggle_config_dir=kaggle_config_dir,
            ).stdout
        )
    except subprocess.CalledProcessError:
        print(
            _run_kaggle(
                ["datasets", "version", "-p", str(prepared.dataset_dir), "-m", message, "--dir-mode", "zip"],
                kaggle_config_dir=kaggle_config_dir,
            ).stdout
        )
    wait_dataset_ready(
        prepared.dataset_id,
        kaggle_config_dir=kaggle_config_dir,
        poll_seconds=poll_seconds,
        max_polls=max_polls,
    )

    print(
        _run_kaggle(
            ["kernels", "push", "-p", str(prepared.kernel_dir)],
            kaggle_config_dir=kaggle_config_dir,
        ).stdout
    )
    for poll_index in range(max_polls):
        status = _run_kaggle(
            ["kernels", "status", prepared.kernel_id],
            kaggle_config_dir=kaggle_config_dir,
        ).stdout
        print(status.strip())
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "FAILED" in status:
            raise RuntimeError(f"Kaggle kernel failed: {status}")
        if poll_index == max_polls - 1:
            raise TimeoutError(f"Kaggle kernel did not complete after {max_polls} polls")
        time.sleep(poll_seconds)

    print(
        _run_kaggle(
            [
                "competitions",
                "submit",
                "-c",
                competition,
                "-k",
                prepared.kernel_id,
                "-v",
                "1",
                "-f",
                "submission.csv",
                "-m",
                message,
            ],
            kaggle_config_dir=kaggle_config_dir,
        ).stdout
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a CSV to Kaggle via copy kernel.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--owner", default="santiagorod247")
    parser.add_argument("--competition", default="freesound-audio-tagging-2019")
    parser.add_argument("--work-dir", type=Path, default=Path("investigation"))
    parser.add_argument("--kaggle-config-dir", type=Path, default=Path("keys"))
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-polls", type=int, default=30)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    prepared = prepare_copy_kernel_submission(
        csv_path=args.csv,
        slug=args.slug,
        title=args.title,
        owner=args.owner,
        competition=args.competition,
        work_dir=args.work_dir,
    )
    print(f"prepared dataset={prepared.dataset_id} kernel={prepared.kernel_id}")
    if args.prepare_only:
        return
    submit_prepared(
        prepared,
        competition=args.competition,
        message=args.message,
        kaggle_config_dir=args.kaggle_config_dir,
        poll_seconds=args.poll_seconds,
        max_polls=args.max_polls,
    )


if __name__ == "__main__":
    main()
