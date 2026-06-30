from __future__ import annotations

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
