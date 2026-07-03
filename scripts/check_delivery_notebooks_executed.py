#!/usr/bin/env python3
"""Check that delivery notebooks were executed and saved with outputs.

The 100. Entregable notebook is intentionally excluded because it is meant to be
run later in a safer environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="+", required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    return parser.parse_args()


def notebook_status(path: Path) -> tuple[bool, str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
    if not code_cells:
        return True, "no code cells"

    missing_execution = [
        index
        for index, cell in enumerate(nb.get("cells", []), start=1)
        if cell.get("cell_type") == "code" and cell.get("execution_count") is None
    ]
    if missing_execution:
        return False, f"code cells without execution_count: {missing_execution}"

    output_count = sum(1 for cell in code_cells if cell.get("outputs"))
    return True, f"{len(code_cells)} executed code cells, {output_count} with outputs"


def main() -> int:
    args = parse_args()
    excluded = {Path(item).as_posix().rstrip("/") for item in args.exclude}
    failures: list[str] = []

    for folder_name in args.include:
        folder = Path(folder_name)
        if folder.as_posix().rstrip("/") in excluded:
            continue
        if not folder.exists():
            failures.append(f"{folder}: folder does not exist")
            continue
        notebooks = sorted(folder.glob("*.ipynb"))
        if not notebooks:
            failures.append(f"{folder}: no notebooks found")
            continue
        for notebook in notebooks:
            ok, message = notebook_status(notebook)
            prefix = "OK" if ok else "FAIL"
            print(f"{prefix}: {notebook}: {message}")
            if not ok:
                failures.append(f"{notebook}: {message}")

    if failures:
        print("\nNotebook execution check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nnotebook_execution_check_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
