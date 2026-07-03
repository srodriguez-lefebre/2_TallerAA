from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


EXPECTED_SHA256 = "4247ab9ff6398fbb1b6af223629d004265e27bb6cbccabf53ec4969a96c61cab"
EXPECTED_ROWS = 3361
EXPECTED_COLUMNS = 81


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_root() -> Path:
    root = Path(__file__).resolve()
    for parent in [root, *root.parents]:
        if (parent / "data" / "sample_submission.csv").exists():
            return parent
    raise FileNotFoundError("No se encontro la raiz del repositorio")


def check_notebook_executed(path: Path) -> dict[str, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    missing = [i for i, cell in enumerate(code_cells) if cell.get("execution_count") is None]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if missing:
        raise AssertionError(f"{path} tiene celdas sin ejecutar: {missing}")
    if errors:
        raise AssertionError(f"{path} tiene outputs de error")
    return {"code_cells": len(code_cells), "error_outputs": len(errors)}


def check_submission(root: Path) -> dict[str, object]:
    submission_path = root / "proyecto_actual" / "codigo" / "submission.csv"
    sample_path = root / "data" / "sample_submission.csv"
    submission = pd.read_csv(submission_path)
    sample = pd.read_csv(sample_path)
    label_columns = list(sample.columns[1:])

    assert submission.shape == (EXPECTED_ROWS, EXPECTED_COLUMNS)
    assert list(submission.columns) == list(sample.columns)
    assert submission["fname"].equals(sample["fname"])
    assert submission[label_columns].ge(0.0).all().all()
    assert submission[label_columns].le(1.0).all().all()

    digest = sha256(submission_path)
    assert digest == EXPECTED_SHA256
    return {"shape": submission.shape, "sha256": digest}


def check_report(root: Path) -> dict[str, object]:
    report_dir = root / "proyecto_actual" / "informe"
    tex_path = report_dir / "informe.tex"
    pdf_path = report_dir / "informe.pdf"
    assert tex_path.exists()
    assert pdf_path.exists()
    completed = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        text=True,
        check=True,
        capture_output=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", completed.stdout, flags=re.MULTILINE)
    if not match:
        raise AssertionError("No se pudo leer la cantidad de paginas del informe")
    pages = int(match.group(1))
    assert pages <= 8
    return {"pdf": str(pdf_path.relative_to(root)), "pages": pages}


def check_presentation(root: Path) -> dict[str, object]:
    html_path = root / "proyecto_actual" / "presentacion" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    slides = len(re.findall(r'<section class="slide', html))
    images = re.findall(r'<img[^>]+src="([^"]+)"', html)
    missing = [src for src in images if not (html_path.parent / src).exists()]
    assert slides >= 10
    assert not missing
    assert "function show(index)" in html
    assert "0.67126" in html
    return {"slides": slides, "images": len(images)}


def main() -> None:
    root = repo_root()
    checks = {
        "notebook": check_notebook_executed(
            root / "proyecto_actual" / "codigo" / "pipeline_final_taller_2.ipynb"
        ),
        "submission": check_submission(root),
        "report": check_report(root),
        "presentation": check_presentation(root),
    }
    print(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    print("proyecto_actual_validation_ok")


if __name__ == "__main__":
    main()
