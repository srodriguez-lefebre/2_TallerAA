from __future__ import annotations

import json
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = ROOT / "proyecto_actual_v2" / "codigo" / "pipeline_final_taller_2_v2.ipynb"
LONG_NOTEBOOK_PATH = (
    ROOT / "proyecto_actual_v2" / "codigo" / "pipeline_final_taller_2_v2_largo.ipynb"
)
PIPELINE_SRC = ROOT / "proyecto_actual_v2" / "codigo" / "pipeline_src"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        md(
            """
            # Pipeline final Taller 2 v2

            Este notebook es el codigo de entrega del proyecto de audio. Sigue la
            misma idea del pipeline final del Taller 1: parte de los datos crudos,
            fija una configuracion reproducible, define el pipeline completo y
            deja la generacion del archivo final en una unica interfaz.

            El modelo final es el ensamble `3-way` con fine-tuning sobre
            `train_curated + train_noisy`: tres ramas entrenadas primero con
            `train_curated` durante 100 epocas y luego continuadas 30 epocas con
            batches 50/50 curated/noisy, menor learning rate, scheduler coseno y
            menor peso para las etiquetas noisy.
            """
        ),
        md("## Librerias utilizadas"),
        code(
            """
            from __future__ import annotations

            import hashlib
            import json
            import os
            import subprocess
            import sys
            from dataclasses import asdict, dataclass
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            ROOT = Path.cwd()
            for candidate in [ROOT, *ROOT.parents]:
                if (candidate / "data" / "sample_submission.csv").exists():
                    ROOT = candidate
                    break
            else:
                raise FileNotFoundError("No se encontro data/sample_submission.csv")

            CODE_DIR = ROOT / "proyecto_actual_v2" / "codigo"
            PIPELINE_SRC = CODE_DIR / "pipeline_src"
            WORK_DIR = CODE_DIR / "work"
            FIGURES_DIR = CODE_DIR / "figures"
            DATA_DIR = ROOT / "data"
            PYTHON_EXE = ROOT / ".venv" / "bin" / "python"
            if not PYTHON_EXE.exists():
                PYTHON_EXE = Path(sys.executable)

            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            WORK_DIR.mkdir(parents=True, exist_ok=True)

            if str(PIPELINE_SRC) not in sys.path:
                sys.path.insert(0, str(PIPELINE_SRC))
            """
        ),
        md(
            """
            ## Configuracion fija del experimento final

            La configuracion queda congelada en el paquete de entrega. Las tres
            ramas comparten representacion log-mel 128 bandas, pero difieren en
            arquitectura/cabezal, normalizacion y ventana temporal. El ensamble
            conserva los pesos que ya habian dado el mejor `3-way` defendible:
            `0.25 / 0.375 / 0.375`.
            """
        ),
        code(
            """
            from final_config import BRANCHES, FINE_TUNE

            EXPECTED_FINAL_SHA256 = "b29288caa6e7b37b29e830a29655decc7c6bc8110ca3a40b828f4dd2f5fabdcc"
            FREESOUND2019_PRIVATE_LB = 0.68122
            COURSE_PUBLIC_LB = 0.64307
            PREVIOUS_3WAY_PRIVATE_LB = 0.67126

            branch_config = pd.DataFrame(
                [
                    {
                        "branch": branch.name,
                        "weight": branch.ensemble_weight,
                        "architecture": branch.architecture,
                        "activation": branch.activation,
                        "frames": branch.frames,
                        "normalization": "global-mel" if branch.cache_tag == "globalmel" else "clip",
                        "base_epochs": branch.base_epochs,
                        "batch_size": branch.batch_size,
                        "optimizer": branch.optimizer,
                        "scheduler": branch.scheduler,
                        "lr_milestones": ",".join(map(str, branch.lr_milestones)),
                        "head_hidden": branch.head_hidden,
                        "head_dropout": branch.head_dropout,
                    }
                    for branch in BRANCHES
                ]
            )

            finetune_config = {
                "epochs_extra": FINE_TUNE.epochs,
                "learning_rate": FINE_TUNE.lr,
                "scheduler": "cosine",
                "min_lr": FINE_TUNE.min_lr,
                "batch_composition": "50% curated / 50% noisy",
                "curated_loss_weight": FINE_TUNE.curated_loss_weight,
                "noisy_loss_weight": FINE_TUNE.noisy_loss_weight,
                "gaussian_noise_std": FINE_TUNE.gaussian_noise_std,
                "augmentations": list(FINE_TUNE.augmentations),
                "ensemble_weights": [branch.ensemble_weight for branch in BRANCHES],
                "freesound2019_private_lb": FREESOUND2019_PRIVATE_LB,
                "course_public_lb": COURSE_PUBLIC_LB,
            }

            display(branch_config)
            finetune_config
            """
        ),
        md(
            """
            ## Datos crudos y validaciones iniciales

            El pipeline asume la estructura original de Freesound Audio Tagging:
            metadatos `train_curated.csv`, `train_noisy.csv`, `sample_submission.csv`
            y archivos `.wav` en `train_curated/`, `train_noisy/` y `test/`.
            Antes de entrenar se valida el formato de columnas y la existencia de
            una muestra de archivos por split.
            """
        ),
        code(
            """
            sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
            curated = pd.read_csv(DATA_DIR / "train_curated.csv")
            noisy = pd.read_csv(DATA_DIR / "train_noisy.csv")
            label_columns = [column for column in sample.columns if column != "fname"]

            def split_labels(raw: str) -> list[str]:
                return [label.strip() for label in str(raw).split(",") if label.strip()]

            raw_checks = {
                "sample_rows": len(sample),
                "curated_rows": len(curated),
                "noisy_rows": len(noisy),
                "num_labels": len(label_columns),
                "curated_unique_files": curated["fname"].nunique(),
                "noisy_unique_files": noisy["fname"].nunique(),
                "sample_has_fname": "fname" in sample.columns,
                "curated_has_labels": {"fname", "labels"}.issubset(curated.columns),
                "noisy_has_labels": {"fname", "labels"}.issubset(noisy.columns),
            }

            for split_name, frame in [("train_curated", curated), ("train_noisy", noisy)]:
                missing = [
                    fname
                    for fname in frame["fname"].astype(str).head(20)
                    if not (DATA_DIR / fname).exists()
                ]
                if missing:
                    raise FileNotFoundError(f"{split_name}: faltan archivos de muestra: {missing[:3]}")

            missing_test = [
                fname for fname in sample["fname"].astype(str).head(20) if not (DATA_DIR / fname).exists()
            ]
            if missing_test:
                raise FileNotFoundError(f"test: faltan archivos de muestra: {missing_test[:3]}")

            assert raw_checks["num_labels"] == 80
            assert raw_checks["sample_rows"] == 3361
            assert raw_checks["curated_has_labels"]
            assert raw_checks["noisy_has_labels"]
            raw_checks
            """
        ),
        md(
            """
            ## Preprocesamiento log-mel

            Cada `.wav` se transforma en un espectrograma log-mel de tamano fijo.
            Las ramas de 512 frames usan caches comprimidos `.npz`; la rama de
            1024 frames usa memmap `.npy` para evitar picos de memoria. La rama
            `globalmel_sep_temporal` normaliza por estadisticos globales de banda
            mel calculados desde curated.
            """
        ),
        code(
            """
            def sha256(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()


            def run_command(command: list[str], *, enabled: bool) -> None:
                print("$ " + " ".join(map(str, command)))
                if not enabled:
                    print("skip: RUN_FULL_PIPELINE=False")
                    return
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                print(completed.stdout)
                if completed.returncode != 0:
                    raise RuntimeError(f"command failed with code {completed.returncode}")
            """
        ),
        md(
            """
            ## Modelo final y clase de pipeline

            La clase siguiente concentra el camino completo de entrega: construir
            caches desde audio crudo, entrenar las tres ramas curated e100, hacer
            fine-tuning mixto con noisy, recrear el ensamble y validar el CSV
            final. Por defecto no lanza el entrenamiento pesado al abrir el
            notebook; para reproducir desde cero hay que activar
            `RUN_FULL_PIPELINE=True` en la celda de ejecucion.
            """
        ),
        code(
            """
            @dataclass
            class PipelineStep:
                name: str
                command: list[str]


            class FinalAudioTaggingPipeline:
                def __init__(self, *, root: Path, data_dir: Path, code_dir: Path, work_dir: Path) -> None:
                    self.root = root
                    self.data_dir = data_dir
                    self.code_dir = code_dir
                    self.work_dir = work_dir
                    self.python = str(PYTHON_EXE)
                    self.src = code_dir / "pipeline_src"

                def cache_steps(self) -> list[PipelineStep]:
                    base = [self.python, str(self.src / "build_logmel_image_cache.py"), "--data-dir", str(self.data_dir)]
                    return [
                        PipelineStep(
                            "logmel_512_clip_curated_test",
                            base + ["--splits", "curated,test", "--n-mels", "128", "--frames", "512", "--normalization", "clip"],
                        ),
                        PipelineStep(
                            "logmel_512_globalmel_curated_test",
                            base + [
                                "--splits",
                                "curated,test",
                                "--n-mels",
                                "128",
                                "--frames",
                                "512",
                                "--normalization",
                                "global-mel",
                                "--cache-tag",
                                "globalmel",
                            ],
                        ),
                        PipelineStep(
                            "logmel_1024_clip_curated_test",
                            base + ["--splits", "curated,test", "--n-mels", "128", "--frames", "1024", "--normalization", "clip"],
                        ),
                        PipelineStep(
                            "logmel_noisy_512",
                            [self.python, str(self.src / "build_noisy_caches.py"), "--data-dir", str(self.data_dir)],
                        ),
                        PipelineStep(
                            "logmel_1024_memmap_all",
                            [self.python, str(self.src / "build_f1024_memmap_caches.py"), "--data-dir", str(self.data_dir)],
                        ),
                    ]

                def base_train_steps(self) -> list[PipelineStep]:
                    steps: list[PipelineStep] = []
                    for branch in BRANCHES:
                        command = [
                            self.python,
                            str(self.src / "train_logmel_cnn.py"),
                            "--data-dir",
                            str(self.data_dir),
                            "--models-dir",
                            str(self.work_dir / "models" / branch.source_run),
                            "--submissions-dir",
                            str(self.work_dir / "submissions" / branch.source_run),
                            "--experiments-dir",
                            str(self.work_dir / "experiments" / branch.source_run),
                            "--n-mels",
                            str(branch.n_mels),
                            "--frames",
                            str(branch.frames),
                            "--epochs",
                            str(branch.base_epochs),
                            "--batch-size",
                            str(branch.batch_size),
                            "--lr",
                            str(branch.base_lr),
                            "--weight-decay",
                            str(branch.base_weight_decay),
                            "--optimizer",
                            branch.optimizer,
                            "--initializer",
                            branch.initializer,
                            "--architecture",
                            branch.architecture,
                            "--activation",
                            branch.activation,
                            "--head-hidden",
                            str(branch.head_hidden),
                            "--head-dropout",
                            str(branch.head_dropout),
                            "--block-dropout",
                            str(branch.block_dropout),
                            "--time-reverse-probability",
                            str(branch.time_reverse_probability),
                            "--contrast-strength",
                            str(branch.contrast_strength),
                            "--scheduler",
                            branch.scheduler,
                            "--lr-milestones",
                            ",".join(map(str, branch.lr_milestones)),
                            "--min-lr",
                            str(FINE_TUNE.min_lr),
                            "--seed",
                            str(FINE_TUNE.seed),
                            "--full-train",
                        ]
                        if branch.cache_tag is not None:
                            command.extend(["--cache-tag", branch.cache_tag])
                        steps.append(PipelineStep(f"train_curated_{branch.name}", command))
                    return steps

                def finetune_steps(self) -> list[PipelineStep]:
                    return [
                        PipelineStep(
                            "finetune_curated_noisy",
                            [
                                self.python,
                                str(self.src / "finetune_mixed_noisy.py"),
                                "--data-dir",
                                str(self.data_dir),
                                "--work-dir",
                                str(self.work_dir),
                                "--branches",
                                "all",
                                "--epochs",
                                str(FINE_TUNE.epochs),
                                "--lr",
                                str(FINE_TUNE.lr),
                                "--min-lr",
                                str(FINE_TUNE.min_lr),
                                "--noisy-loss-weight",
                                str(FINE_TUNE.noisy_loss_weight),
                                "--curated-loss-weight",
                                str(FINE_TUNE.curated_loss_weight),
                                "--gaussian-noise-std",
                                str(FINE_TUNE.gaussian_noise_std),
                                "--seed",
                                str(FINE_TUNE.seed),
                                "--num-workers",
                                str(FINE_TUNE.num_workers),
                                "--submission-checkpoint",
                                FINE_TUNE.submission_checkpoint,
                            ],
                        ),
                        PipelineStep(
                            "blend_final_submission",
                            [
                                self.python,
                                str(self.src / "evaluate_and_blend.py"),
                                "--data-dir",
                                str(self.data_dir),
                                "--work-dir",
                                str(self.work_dir),
                                "--output-dir",
                                str(self.code_dir),
                            ],
                        ),
                    ]

                def steps(self) -> list[PipelineStep]:
                    return self.cache_steps() + self.base_train_steps() + self.finetune_steps()

                def run(self, *, enabled: bool) -> pd.DataFrame:
                    rows = []
                    for step in self.steps():
                        run_command(step.command, enabled=enabled)
                        rows.append({"step": step.name, "command": " ".join(map(str, step.command))})
                    return pd.DataFrame(rows)

                def validate_submission(self, path: Path) -> dict[str, object]:
                    if not path.exists():
                        raise FileNotFoundError(
                            f"No existe {path}. Ejecutar el pipeline completo para generarlo."
                        )
                    submission = pd.read_csv(path)
                    assert submission.shape == (len(sample), len(sample.columns))
                    assert list(submission.columns) == list(sample.columns)
                    assert submission["fname"].astype(str).equals(sample["fname"].astype(str))
                    values = submission[label_columns].to_numpy(dtype=float)
                    assert np.isfinite(values).all()
                    assert values.min() >= 0.0
                    assert values.max() <= 1.0
                    return {
                        "path": str(path.relative_to(ROOT)),
                        "rows": len(submission),
                        "labels": len(label_columns),
                        "min_probability": float(values.min()),
                        "max_probability": float(values.max()),
                        "sha256": sha256(path),
                    }


            pipeline = FinalAudioTaggingPipeline(
                root=ROOT,
                data_dir=DATA_DIR,
                code_dir=CODE_DIR,
                work_dir=WORK_DIR,
            )

            planned_steps = pd.DataFrame(
                [{"step": step.name, "command": " ".join(map(str, step.command))} for step in pipeline.steps()]
            )
            display(planned_steps)
            """
        ),
        md(
            """
            ## Entrenamiento y generacion de archivos finales

            Para una ejecucion completa desde cero, cambiar `RUN_FULL_PIPELINE` a
            `True` y correr la celda. Esto reconstruye caches desde `.wav`,
            entrena las ramas base, aplica el fine-tuning noisy y escribe
            `proyecto_actual_v2/codigo/submission.csv`.

            En esta copia de entrega el flag queda en `False` para no relanzar
            horas de entrenamiento al abrir el notebook; aun asi, la celda muestra
            exactamente los comandos del pipeline completo y valida el CSV final
            disponible.
            """
        ),
        code(
            """
            RUN_FULL_PIPELINE = False

            executed_steps = pipeline.run(enabled=RUN_FULL_PIPELINE)
            submission_checks = pipeline.validate_submission(CODE_DIR / "submission.csv")
            assert submission_checks["sha256"] == EXPECTED_FINAL_SHA256

            display(executed_steps)
            submission_checks
            """
        ),
        md("## Figuras y evidencia para entrega"),
        code(
            """
            plt.style.use("default")

            split_counts = pd.DataFrame(
                [
                    {"split": "train_curated", "rows": len(curated)},
                    {"split": "train_noisy", "rows": len(noisy)},
                    {"split": "test", "rows": len(sample)},
                ]
            )
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            colors = ["#2f6f73", "#d28b35", "#575a89"]
            ax.bar(split_counts["split"], split_counts["rows"], color=colors)
            ax.set_title("Tamanos de splits")
            ax.set_ylabel("audios")
            ax.grid(axis="y", alpha=0.25)
            for index, value in enumerate(split_counts["rows"]):
                ax.text(index, value + 300, f"{value}", ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig_v2_split_sizes.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.2, 3.5))
            ax.pie(
                branch_config["weight"],
                labels=branch_config["branch"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#4b7f52", "#386c9f", "#ba7a2a"],
            )
            ax.set_title("Pesos del ensamble final")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig_v2_component_weights.png", dpi=160)
            plt.close(fig)

            scores = pd.DataFrame(
                [
                    {"modelo": "3-way e100", "score": PREVIOUS_3WAY_PRIVATE_LB, "tipo": "Freesound 2019 private"},
                    {"modelo": "noisy fine-tune", "score": FREESOUND2019_PRIVATE_LB, "tipo": "Freesound 2019 private"},
                    {"modelo": "noisy fine-tune", "score": COURSE_PUBLIC_LB, "tipo": "Kaggle curso public"},
                ]
            )
            fig, ax = plt.subplots(figsize=(7.2, 3.7))
            labels = scores["modelo"] + " - " + scores["tipo"]
            ax.barh(labels, scores["score"], color=["#6b7280", "#1f7a5c", "#315c99"])
            ax.set_xlim(0.50, 0.70)
            ax.set_xlabel("score")
            ax.set_title("Resultados de leaderboard")
            ax.grid(axis="x", alpha=0.25)
            for index, value in enumerate(scores["score"]):
                ax.text(value + 0.004, index, f"{value:.5f}", va="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig_v2_kaggle_scores.png", dpi=160)
            plt.close(fig)

            local_metrics = pd.DataFrame(
                [
                    {"branch": "separable_headsep", "baseline_lwlrap": 0.999569, "final_lwlrap": 0.995336},
                    {"branch": "globalmel_sep_temporal", "baseline_lwlrap": 0.998965, "final_lwlrap": 0.995154},
                    {"branch": "sep_temporal_f1024", "baseline_lwlrap": 0.998850, "final_lwlrap": 0.995542},
                ]
            )
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            x = np.arange(len(local_metrics))
            width = 0.35
            ax.bar(x - width / 2, local_metrics["baseline_lwlrap"], width, label="checkpoint e100", color="#7a869a")
            ax.bar(x + width / 2, local_metrics["final_lwlrap"], width, label="tras noisy", color="#2f7d6d")
            ax.set_xticks(x)
            ax.set_xticklabels(local_metrics["branch"], rotation=15, ha="right")
            ax.set_ylim(0.990, 1.0005)
            ax.set_ylabel("lwlrap local")
            ax.set_title("Control local durante fine-tuning")
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig_v2_branch_local.png", dpi=160)
            plt.close(fig)

            figures = sorted(path.name for path in FIGURES_DIR.glob("fig_v2_*.png"))
            figures
            """
        ),
        md(
            """
            ## Revision final

            La metrica local del fine-tuning no se interpreta como validacion
            independiente: los checkpoints base fueron entrenados con
            `full_train=True` sobre curated. Sirve como control de degradacion.
            La decision final se apoya en Kaggle: `0.68122` en Freesound 2019
            private y `0.64307` en la variante del curso.
            """
        ),
        code(
            """
            metadata = {
                "pipeline": {
                    "self_contained_from_raw_data": True,
                    "uses_investigation_artifacts_as_source": False,
                    "run_full_pipeline_flag": RUN_FULL_PIPELINE,
                    "work_dir": str(WORK_DIR.relative_to(ROOT)),
                    "pipeline_src": str(PIPELINE_SRC.relative_to(ROOT)),
                },
                "raw_data_checks": raw_checks,
                "branch_config": branch_config.to_dict(orient="records"),
                "finetune_config": finetune_config,
                "submission": submission_checks,
                "leaderboard": {
                    "previous_3way_private_lb": PREVIOUS_3WAY_PRIVATE_LB,
                    "freesound2019_private_lb": FREESOUND2019_PRIVATE_LB,
                    "course_public_lb": COURSE_PUBLIC_LB,
                },
                "planned_steps": planned_steps.to_dict(orient="records"),
                "figures": figures,
            }
            metadata_path = CODE_DIR / "pipeline_final_taller_2_v2_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\\n")
            metadata
            """
        ),
    ]
    return nb


def embedded_source_mapping() -> dict[str, str]:
    files = sorted(
        path
        for path in PIPELINE_SRC.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    return {
        str(path.relative_to(PIPELINE_SRC)): path.read_text(encoding="utf-8")
        for path in files
    }


def build_long_notebook() -> nbf.NotebookNode:
    embedded_json = json.dumps(embedded_source_mapping(), indent=2, ensure_ascii=False)
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb["cells"] = [
        md(
            """
            # Pipeline final Taller 2 v2 - version larga

            Esta es la version autocontenida extendida del notebook de codigo. A
            diferencia de la version modular, aca se embebe el codigo completo del
            pipeline dentro del propio notebook: utilidades de datos, features
            log-mel, metricas, modelo, entrenamiento curated e100, fine-tuning con
            noisy y blend final.

            Para que el notebook pueda ejecutarse en modo liviano sin depender del
            kernel con Torch, las fuentes embebidas se escriben en
            `codigo/work/_embedded_pipeline_src` y los pasos pesados se lanzan con
            `.venv/bin/python` solamente cuando `RUN_FULL_PIPELINE=True`.
            """
        ),
        md("## Librerias y rutas"),
        code(
            """
            from __future__ import annotations

            import hashlib
            import json
            import subprocess
            import sys
            from dataclasses import dataclass
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from IPython.display import display

            ROOT = Path.cwd()
            for candidate in [ROOT, *ROOT.parents]:
                if (candidate / "data" / "sample_submission.csv").exists():
                    ROOT = candidate
                    break
            else:
                raise FileNotFoundError("No se encontro data/sample_submission.csv")

            CODE_DIR = ROOT / "proyecto_actual_v2" / "codigo"
            WORK_DIR = CODE_DIR / "work"
            EMBEDDED_SRC = WORK_DIR / "_embedded_pipeline_src"
            FIGURES_DIR = CODE_DIR / "figures"
            DATA_DIR = ROOT / "data"
            PYTHON_EXE = ROOT / ".venv" / "bin" / "python"
            if not PYTHON_EXE.exists():
                PYTHON_EXE = Path(sys.executable)

            WORK_DIR.mkdir(parents=True, exist_ok=True)
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)

            EXPECTED_FINAL_SHA256 = "b29288caa6e7b37b29e830a29655decc7c6bc8110ca3a40b828f4dd2f5fabdcc"
            FREESOUND2019_PRIVATE_LB = 0.68122
            COURSE_PUBLIC_LB = 0.64307
            PREVIOUS_3WAY_PRIVATE_LB = 0.67126
            """
        ),
        md(
            """
            ## Codigo fuente embebido

            `EMBEDDED_FILES` contiene todos los archivos Python necesarios para
            ejecutar el pipeline. Esta celda es larga a proposito: es la version
            que permite entregar o revisar el flujo sin abrir `pipeline_src/`.
            """
        ),
        code(f"EMBEDDED_FILES = {embedded_json}\n\nlen(EMBEDDED_FILES), sorted(EMBEDDED_FILES)\n"),
        md(
            """
            ## Materializacion de fuentes embebidas

            El notebook escribe esas fuentes a una carpeta de trabajo generada y
            despues importa la configuracion final desde ahi. Si se copia solo
            este notebook junto con los datos crudos, el codigo necesario viaja
            dentro del propio `.ipynb`.
            """
        ),
        code(
            """
            def materialize_embedded_source(*, force: bool = True) -> list[Path]:
                written: list[Path] = []
                for relative_path, source in EMBEDDED_FILES.items():
                    destination = EMBEDDED_SRC / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if force or not destination.exists() or destination.read_text(encoding="utf-8") != source:
                        destination.write_text(source, encoding="utf-8")
                    written.append(destination)
                return written


            materialized_files = materialize_embedded_source()
            if str(EMBEDDED_SRC) not in sys.path:
                sys.path.insert(0, str(EMBEDDED_SRC))

            from final_config import BRANCHES, FINE_TUNE

            {
                "embedded_files": len(EMBEDDED_FILES),
                "materialized_to": str(EMBEDDED_SRC.relative_to(ROOT)),
                "first_files": sorted(EMBEDDED_FILES)[:5],
            }
            """
        ),
        md("## Configuracion fija del modelo final"),
        code(
            """
            branch_config = pd.DataFrame(
                [
                    {
                        "branch": branch.name,
                        "weight": branch.ensemble_weight,
                        "architecture": branch.architecture,
                        "activation": branch.activation,
                        "frames": branch.frames,
                        "normalization": "global-mel" if branch.cache_tag == "globalmel" else "clip",
                        "base_epochs": branch.base_epochs,
                        "batch_size": branch.batch_size,
                        "optimizer": branch.optimizer,
                        "scheduler": branch.scheduler,
                        "lr_milestones": ",".join(map(str, branch.lr_milestones)),
                        "head_hidden": branch.head_hidden,
                        "head_dropout": branch.head_dropout,
                    }
                    for branch in BRANCHES
                ]
            )

            finetune_config = {
                "epochs_extra": FINE_TUNE.epochs,
                "learning_rate": FINE_TUNE.lr,
                "scheduler": "cosine",
                "min_lr": FINE_TUNE.min_lr,
                "batch_composition": "50% curated / 50% noisy",
                "curated_loss_weight": FINE_TUNE.curated_loss_weight,
                "noisy_loss_weight": FINE_TUNE.noisy_loss_weight,
                "gaussian_noise_std": FINE_TUNE.gaussian_noise_std,
                "augmentations": list(FINE_TUNE.augmentations),
                "ensemble_weights": [branch.ensemble_weight for branch in BRANCHES],
                "freesound2019_private_lb": FREESOUND2019_PRIVATE_LB,
                "course_public_lb": COURSE_PUBLIC_LB,
            }

            display(branch_config)
            finetune_config
            """
        ),
        md("## Datos crudos y validaciones iniciales"),
        code(
            """
            sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
            curated = pd.read_csv(DATA_DIR / "train_curated.csv")
            noisy = pd.read_csv(DATA_DIR / "train_noisy.csv")
            label_columns = [column for column in sample.columns if column != "fname"]

            raw_checks = {
                "sample_rows": len(sample),
                "curated_rows": len(curated),
                "noisy_rows": len(noisy),
                "num_labels": len(label_columns),
                "curated_unique_files": curated["fname"].nunique(),
                "noisy_unique_files": noisy["fname"].nunique(),
                "sample_has_fname": "fname" in sample.columns,
                "curated_has_labels": {"fname", "labels"}.issubset(curated.columns),
                "noisy_has_labels": {"fname", "labels"}.issubset(noisy.columns),
            }

            for split_name, frame in [("train_curated", curated), ("train_noisy", noisy)]:
                missing = [
                    fname
                    for fname in frame["fname"].astype(str).head(20)
                    if not (DATA_DIR / fname).exists()
                ]
                if missing:
                    raise FileNotFoundError(f"{split_name}: faltan archivos de muestra: {missing[:3]}")

            missing_test = [
                fname for fname in sample["fname"].astype(str).head(20) if not (DATA_DIR / fname).exists()
            ]
            if missing_test:
                raise FileNotFoundError(f"test: faltan archivos de muestra: {missing_test[:3]}")

            assert raw_checks["num_labels"] == 80
            assert raw_checks["sample_rows"] == 3361
            assert raw_checks["curated_has_labels"]
            assert raw_checks["noisy_has_labels"]
            raw_checks
            """
        ),
        md("## Funciones de ejecucion y validacion"),
        code(
            """
            def sha256(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                return digest.hexdigest()


            def run_command(command: list[str], *, enabled: bool) -> None:
                print("$ " + " ".join(map(str, command)))
                if not enabled:
                    print("skip: RUN_FULL_PIPELINE=False")
                    return
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                print(completed.stdout)
                if completed.returncode != 0:
                    raise RuntimeError(f"command failed with code {completed.returncode}")


            @dataclass
            class PipelineStep:
                name: str
                command: list[str]


            class FinalAudioTaggingPipelineLargo:
                def __init__(self, *, data_dir: Path, code_dir: Path, work_dir: Path, embedded_src: Path) -> None:
                    self.data_dir = data_dir
                    self.code_dir = code_dir
                    self.work_dir = work_dir
                    self.embedded_src = embedded_src
                    self.python = str(PYTHON_EXE)

                def cache_steps(self) -> list[PipelineStep]:
                    base = [self.python, str(self.embedded_src / "build_logmel_image_cache.py"), "--data-dir", str(self.data_dir)]
                    return [
                        PipelineStep("logmel_512_clip_curated_test", base + ["--splits", "curated,test", "--n-mels", "128", "--frames", "512", "--normalization", "clip"]),
                        PipelineStep("logmel_512_globalmel_curated_test", base + ["--splits", "curated,test", "--n-mels", "128", "--frames", "512", "--normalization", "global-mel", "--cache-tag", "globalmel"]),
                        PipelineStep("logmel_1024_clip_curated_test", base + ["--splits", "curated,test", "--n-mels", "128", "--frames", "1024", "--normalization", "clip"]),
                        PipelineStep("logmel_noisy_512", [self.python, str(self.embedded_src / "build_noisy_caches.py"), "--data-dir", str(self.data_dir)]),
                        PipelineStep("logmel_1024_memmap_all", [self.python, str(self.embedded_src / "build_f1024_memmap_caches.py"), "--data-dir", str(self.data_dir)]),
                    ]

                def base_train_steps(self) -> list[PipelineStep]:
                    steps: list[PipelineStep] = []
                    for branch in BRANCHES:
                        command = [
                            self.python,
                            str(self.embedded_src / "train_logmel_cnn.py"),
                            "--data-dir", str(self.data_dir),
                            "--models-dir", str(self.work_dir / "models" / branch.source_run),
                            "--submissions-dir", str(self.work_dir / "submissions" / branch.source_run),
                            "--experiments-dir", str(self.work_dir / "experiments" / branch.source_run),
                            "--n-mels", str(branch.n_mels),
                            "--frames", str(branch.frames),
                            "--epochs", str(branch.base_epochs),
                            "--batch-size", str(branch.batch_size),
                            "--lr", str(branch.base_lr),
                            "--weight-decay", str(branch.base_weight_decay),
                            "--optimizer", branch.optimizer,
                            "--initializer", branch.initializer,
                            "--architecture", branch.architecture,
                            "--activation", branch.activation,
                            "--head-hidden", str(branch.head_hidden),
                            "--head-dropout", str(branch.head_dropout),
                            "--block-dropout", str(branch.block_dropout),
                            "--time-reverse-probability", str(branch.time_reverse_probability),
                            "--contrast-strength", str(branch.contrast_strength),
                            "--scheduler", branch.scheduler,
                            "--lr-milestones", ",".join(map(str, branch.lr_milestones)),
                            "--min-lr", str(FINE_TUNE.min_lr),
                            "--seed", str(FINE_TUNE.seed),
                            "--full-train",
                        ]
                        if branch.cache_tag is not None:
                            command.extend(["--cache-tag", branch.cache_tag])
                        steps.append(PipelineStep(f"train_curated_{branch.name}", command))
                    return steps

                def finetune_steps(self) -> list[PipelineStep]:
                    return [
                        PipelineStep(
                            "finetune_curated_noisy",
                            [
                                self.python,
                                str(self.embedded_src / "finetune_mixed_noisy.py"),
                                "--data-dir", str(self.data_dir),
                                "--work-dir", str(self.work_dir),
                                "--branches", "all",
                                "--epochs", str(FINE_TUNE.epochs),
                                "--lr", str(FINE_TUNE.lr),
                                "--min-lr", str(FINE_TUNE.min_lr),
                                "--noisy-loss-weight", str(FINE_TUNE.noisy_loss_weight),
                                "--curated-loss-weight", str(FINE_TUNE.curated_loss_weight),
                                "--gaussian-noise-std", str(FINE_TUNE.gaussian_noise_std),
                                "--seed", str(FINE_TUNE.seed),
                                "--num-workers", str(FINE_TUNE.num_workers),
                                "--submission-checkpoint", FINE_TUNE.submission_checkpoint,
                            ],
                        ),
                        PipelineStep(
                            "blend_final_submission",
                            [
                                self.python,
                                str(self.embedded_src / "evaluate_and_blend.py"),
                                "--data-dir", str(self.data_dir),
                                "--work-dir", str(self.work_dir),
                                "--output-dir", str(self.code_dir),
                            ],
                        ),
                    ]

                def steps(self) -> list[PipelineStep]:
                    return self.cache_steps() + self.base_train_steps() + self.finetune_steps()

                def run(self, *, enabled: bool) -> pd.DataFrame:
                    rows = []
                    for step in self.steps():
                        run_command(step.command, enabled=enabled)
                        rows.append({"step": step.name, "command": " ".join(map(str, step.command))})
                    return pd.DataFrame(rows)

                def validate_submission(self, path: Path) -> dict[str, object]:
                    if not path.exists():
                        raise FileNotFoundError(f"No existe {path}. Ejecutar el pipeline completo para generarlo.")
                    submission = pd.read_csv(path)
                    assert submission.shape == (len(sample), len(sample.columns))
                    assert list(submission.columns) == list(sample.columns)
                    assert submission["fname"].astype(str).equals(sample["fname"].astype(str))
                    values = submission[label_columns].to_numpy(dtype=float)
                    assert np.isfinite(values).all()
                    assert values.min() >= 0.0
                    assert values.max() <= 1.0
                    return {
                        "path": str(path.relative_to(ROOT)),
                        "rows": len(submission),
                        "labels": len(label_columns),
                        "min_probability": float(values.min()),
                        "max_probability": float(values.max()),
                        "sha256": sha256(path),
                    }


            pipeline = FinalAudioTaggingPipelineLargo(
                data_dir=DATA_DIR,
                code_dir=CODE_DIR,
                work_dir=WORK_DIR,
                embedded_src=EMBEDDED_SRC,
            )
            planned_steps = pd.DataFrame(
                [{"step": step.name, "command": " ".join(map(str, step.command))} for step in pipeline.steps()]
            )
            display(planned_steps)
            """
        ),
        md("## Ejecucion del pipeline"),
        code(
            """
            RUN_FULL_PIPELINE = False

            executed_steps = pipeline.run(enabled=RUN_FULL_PIPELINE)
            submission_checks = pipeline.validate_submission(CODE_DIR / "submission.csv")
            assert submission_checks["sha256"] == EXPECTED_FINAL_SHA256

            display(executed_steps)
            submission_checks
            """
        ),
        md("## Figuras y metadata final"),
        code(
            """
            plt.style.use("default")

            split_counts = pd.DataFrame(
                [
                    {"split": "train_curated", "rows": len(curated)},
                    {"split": "train_noisy", "rows": len(noisy)},
                    {"split": "test", "rows": len(sample)},
                ]
            )
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.bar(split_counts["split"], split_counts["rows"], color=["#2f6f73", "#d28b35", "#575a89"])
            ax.set_title("Tamanos de splits")
            ax.set_ylabel("audios")
            ax.grid(axis="y", alpha=0.25)
            for index, value in enumerate(split_counts["rows"]):
                ax.text(index, value + 300, f"{value}", ha="center", fontsize=9)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig_v2_largo_split_sizes.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.2, 3.5))
            ax.pie(
                branch_config["weight"],
                labels=branch_config["branch"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#4b7f52", "#386c9f", "#ba7a2a"],
            )
            ax.set_title("Pesos del ensamble final")
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "fig_v2_largo_component_weights.png", dpi=160)
            plt.close(fig)

            figures = sorted(path.name for path in FIGURES_DIR.glob("fig_v2_largo_*.png"))

            metadata = {
                "pipeline": {
                    "self_contained_from_raw_data": True,
                    "contains_embedded_source": True,
                    "uses_pipeline_src_as_runtime_dependency": False,
                    "uses_investigation_artifacts_as_source": False,
                    "run_full_pipeline_flag": RUN_FULL_PIPELINE,
                    "embedded_source_dir": str(EMBEDDED_SRC.relative_to(ROOT)),
                    "embedded_file_count": len(EMBEDDED_FILES),
                },
                "raw_data_checks": raw_checks,
                "branch_config": branch_config.to_dict(orient="records"),
                "finetune_config": finetune_config,
                "submission": submission_checks,
                "leaderboard": {
                    "previous_3way_private_lb": PREVIOUS_3WAY_PRIVATE_LB,
                    "freesound2019_private_lb": FREESOUND2019_PRIVATE_LB,
                    "course_public_lb": COURSE_PUBLIC_LB,
                },
                "planned_steps": planned_steps.to_dict(orient="records"),
                "figures": figures,
            }
            metadata_path = CODE_DIR / "pipeline_final_taller_2_v2_largo_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\\n")
            metadata
            """
        ),
    ]
    return nb


def main() -> None:
    notebook = build_notebook()
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}")
    long_notebook = build_long_notebook()
    nbf.write(long_notebook, LONG_NOTEBOOK_PATH)
    print(f"wrote {LONG_NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
