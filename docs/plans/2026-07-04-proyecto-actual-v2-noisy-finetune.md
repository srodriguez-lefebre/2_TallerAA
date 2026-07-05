# Proyecto Actual V2 Noisy Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `proyecto_actual_v2/` as the new delivery package centered on the noisy fine-tuned 3-way model, replacing the previous 0.67126 final narrative with the 0.68122 Kaggle evidence.

**Architecture:** Keep `proyecto_actual/` intact and build a parallel v2 folder with `codigo/`, `informe/`, `presentacion/`, and a root validator. The v2 code notebook must be self-contained from raw data: validate raw `.wav`/CSV inputs, build log-mel caches, train the three curated e100 branches, continue with curated+noisy fine-tuning, blend the three outputs, and validate the final CSV. Heavy training is controlled by `RUN_FULL_PIPELINE`; the notebook is executed in light mode for delivery, but the code path no longer depends on `investigation/noisy_finetune_20260704/` as the source of truth.

**Tech Stack:** Python, Jupyter/nbformat, pandas, matplotlib, LaTeX, static HTML/CSS/JS.

---

### Task 1: Structure and Evidence

**Files:**
- Create: `proyecto_actual_v2/README.md`
- Create: `proyecto_actual_v2/codigo/README.md`
- Create: `proyecto_actual_v2/informe/README.md`
- Create: `proyecto_actual_v2/presentacion/README.md`

- [ ] Create the folder tree matching `proyecto_actual/`.
- [ ] Keep the scored noisy fine-tune final submission in `proyecto_actual_v2/codigo/submission.csv`.
- [ ] Record the expected hash `b29288caa6e7b37b29e830a29655decc7c6bc8110ca3a40b828f4dd2f5fabdcc`.
- [ ] Keep heavyweight checkpoints and generated caches under the notebook-controlled `codigo/work/` path when the full pipeline is run.

### Task 2: Code Notebook

**Files:**
- Create: `proyecto_actual_v2/codigo/pipeline_final_taller_2_v2.ipynb`
- Create: `proyecto_actual_v2/codigo/pipeline_final_taller_2_v2_metadata.json`
- Create: `proyecto_actual_v2/codigo/build_v2_notebook.py`
- Create: `proyecto_actual_v2/codigo/pipeline_src/`

- [ ] Build an executed notebook with sections: libraries, fixed configuration, raw data validation, log-mel preprocessing, model pipeline class, final execution, figures, and revision.
- [ ] Use no heavy training by default. `RUN_FULL_PIPELINE=True` runs the full path from raw data to `submission.csv`.
- [ ] Include local helpers under `pipeline_src/` so the notebook does not call `investigation/` scripts.
- [ ] Generate and save lightweight figures under `proyecto_actual_v2/codigo/figures/`.
- [ ] Validate the final CSV shape, columns, fname order, probabilities, and hash.

### Task 3: Report

**Files:**
- Create: `proyecto_actual_v2/informe/informe.tex`
- Create: `proyecto_actual_v2/informe/informe.pdf`
- Create: figures under `proyecto_actual_v2/informe/img/`

- [ ] Rewrite the report around the final noisy fine-tune model.
- [ ] Include the earlier 0.67126 model as the previous milestone, not the final point.
- [ ] Include the new public Kaggle 2019 private score `0.68122` and course Kaggle public score `0.64307`, clearly separated.
- [ ] Explain why local validation is a degradation-control signal because the source E100 checkpoints were full-train.
- [ ] Keep the report within 8 pages.

### Task 4: Presentation

**Files:**
- Create: `proyecto_actual_v2/presentacion/index.html`
- Create: assets under `proyecto_actual_v2/presentacion/assets/`

- [ ] Build an HTML presentation with at least 10 slides.
- [ ] Center the story on the final noisy fine-tune system: curated pretraining, controlled noisy continuation, 3-way blend, Kaggle evidence.
- [ ] Reuse/generated figures must load from local assets.
- [ ] Include keyboard navigation and no missing images.

### Task 5: Validation

**Files:**
- Create: `proyecto_actual_v2/validar_entrega.py`

- [ ] Validate notebook executed with no error outputs.
- [ ] Validate metadata marks `self_contained_from_raw_data=true`.
- [ ] Validate `pipeline_src/` exists and has no `investigation`/`scripts.` external references.
- [ ] Validate submission hash, rows, columns, fname order, NaNs and probability range.
- [ ] Validate PDF exists and has at most 8 pages.
- [ ] Validate presentation has at least 10 slides, required score strings, and no missing assets.
- [ ] Run the validator and document `proyecto_actual_v2_validation_ok`.
