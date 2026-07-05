# Notebook Largo V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `*_largo` notebook for `proyecto_actual_v2` that contains all final audio pipeline code inside the notebook itself.

**Architecture:** Keep the current modular notebook as the clean delivery version. Add a parallel long notebook whose cells embed the full helper source code and materialize it only when the heavy pipeline is explicitly enabled. Validate that the long notebook is executed in light mode, does not import `pipeline_src`, and still verifies the final submission.

**Tech Stack:** Python, Jupyter/nbformat, pandas, matplotlib, local `.venv` for heavy PyTorch/Torchaudio execution.

---

### Task 1: Generate Long Notebook

**Files:**
- Modify: `proyecto_actual_v2/codigo/build_v2_notebook.py`
- Create: `proyecto_actual_v2/codigo/pipeline_final_taller_2_v2_largo.ipynb`

- [x] Add a second notebook builder that embeds every `.py` file from `proyecto_actual_v2/codigo/pipeline_src`.
- [x] The long notebook must not import `pipeline_src`; it must use `EMBEDDED_FILES` as the source of truth.
- [x] Include `RUN_FULL_PIPELINE = False` for executed delivery mode.
- [x] Include a materialization function that writes embedded files to `proyecto_actual_v2/codigo/work/_embedded_pipeline_src` when full execution is enabled.
- [x] Execute the long notebook in light mode and save outputs.

### Task 2: Validation And Documentation

**Files:**
- Modify: `proyecto_actual_v2/validar_entrega.py`
- Modify: `proyecto_actual_v2/codigo/README.md`
- Modify: `proyecto_actual_v2/README.md`

- [x] Validate the long notebook is executed with no error outputs.
- [x] Validate the long notebook metadata marks `contains_embedded_source=true`.
- [x] Validate the long notebook text has no `from pipeline_src` import and no `SOURCE_SUBMISSION`.
- [x] Document the difference between the modular notebook and the long notebook.
- [x] Run `proyecto_actual_v2/validar_entrega.py` and expect `proyecto_actual_v2_validation_ok`.
