# Noisy Fine-Tune Curated Init Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated experiment that resumes the three final curated-only E100 branches from the 0.67126 ensemble, fine-tunes them with curated + noisy and controlled augmentation for 30 epochs, then rebuilds and evaluates the same weighted ensemble.

**Architecture:** Keep the delivery pipeline untouched. Add experiment-local scripts under `investigation/noisy_finetune_20260704/` that reuse the existing model, metrics, submission, and log-mel cache helpers from `investigation/scripts`. Outputs remain under the new experiment folder and generated CSV/JSON/checkpoint files stay ignored by `.gitignore`.

**Tech Stack:** Python, PyTorch, NumPy, pandas, existing FAT2019 helpers, Kaggle CLI if credentials are available.

---

### Task 1: Experiment Skeleton and Artifact Inventory

**Files:**
- Create: `investigation/noisy_finetune_20260704/README.md`
- Create: `investigation/noisy_finetune_20260704/configs.md`
- Create: `investigation/noisy_finetune_20260704/scripts/prepare_experiment.py`

- [ ] **Step 1: Define the three source branches**

Use these exact source checkpoints and metadata:

```text
separable_headsep:
  checkpoint: investigation/models/parallel100_20260702_separable_headsep_e100_seed42/small_logmel_cnn_best.pt
  metadata: investigation/experiments/parallel100_20260702_separable_headsep_e100_seed42/small_logmel_cnn_metadata.json
  weight: 0.25

globalmel_sep_temporal:
  checkpoint: investigation/models/parallel100_20260702_globalmel_sep_temporal_e100_seed42/small_logmel_cnn_best.pt
  metadata: investigation/experiments/parallel100_20260702_globalmel_sep_temporal_e100_seed42/small_logmel_cnn_metadata.json
  weight: 0.375

sep_temporal_f1024:
  checkpoint: investigation/models/parallel100_20260702_sep_temporal_f1024_e100_seed42/small_logmel_cnn_best.pt
  metadata: investigation/experiments/parallel100_20260702_sep_temporal_f1024_e100_seed42/small_logmel_cnn_metadata.json
  weight: 0.375
```

- [ ] **Step 2: Write `prepare_experiment.py`**

The script must validate every source checkpoint/metadata/submission exists, copy source checkpoints and metadata into `investigation/noisy_finetune_20260704/source_states/`, and write `source_states/manifest.md` with hashes and branch configs.

- [ ] **Step 3: Run artifact preparation**

Run:

```bash
.venv/bin/python investigation/noisy_finetune_20260704/scripts/prepare_experiment.py
```

Expected: prints `source_state_manifest_ok` and writes the manifest.

### Task 2: Noisy Log-Mel Caches

**Files:**
- Create: `investigation/noisy_finetune_20260704/scripts/build_noisy_caches.py`

- [ ] **Step 1: Support the noisy split**

The script must build log-mel image caches for `train_noisy.csv` using the same representation settings as the three source branches:

```text
noisy_logmel_image_m128_f512.npz
noisy_logmel_image_m128_f512_globalmel.npz
noisy_logmel_image_m128_f1024.npz
```

For `global-mel`, the mean/std must be computed on curated training audio, matching the existing cache builder's normalization policy.

- [ ] **Step 2: Run cache build only for missing files**

Run:

```bash
.venv/bin/python investigation/noisy_finetune_20260704/scripts/build_noisy_caches.py
```

Expected: existing caches are skipped, missing noisy caches are written with row count matching `train_noisy.csv` after known bad files are dropped.

### Task 3: Mixed Fine-Tuning Trainer

**Files:**
- Create: `investigation/noisy_finetune_20260704/scripts/finetune_mixed_noisy.py`

- [ ] **Step 1: Reuse existing model construction**

Import `SmallLogmelCnn`, `LogmelDataset`, `build_optimizer`, `build_scheduler`, `_predict_logits`, and FAT2019 helpers from existing scripts. Instantiate each branch from metadata and load the copied checkpoint state before any training.

- [ ] **Step 2: Implement curated + noisy training**

Training must use:

```text
epochs: 30
curated/noisy batch ratio: 50/50
noisy loss weight: 0.30
curated loss weight: 1.00
lr: 0.0001
scheduler: cosine
min_lr: 0.000001
augmentations: time shift, time mask, frequency mask, gaussian noise
```

Use per-sample loss reduction so noisy rows receive the lower weight. Compute `pos_weight` from curated training rows only.

- [ ] **Step 3: Preserve a local curated validation split**

For local evaluation, hold out a deterministic 20% curated split with seed 42. Train fine-tuning on the remaining curated rows plus noisy rows. Report baseline validation lwlrap before fine-tuning, best validation lwlrap during fine-tuning, final validation lwlrap, and delta.

- [ ] **Step 4: Save branch outputs**

For each branch, save:

```text
runs/<branch>/models/small_logmel_cnn_best.pt
runs/<branch>/models/small_logmel_cnn_final.pt
runs/<branch>/submissions/small_logmel_cnn.csv
runs/<branch>/experiments/history.csv
runs/<branch>/experiments/metadata.md
```

### Task 4: Ensemble and Evaluation

**Files:**
- Create: `investigation/noisy_finetune_20260704/scripts/run_experiment.sh`
- Create: `investigation/noisy_finetune_20260704/scripts/evaluate_and_blend.py`

- [ ] **Step 1: Orchestrate the run**

`run_experiment.sh` must prepare source states, build noisy caches, run all three fine-tuning branches, and blend with weights `0.25, 0.375, 0.375`.

- [ ] **Step 2: Validate final submission**

The final submission must have the same columns and row count as `data/sample_submission.csv`, no NaNs, and all probabilities clipped to `[0, 1]`.

- [ ] **Step 3: Document results**

Write `results.md` with local branch metrics, ensemble local metric, paths to component submissions, final submission hash, and Kaggle status.

### Task 5: Verification

**Files:**
- Create: `investigation/noisy_finetune_20260704/scripts/validate_experiment.py`

- [ ] **Step 1: Static validation**

Run:

```bash
.venv/bin/python investigation/noisy_finetune_20260704/scripts/validate_experiment.py --static
```

Expected: validates source paths, configs, script imports, and planned output directories.

- [ ] **Step 2: Final validation**

Run:

```bash
.venv/bin/python investigation/noisy_finetune_20260704/scripts/validate_experiment.py
```

Expected: validates caches, branch checkpoints, branch submissions, blend submission, metrics files, and `results.md`.

