# Simple Defensible Audio System Implementation Plan

> **For Codex:** Execute this plan task-by-task. If an `executing-plans` skill is installed, use it; otherwise follow the steps manually.

**Goal:** Recover a Kaggle private LB around `0.66` with a defensible 3-4 component audio system, avoiding the hidden 10-component expanded ensemble.

**Architecture:** Treat a component as an independently trained prediction source/model recipe, not as a CSV alias. The primary track removes the nested `current` ensemble and uses only disclosed atomic branches. The fallback track keeps a macro-level 3-way system using `current` only if the strict track cannot reach `0.66`.

**Tech Stack:** Python, PyTorch, NumPy/Pandas, log-mel image caches, Kaggle CLI copy-kernel submission flow.

**Files to Understand:**

- `investigation/results/theory_supported_experiments_2026_06_29.md` - latest Kaggle scores and model decisions.
- `investigation/results/geron_ensemble_search.md` - exact definition of `current` and historical blend evidence.
- `investigation/scripts/blend_submissions.py` - creates weighted submission CSVs.
- `investigation/scripts/kaggle_code_submission.py` - submits CSVs to Kaggle for this code competition.
- `investigation/scripts/train_logmel_cnn.py` - trains CNN/BiGRU log-mel branches.
- `investigation/scripts/build_logmel_image_cache.py` - builds clip/global-mel log-mel image caches.

---

## Planning Notes From Subagents

- Subagent 1 recommendation: the already verified macro shrink is
  `0.55 * current + 0.25 * globalmel + 0.20 * f1024`, private LB `0.66996`.
  This is useful as fallback, but it still hides the old 7-model `current`.
- Subagent 2 recommendation: start from the stricter constraint: no hidden
  `current`. First submit simple soft-voting controls using existing atomic
  branches, then train one new branch combining the two strongest ideas:
  global mel-band normalization and `frames=1024` temporal context.

## Success Criteria

- Primary success: a current-free system with 3-4 disclosed components and
  Kaggle private LB `>= 0.66000`.
- Preferred success: same constraint with private LB `>= 0.66500`.
- Fallback success: a 3-component macro system with `current` disclosed as a
  previous ensemble, private LB `>= 0.66500`.
- Reject any candidate that requires TTA, MC Dropout, stacking, row-z/rank
  transforms, or a nested ensemble that cannot be explained in one slide.

## Component Accounting Rule

Count one component per independently trained prediction source:

- `globalmel_sep_temporal`: one component.
- `sep_temporal_f1024`: one component.
- `sepres_se_head256`: one component.
- `separable_headsep`: one component if using its own full-train CSV.
- `current`: not one strict component; it expands to seven old components.

Use `current` only in fallback reporting, never in the strict final candidate.

## Theory Justification

- Soft voting / simple averaging: Geron ensemble learning, chapter 7.
- Log-mel standardization: course preprocessing/standardization.
- CNN over log-mel images: course CNN material and Taller 6/7.
- Data augmentation and regularization: Taller 7 and neural network
  regularization material.
- BiGRU temporal head: Taller 8/9 and course RNN material; audio is a temporal
  sequence over log-mel frames.
- Transfer/noisy pretraining, if reached: Taller 7 transfer-learning framing.

---

### Task 1: Freeze The Baselines And Naming

**Files:**

- Read: `investigation/results/theory_supported_experiments_2026_06_29.md`
- Read: `investigation/results/geron_ensemble_search.md`
- Create: `investigation/results/simple_defensible_blends_2026_07_01.md`
- Modify: `investigation/results/experiment_log.csv`

**Step 1: Record the baseline table**

Write a short run note with these fixed references:

- `current` original: private LB `0.65289`.
- best giant expanded ensemble: private LB `0.67025`.
- macro 3-way fallback: private LB `0.66996`.
- strict target: `>= 0.66000` without hidden `current`.

**Step 2: Define artifact names**

Use these existing CSVs:

- `globalmel`: `investigation/submissions/theory_globalmel_sep_temporal_full_e40_seed42/small_logmel_cnn.csv`
- `f1024`: `investigation/submissions/theory_sep_temporal_f1024_full_e40_seed42/small_logmel_cnn.csv`
- `se`: `investigation/submissions/theory_sepres_se_head256_full_e34_seed42/small_logmel_cnn.csv`
- `separable_headsep`: `investigation/submissions/logmel_cnn_catsdogs_sepres_head256_full_e56_seed42/small_logmel_cnn.csv`
- `sepres`: `investigation/submissions/logmel_cnn_catsdogs_sepres_full_e37_seed42/small_logmel_cnn.csv`
- `head256`: `investigation/submissions/logmel_cnn_fashion_head256_full_e54_seed42/small_logmel_cnn.csv`
- `resnet50`: `investigation/submissions/imagenet_transfer_catsdogs_resnet50_full_e19_seed42/resnet50_transfer.csv`
- `current`: `investigation/submissions/catsdogs_headsep_final/translated_local.csv`

**Step 3: Validate CSV compatibility**

Run:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

paths = [
    Path("investigation/submissions/theory_globalmel_sep_temporal_full_e40_seed42/small_logmel_cnn.csv"),
    Path("investigation/submissions/theory_sep_temporal_f1024_full_e40_seed42/small_logmel_cnn.csv"),
    Path("investigation/submissions/logmel_cnn_catsdogs_sepres_head256_full_e56_seed42/small_logmel_cnn.csv"),
]
frames = [pd.read_csv(path) for path in paths]
print([path.exists() for path in paths])
print(all(list(df.columns) == list(frames[0].columns) for df in frames))
print(all(df["fname"].equals(frames[0]["fname"]) for df in frames))
PY
```

Expected: all paths exist, columns match, filenames match.

---

### Task 2: Submit Strict Current-Free Controls

**Files:**

- Use: `investigation/scripts/blend_submissions.py`
- Use: `investigation/scripts/kaggle_code_submission.py`
- Create: `investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv`
- Create: `investigation/results/submissions/simple_sepres_headsep_globalmel_f1024_equal.csv`

**Step 1: Build strict 3-component equal blend**

Run:

```bash
PYTHONPATH=investigation .venv/bin/python investigation/scripts/blend_submissions.py \
  --input investigation/submissions/logmel_cnn_catsdogs_sepres_head256_full_e56_seed42/small_logmel_cnn.csv \
  --weight 1 \
  --input investigation/submissions/theory_globalmel_sep_temporal_full_e40_seed42/small_logmel_cnn.csv \
  --weight 1 \
  --input investigation/submissions/theory_sep_temporal_f1024_full_e40_seed42/small_logmel_cnn.csv \
  --weight 1 \
  --output investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv
```

Rationale: simple Geron soft voting over three disclosed, diverse branches.

**Step 2: Build strict 4-component equal blend**

Run:

```bash
PYTHONPATH=investigation .venv/bin/python investigation/scripts/blend_submissions.py \
  --input investigation/submissions/logmel_cnn_catsdogs_sepres_full_e37_seed42/small_logmel_cnn.csv \
  --weight 1 \
  --input investigation/submissions/logmel_cnn_catsdogs_sepres_head256_full_e56_seed42/small_logmel_cnn.csv \
  --weight 1 \
  --input investigation/submissions/theory_globalmel_sep_temporal_full_e40_seed42/small_logmel_cnn.csv \
  --weight 1 \
  --input investigation/submissions/theory_sep_temporal_f1024_full_e40_seed42/small_logmel_cnn.csv \
  --weight 1 \
  --output investigation/results/submissions/simple_sepres_headsep_globalmel_f1024_equal.csv
```

Rationale: still simple, adds one disclosed separable-residual branch for
diversity.

**Step 3: Submit both to Kaggle**

Run one submission at a time:

```bash
KAGGLE_CONFIG_DIR=keys PYTHONPATH=investigation .venv/bin/python investigation/scripts/kaggle_code_submission.py \
  --csv investigation/results/submissions/simple_headsep_globalmel_f1024_equal.csv \
  --slug fat2019-simple-headsep-globalmel-f1024-eq \
  --title "FAT2019 simple headsep globalmel f1024 equal" \
  --message "simple current-free headsep globalmel f1024 equal"
```

Then repeat for:

```text
csv: investigation/results/submissions/simple_sepres_headsep_globalmel_f1024_equal.csv
slug: fat2019-simple-sepres-headsep-globalmel-f1024-eq
title: FAT2019 simple sepres headsep globalmel f1024 equal
message: simple current-free sepres headsep globalmel f1024 equal
```

**Step 4: Decision gate**

- If either candidate reaches private LB `>= 0.66000`, keep it as the first
  defensible result.
- If it reaches `>= 0.66500`, make it the presentation candidate unless a later
  strict candidate beats it.
- If both are below `0.66000`, continue; do not rescue with hidden `current`
  yet.

---

### Task 3: Submit Rounded Strict Weighted Controls

**Files:**

- Create: `investigation/results/submissions/simple_headsep50_globalmel25_f1024_25.csv`
- Create: `investigation/results/submissions/simple_head25635_headsep20_globalmel25_f1024_20.csv`

**Step 1: Build a 3-component weighted blend**

Use:

```text
0.50 * separable_headsep
0.25 * globalmel
0.25 * f1024
```

This tests whether the old strongest atomic neural branch should anchor the
strict system.

**Step 2: Build a 4-component weighted blend**

Use:

```text
0.35 * head256
0.20 * separable_headsep
0.25 * globalmel
0.20 * f1024
```

This replaces the hidden `current` with two disclosed older branches plus two
new temporal branches.

**Step 3: Submit only if Task 2 is below `0.66500`**

Do not spend Kaggle submits if equal-weight already clears preferred success.

---

### Task 4: Train The Highest-Value New Branch

**Files:**

- Use: `investigation/scripts/build_logmel_image_cache.py`
- Use: `investigation/scripts/train_logmel_cnn.py`
- Create: `data/curated_logmel_image_m128_f1024_globalmel.npz`
- Create: `data/test_logmel_image_m128_f1024_globalmel.npz`
- Create: `investigation/models/simple_globalmel_f1024_sep_temporal`
- Create: `investigation/experiments/simple_globalmel_f1024_sep_temporal`
- Create: `investigation/submissions/simple_globalmel_f1024_sep_temporal`

**Step 1: Build global-mel 1024-frame cache**

Run:

```bash
PYTHONPATH=investigation .venv/bin/python investigation/scripts/build_logmel_image_cache.py \
  --data-dir data \
  --n-mels 128 \
  --frames 1024 \
  --normalization global-mel \
  --cache-tag globalmel
```

Expected output includes `curated_logmel_image_m128_f1024_globalmel.npz` and
`test_logmel_image_m128_f1024_globalmel.npz`.

**Step 2: Train holdout model**

Run:

```bash
PYTHONPATH=investigation .venv/bin/python investigation/scripts/train_logmel_cnn.py \
  --data-dir data \
  --models-dir investigation/models/simple_globalmel_f1024_sep_temporal \
  --submissions-dir investigation/submissions/simple_globalmel_f1024_sep_temporal \
  --experiments-dir investigation/experiments/simple_globalmel_f1024_sep_temporal \
  --n-mels 128 \
  --frames 1024 \
  --cache-tag globalmel \
  --architecture separable_temporal_bigru \
  --activation silu \
  --head-dropout 0.3 \
  --optimizer adamw \
  --lr 0.0005 \
  --weight-decay 0.0001 \
  --scheduler plateau \
  --plateau-patience 2 \
  --plateau-factor 0.5 \
  --epochs 45 \
  --batch-size 12 \
  --seed 42
```

Promote if it beats either:

- `globalmel` individual holdout `0.813036`
- `f1024` individual holdout `0.801006`

or improves a strict blend by at least `0.0015` local.

**Step 3: Full-train promoted model**

If promoted, full-train for the best holdout epoch count:

```bash
PYTHONPATH=investigation .venv/bin/python investigation/scripts/train_logmel_cnn.py \
  --data-dir data \
  --models-dir investigation/models/simple_globalmel_f1024_sep_temporal_full_e<BEST>_seed42 \
  --submissions-dir investigation/submissions/simple_globalmel_f1024_sep_temporal_full_e<BEST>_seed42 \
  --experiments-dir investigation/experiments/simple_globalmel_f1024_sep_temporal_full_e<BEST>_seed42 \
  --n-mels 128 \
  --frames 1024 \
  --cache-tag globalmel \
  --architecture separable_temporal_bigru \
  --activation silu \
  --head-dropout 0.3 \
  --optimizer adamw \
  --lr 0.0005 \
  --weight-decay 0.0001 \
  --scheduler multistep \
  --epochs <BEST> \
  --batch-size 12 \
  --seed 42 \
  --full-train
```

**Step 4: Blend and submit**

Try simple strict candidates:

```text
0.40 * separable_headsep
0.30 * simple_globalmel_f1024
0.30 * best_existing_temporal_or_globalmel
```

and:

```text
1/3 * separable_headsep
1/3 * simple_globalmel_f1024
1/3 * f1024_or_globalmel
```

Submit only the best local candidate first.

---

### Task 5: Add MixUp Only If Needed

**Files:**

- Modify: `investigation/scripts/train_logmel_cnn.py`
- Test: `investigation/tests/test_train_logmel_cnn.py`
- Create: `investigation/models/simple_mixup_globalmel_f1024`
- Create: `investigation/experiments/simple_mixup_globalmel_f1024`

**Step 1: Add MixUp flags**

Add CLI args:

```text
--mixup-alpha
--mixup-probability
```

Apply MixUp only in `_train_epoch`, after moving tensors to device:

```text
mixed_x = lam * x + (1 - lam) * x[perm]
mixed_y = lam * y + (1 - lam) * y[perm]
```

Use multilabel soft targets with `BCEWithLogitsLoss`.

**Step 2: Test the helper**

Add a unit test that verifies:

- shapes are unchanged;
- labels become convex combinations;
- `alpha=0` disables MixUp.

**Step 3: Train one holdout run**

Try `mixup_alpha=0.2`, `mixup_probability=0.5` on the best temporal branch.

Promote only if it beats the non-MixUp same branch by `>= 0.003` local.

---

### Task 6: Consider Staged Noisy Pretraining Only After Strict Submits

**Files:**

- Modify: `investigation/scripts/build_logmel_image_cache.py`
- Modify: `investigation/scripts/train_logmel_cnn.py`
- Test: `investigation/tests/test_train_logmel_cnn.py`

**Step 1: Do not start here**

This is higher implementation risk. Only start if:

- strict existing-branch submissions are below `0.66000`; and
- globalmel+f1024 does not produce a usable strict system.

**Step 2: Implement as staged transfer, not noisy concat**

Pretrain on noisy labels, then fine-tune curated. Present as transfer learning
from a larger, noisier acoustic corpus to curated labels.

Promotion gate: same architecture improves curated holdout by `>= 0.005`.

---

### Task 7: Record Every Result

**Files:**

- Modify: `investigation/results/simple_defensible_blends_2026_07_01.md`
- Modify: `investigation/results/experiment_log.csv`
- Modify: `investigation/results/README.md`

For every submitted candidate, record:

- component count under strict rule;
- exact weights;
- whether `current` is used;
- local holdout/blend score if available;
- Kaggle public/private score;
- decision: `keep`, `discard`, `fallback`, or `needs-rerun`.

---

## Execution Order

1. Task 1.
2. Task 2.
3. Stop and inspect Kaggle scores.
4. If strict score `>= 0.66500`, polish docs and stop.
5. If strict score is `0.66000-0.66500`, run Task 3 or Task 4 depending on time.
6. If strict score `< 0.66000`, run Task 4.
7. Run Task 5 only if Task 4 is close but not enough.
8. Run Task 6 only as last resort.

## Current Recommendation

Start by submitting the strict 3-way and strict 4-way equal blends. They require
no retraining, directly test the presentation constraint, and establish whether
we can hit `0.66` without the hidden `current` ensemble. In parallel or after
those submissions, train `globalmel + f1024` as one stronger temporal branch.
