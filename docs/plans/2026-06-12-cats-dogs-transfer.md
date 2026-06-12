# Cats and Dogs Transfer Implementation Plan

> **For Codex:** Execute this plan task-by-task with TDD and fresh verification.

**Goal:** Improve the Freesound Audio Tagging 2019 system using techniques learned exclusively from `7TAA_gatos_y_perros.ipynb`.

**Architecture:** Keep the existing normalized log-mel cache and multilabel training loop. Add a separable-residual CNN and audio-safe augmentation controls to the current trainer, then add a separate frozen ImageNet transfer trainer so the pretrained workflow remains isolated. Compare every candidate on the existing seed-42 holdout before full training and Kaggle submission.

**Tech Stack:** Python, PyTorch 2.6, torchvision 0.21, NumPy, pandas, unittest, CUDA.

**Files to Understand:**

- `scripts/train_logmel_cnn.py` - current CNN, augmentation and training loop.
- `scripts/predict_logmel_cnn.py` - checkpoint reconstruction.
- `scripts/fat2019/neural_helpers.py` - split and multilabel utilities.
- `tests/test_train_logmel_cnn.py` - current neural unit tests.
- `experiments/resultados.md` - experiment registry.
- `research/fashion_mnist_transfer.md` - previous transfer-study format.

---

### Task 1: Separable-Residual Architecture

**Files:**

- Modify: `tests/test_train_logmel_cnn.py`
- Modify: `scripts/train_logmel_cnn.py`
- Modify: `scripts/predict_logmel_cnn.py`

**Steps:**

1. Add a failing test that builds `SmallLogmelCnn(..., architecture="separable_residual")`.
2. Assert that the model contains depthwise convolutions, residual projections and returns `(batch, classes)`.
3. Run the targeted test and confirm it fails because the option does not exist.
4. Implement the minimal architecture and CLI/checkpoint support.
5. Run targeted and existing CNN tests.

### Task 2: Audio-Safe Vision Augmentations

**Files:**

- Modify: `tests/test_train_logmel_cnn.py`
- Modify: `scripts/train_logmel_cnn.py`

**Steps:**

1. Add failing deterministic tests for temporal reversal and contrast scaling.
2. Confirm reversal affects only the time axis and contrast preserves shape/finite values.
3. Implement `--time-reverse-probability` and `--contrast-strength`.
4. Keep both disabled by default for backward compatibility.
5. Run targeted tests and the complete CNN test module.

### Task 3: Frozen ImageNet Transfer Trainer

**Files:**

- Create: `tests/test_train_imagenet_transfer.py`
- Create: `scripts/train_imagenet_transfer.py`

**Steps:**

1. Add failing tests for conversion of a normalized one-channel log-mel batch into a three-channel 224x224 ImageNet input.
2. Add a failing test that the ResNet50 backbone is frozen while the 80-label head is trainable.
3. Implement dataset conversion, torchvision weight loading, frozen backbone, global pooling, dropout and multilabel head.
4. Reuse the existing split, `pos_weight`, lwlrap and submission validation.
5. Run targeted tests without downloading weights.

### Task 4: Local Ablations

**Experiments:**

1. Separable-residual CNN with baseline augmentation.
2. Best current head256 CNN plus temporal reversal.
3. Best current head256 CNN plus contrast.
4. Best safe augmentation combination only if individual evidence supports it.
5. Frozen ResNet50 ImageNet head training.

Use seed 42, the same 80/20 holdout and early stopping. Record best lwlrap, epoch, parameter count and runtime. Do not full-train candidates that are both weaker and highly correlated with existing models.

### Task 5: Full Training and Kaggle

1. Replay the selected epoch/LR schedule on all curated rows.
2. Generate standalone predictions and blends against `final_relu2seed_50_50.csv`.
3. Validate shape, columns, filenames and probability range.
4. Upload through the private copy dataset/kernel.
5. Download and compare both dataset and kernel output byte-for-byte before submitting kernel version to `freesound-audio-tagging-2019`.
6. Stop the weight sweep when leaderboard values show a clear local maximum.

### Task 6: Documentation and Verification

**Files:**

- Modify: `experiments/resultados.md`
- Create: `research/cats_dogs_transfer.md`
- Modify: `research/improvement_plan.md`
- Modify: `research/solutions.md`

Run:

```bash
.venv/bin/python -m unittest discover -s tests
python3 -m py_compile scripts/train_logmel_cnn.py scripts/predict_logmel_cnn.py scripts/train_imagenet_transfer.py
```

Verify the final CSV against `sample_submission.csv`, compare its SHA-256 with the downloaded kernel output, and query the final Kaggle score.
