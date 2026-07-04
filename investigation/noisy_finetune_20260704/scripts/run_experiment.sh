#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SUBMIT_KAGGLE="${SUBMIT_KAGGLE:-0}"

"${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/validate_experiment.py --static
"${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/prepare_experiment.py
"${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/build_noisy_caches.py
"${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/build_f1024_memmap_caches.py
for BRANCH in separable_headsep globalmel_sep_temporal sep_temporal_f1024; do
  "${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/finetune_mixed_noisy.py \
    --branches "${BRANCH}" \
    --num-workers "${NUM_WORKERS}"
done

if [[ "${SUBMIT_KAGGLE}" == "1" ]]; then
  "${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/evaluate_and_blend.py --submit-kaggle
else
  "${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/evaluate_and_blend.py
fi

"${PYTHON_BIN}" investigation/noisy_finetune_20260704/scripts/validate_experiment.py
