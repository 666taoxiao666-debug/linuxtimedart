#!/usr/bin/env bash
set -euo pipefail

# Random-init TimeDART ablation on sealed-holdout CV folds.
for FOLD in 0 1 2; do
  for SEED in 2024 2025 2026; do
    echo "===== fold=${FOLD} seed=${SEED} ====="
    FOLD="${FOLD}" SEED="${SEED}" SPLIT=rolling_holdout \
      bash scripts/finetune/SDWPF.sh
  done
done
