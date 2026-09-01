#!/usr/bin/env bash
set -euo pipefail

# Rolling-origin folds + three seeds for the main TimeDART protocol.
for FOLD in 0 1 2; do
  for SEED in 2024 2025 2026; do
    echo "===== fold=${FOLD} seed=${SEED} ====="
    FOLD="${FOLD}" SEED="${SEED}" bash scripts/finetune/SDWPF.sh
  done
done
