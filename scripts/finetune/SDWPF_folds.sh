#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

# Random-init TimeDART ablation on sealed-holdout CV folds.
PRED_LEN="${PRED_LEN:-12}"
SPLIT="${SPLIT:-rolling_holdout}"
N_FOLDS="${N_FOLDS:-3}"
if [[ "${SPLIT}" != "rolling_holdout" ]]; then
  echo "SDWPF_folds.sh requires SPLIT=rolling_holdout." >&2
  exit 2
fi
if [[ "${N_FOLDS}" != "3" ]]; then
  echo "SDWPF_folds.sh runs exactly three folds; N_FOLDS must be 3." >&2
  exit 2
fi
LEARNING_RATE="${LEARNING_RATE:-0.00003}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.0001}"
FOLDS_STAMP="$(date +%Y%m%d_%H%M%S)"
FOLDS_ID="${FOLDS_ID:-random_folds_h${PRED_LEN}_${SPLIT}_${FOLDS_STAMP}}"
LOG_PARAMETERS="TimeDART_random_h${PRED_LEN}_${SPLIT}_3fold_3seed_lr${LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}"
sdwpf_log_init "folds" "${LOG_PARAMETERS}" "folds.log" "${FOLDS_ID}"
sdwpf_log_install_exit_trap
FOLDS_LOG_DIR="${SDWPF_LOG_DIR}"
sdwpf_log_capture

{
  echo "TASK=random_folds"
  echo "FOLDS_ID=${FOLDS_ID}"
  echo "PRED_LEN=${PRED_LEN}"
  echo "SPLIT=${SPLIT}"
  echo "N_FOLDS=${N_FOLDS}"
  echo "FOLDS=0,1,2"
  echo "SEEDS=2024,2025,2026"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
  echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${FOLDS_LOG_DIR}/folds.env"

for FOLD in 0 1 2; do
  for SEED in 2024 2025 2026; do
    echo "===== fold=${FOLD} seed=${SEED} ====="
    FOLD="${FOLD}" N_FOLDS="${N_FOLDS}" SEED="${SEED}" SPLIT="${SPLIT}" \
      PRED_LEN="${PRED_LEN}" \
      LEARNING_RATE="${LEARNING_RATE}" \
      NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
      RUN_ID="${FOLDS_ID}_f${FOLD}_s${SEED}" \
      SDWPF_LOG_DIR="${FOLDS_LOG_DIR}/runs/f${FOLD}_s${SEED}" \
      SDWPF_LOG_FILE= \
      bash scripts/finetune/SDWPF.sh
  done
done

{
  for FOLD in 0 1 2; do
    for SEED in 2024 2025 2026; do
      echo "===== fold=${FOLD} seed=${SEED} ====="
      cat "${FOLDS_LOG_DIR}/runs/f${FOLD}_s${SEED}/finetune.summary.txt"
    done
  done
} > "${FOLDS_LOG_DIR}/summary.txt"
echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${FOLDS_LOG_DIR}/folds.env"
echo "[FOLDS] Log directory: ${FOLDS_LOG_DIR}"
echo "[FOLDS] Summary: ${FOLDS_LOG_DIR}/summary.txt"
