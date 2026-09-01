#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_FINAL_EVAL:-0}" != "1" ]]; then
    echo "Set CONFIRM_FINAL_EVAL=1 only after validation-based model selection." >&2
    exit 2
fi
if [[ -z "${FINETUNE_CHECKPOINT:-}" ]]; then
    echo "FINETUNE_CHECKPOINT is required." >&2
    exit 2
fi

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling}"
PRED_LEN="${PRED_LEN:-24}"
MODEL="${MODEL:-PromptTimeDART}"
RUN_ID="${RUN_ID:-final_h${PRED_LEN}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="outputs/logs/SDWPF/${RUN_ID}"
mkdir -p "${LOG_DIR}"

{
    echo "RUN_ID=${RUN_ID}"
    echo "MODEL=${MODEL}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SPLIT=${SPLIT}"
    echo "SEED=${SEED}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "FINETUNE_CHECKPOINT=${FINETUNE_CHECKPOINT}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_DIR}/eval.env"

RUN_ID="${RUN_ID}" \
MODEL="${MODEL}" \
FOLD="${FOLD}" \
N_FOLDS="${N_FOLDS}" \
SPLIT="${SPLIT}" \
SEED="${SEED}" \
PRED_LEN="${PRED_LEN}" \
CONFIRM_FINAL_EVAL=1 \
FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT}" \
bash scripts/eval/SDWPF_final_eval.sh 2>&1 | tee "${LOG_DIR}/test.log"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_DIR}/eval.env"
echo "[EVAL] Log: ${LOG_DIR}/test.log"
