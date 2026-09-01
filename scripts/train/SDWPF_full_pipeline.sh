#!/usr/bin/env bash
set -euo pipefail

# Leakage-safe full method: fold-matched PromptTimeDART pretraining followed by
# supervised fine-tuning.  Test evaluation is intentionally a separate command.
SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling}"
PRED_LEN="${PRED_LEN:-24}"
PIPELINE_ID="${PIPELINE_ID:-full_h${PRED_LEN}_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-${PIPELINE_ID}_pretrain}"
LOG_DIR="outputs/logs/SDWPF/${PIPELINE_ID}"

mkdir -p "${LOG_DIR}"
{
    echo "PIPELINE_ID=${PIPELINE_ID}"
    echo "PRETRAIN_RUN_ID=${PRETRAIN_RUN_ID}"
    echo "SEED=${SEED}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SPLIT=${SPLIT}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_DIR}/pipeline.env"

echo "[PIPELINE] Stage 1/2: fold-matched pretraining"
SEED="${SEED}" \
FOLD="${FOLD}" \
N_FOLDS="${N_FOLDS}" \
SPLIT="${SPLIT}" \
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" \
RUN_ID="${PIPELINE_ID}_pretrain" \
bash scripts/pretrain/SDWPF.sh 2>&1 | tee "${LOG_DIR}/pretrain.log"

echo "[PIPELINE] Stage 2/2: supervised fine-tuning with the matched checkpoint"
SEED="${SEED}" \
FOLD="${FOLD}" \
N_FOLDS="${N_FOLDS}" \
SPLIT="${SPLIT}" \
PRED_LEN="${PRED_LEN}" \
EVAL_STRIDE="${PRED_LEN}" \
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" \
ALLOW_RANDOM=0 \
RUN_ID="${PIPELINE_ID}_finetune" \
bash scripts/finetune/SDWPF_ablation_prompt.sh 2>&1 | tee "${LOG_DIR}/finetune.log"

CHECKPOINT="$(grep -F '[AUDIT] FINETUNE_CHECKPOINT=' "${LOG_DIR}/finetune.log" | tail -n 1 | sed 's/^.*FINETUNE_CHECKPOINT=//')"
if [[ -z "${CHECKPOINT}" ]]; then
    echo "Fine-tuning completed but no checkpoint marker was found." >&2
    exit 3
fi

{
    echo "FINETUNE_CHECKPOINT=${CHECKPOINT}"
    echo "COMPLETED_AT=$(date --iso-8601=seconds)"
} >> "${LOG_DIR}/pipeline.env"

echo "[PIPELINE] Training complete; test split has not been read."
echo "[PIPELINE] Logs: ${LOG_DIR}"
echo "[PIPELINE] FINETUNE_CHECKPOINT=${CHECKPOINT}"
echo "[PIPELINE] Final test (run once after model selection):"
echo "CONFIRM_FINAL_EVAL=1 MODEL=PromptTimeDART SEED=${SEED} FOLD=${FOLD} N_FOLDS=${N_FOLDS} SPLIT=${SPLIT} PRED_LEN=${PRED_LEN} FINETUNE_CHECKPOINT='${CHECKPOINT}' bash scripts/eval/SDWPF_logged_final_eval.sh"
