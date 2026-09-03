#!/usr/bin/env bash
set -euo pipefail

# Leakage-safe full method: fold-matched PromptTimeDART pretraining followed by
# supervised fine-tuning.  Test evaluation is intentionally a separate command.
SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling_holdout}"
PRED_LEN="${PRED_LEN:-12}"
PRETRAIN_PRED_LEN="${PRETRAIN_PRED_LEN:-24}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-20}"
PRETRAIN_LEARNING_RATE="${PRETRAIN_LEARNING_RATE:-0.0001}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-3}"
PRETRAIN_LAMBDA_CE="${PRETRAIN_LAMBDA_CE:-0.02}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-0.000001}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.0001}"
FINETUNE_PATIENCE="${FINETUNE_PATIENCE:-2}"
FINETUNE_PCT_START="${FINETUNE_PCT_START:-0.20}"
LOSS="${LOSS:-MIXED}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.2}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-original_mae}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
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
    echo "PRETRAIN_PRED_LEN=${PRETRAIN_PRED_LEN}"
    echo "PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS}"
    echo "PRETRAIN_LEARNING_RATE=${PRETRAIN_LEARNING_RATE}"
    echo "PRETRAIN_PATIENCE=${PRETRAIN_PATIENCE}"
    echo "PRETRAIN_LAMBDA_CE=${PRETRAIN_LAMBDA_CE}"
    echo "FINETUNE_EPOCHS=${FINETUNE_EPOCHS}"
    echo "FINETUNE_LEARNING_RATE=${FINETUNE_LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "FINETUNE_PATIENCE=${FINETUNE_PATIENCE}"
    echo "FINETUNE_PCT_START=${FINETUNE_PCT_START}"
    echo "LOSS=${LOSS}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "EARLY_STOP_METRIC=${EARLY_STOP_METRIC}"
    echo "RATED_POWER=${RATED_POWER}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_DIR}/pipeline.env"

echo "[PIPELINE] Stage 1/2: fold-matched pretraining"
SEED="${SEED}" \
FOLD="${FOLD}" \
N_FOLDS="${N_FOLDS}" \
SPLIT="${SPLIT}" \
PRED_LEN="${PRETRAIN_PRED_LEN}" \
EVAL_STRIDE=6 \
TRAIN_EPOCHS="${PRETRAIN_EPOCHS}" \
LEARNING_RATE="${PRETRAIN_LEARNING_RATE}" \
PATIENCE="${PRETRAIN_PATIENCE}" \
LAMBDA_CE="${PRETRAIN_LAMBDA_CE}" \
RATED_POWER="${RATED_POWER}" \
GPU="${GPU}" \
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
TRAIN_EPOCHS="${FINETUNE_EPOCHS}" \
LEARNING_RATE="${FINETUNE_LEARNING_RATE}" \
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
PATIENCE="${FINETUNE_PATIENCE}" \
PCT_START="${FINETUNE_PCT_START}" \
LOSS="${LOSS}" \
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT}" \
EARLY_STOP_METRIC="${EARLY_STOP_METRIC}" \
RATED_POWER="${RATED_POWER}" \
GPU="${GPU}" \
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
if [[ "${SPLIT}" == "rolling_holdout" ]]; then
    echo "[PIPELINE] CV mode: the shared final holdout remains sealed."
    echo "[PIPELINE] Do not test this fold while selecting hyperparameters."
else
    echo "[PIPELINE] Final test (run only after model selection):"
    echo "CONFIRM_FINAL_EVAL=1 MODEL=PromptTimeDART SEED=${SEED} FOLD=${FOLD} N_FOLDS=${N_FOLDS} SPLIT=${SPLIT} PRED_LEN=${PRED_LEN} FINETUNE_CHECKPOINT='${CHECKPOINT}' bash scripts/eval/SDWPF_logged_final_eval.sh"
fi
