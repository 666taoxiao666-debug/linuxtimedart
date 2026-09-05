#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

if [[ -z "${PRETRAIN_RUN_ID:-}" ]]; then
    echo "PRETRAIN_RUN_ID must identify the matched fold/seed checkpoint." >&2
    exit 2
fi

COMMON_ID="${ABLATION_ID:-ablation_h${PRED_LEN:-12}_${SPLIT:-rolling_holdout}_f${FOLD:-0}_s${SEED:-2024}_$(date +%Y%m%d_%H%M%S)}"
PRED_LEN="${PRED_LEN:-12}"
SPLIT="${SPLIT:-rolling_holdout}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SEED="${SEED:-2024}"
MODEL="PromptTimeDART"
LEARNING_RATE="${LEARNING_RATE:-0.000001}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.0001}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
PCT_START="${PCT_START:-0.20}"
PATIENCE="${PATIENCE:-2}"
LOSS="${LOSS:-MIXED}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.2}"
LOG_PARAMETERS="${MODEL}_h${PRED_LEN}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_blr${LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_${LOSS}_mix${MIX_MSE_WEIGHT}_ep${TRAIN_EPOCHS}_pat${PATIENCE}_4variants"
sdwpf_log_init "ablation" "${LOG_PARAMETERS}" "ablation.log" "${COMMON_ID}"
sdwpf_log_install_exit_trap
ABLATION_LOG_DIR="${SDWPF_LOG_DIR}"
sdwpf_log_capture

{
    echo "TASK=ablation"
    echo "ABLATION_ID=${COMMON_ID}"
    echo "PRETRAIN_RUN_ID=${PRETRAIN_RUN_ID}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "SPLIT=${SPLIT}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SEED=${SEED}"
    echo "MODEL=${MODEL}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "PCT_START=${PCT_START}"
    echo "PATIENCE=${PATIENCE}"
    echo "LOSS=${LOSS}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${ABLATION_LOG_DIR}/ablation.env"

echo "[ABLATION 1/4] full pretrained method"
MODEL="${MODEL}" PRED_LEN="${PRED_LEN}" SPLIT="${SPLIT}" FOLD="${FOLD}" N_FOLDS="${N_FOLDS}" SEED="${SEED}" \
LEARNING_RATE="${LEARNING_RATE}" NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" PCT_START="${PCT_START}" PATIENCE="${PATIENCE}" \
LOSS="${LOSS}" MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT}" PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" \
RUN_ID="${COMMON_ID}_full" \
ALLOW_RANDOM=0 CHANNEL_PRIOR=1 OP_CONTEXT=1 REVIN_KEEP_WIND=1 \
SDWPF_LOG_DIR="${ABLATION_LOG_DIR}/runs/full" SDWPF_LOG_FILE= \
bash scripts/finetune/SDWPF_ablation_prompt.sh

echo "[ABLATION 2/4] matched random initialization"
MODEL="${MODEL}" PRED_LEN="${PRED_LEN}" SPLIT="${SPLIT}" FOLD="${FOLD}" N_FOLDS="${N_FOLDS}" SEED="${SEED}" \
LEARNING_RATE="${LEARNING_RATE}" NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" PCT_START="${PCT_START}" PATIENCE="${PATIENCE}" \
LOSS="${LOSS}" MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT}" PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" \
RUN_ID="${COMMON_ID}_random" \
ALLOW_RANDOM=1 CHANNEL_PRIOR=1 OP_CONTEXT=1 REVIN_KEEP_WIND=1 \
SDWPF_LOG_DIR="${ABLATION_LOG_DIR}/runs/random" SDWPF_LOG_FILE= \
bash scripts/finetune/SDWPF_ablation_prompt.sh

echo "[ABLATION 3/4] pretrained with uniform channel initialization"
MODEL="${MODEL}" PRED_LEN="${PRED_LEN}" SPLIT="${SPLIT}" FOLD="${FOLD}" N_FOLDS="${N_FOLDS}" SEED="${SEED}" \
LEARNING_RATE="${LEARNING_RATE}" NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" PCT_START="${PCT_START}" PATIENCE="${PATIENCE}" \
LOSS="${LOSS}" MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT}" PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" \
RUN_ID="${COMMON_ID}_uniform_prior" \
ALLOW_RANDOM=0 CHANNEL_PRIOR=0 OP_CONTEXT=1 REVIN_KEEP_WIND=1 \
SDWPF_LOG_DIR="${ABLATION_LOG_DIR}/runs/uniform_prior" SDWPF_LOG_FILE= \
bash scripts/finetune/SDWPF_ablation_prompt.sh

echo "[ABLATION 4/4] pretrained without operating-point context"
MODEL="${MODEL}" PRED_LEN="${PRED_LEN}" SPLIT="${SPLIT}" FOLD="${FOLD}" N_FOLDS="${N_FOLDS}" SEED="${SEED}" \
LEARNING_RATE="${LEARNING_RATE}" NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" PCT_START="${PCT_START}" PATIENCE="${PATIENCE}" \
LOSS="${LOSS}" MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT}" PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" \
RUN_ID="${COMMON_ID}_no_context" \
ALLOW_RANDOM=0 CHANNEL_PRIOR=1 OP_CONTEXT=0 REVIN_KEEP_WIND=1 \
SDWPF_LOG_DIR="${ABLATION_LOG_DIR}/runs/no_context" SDWPF_LOG_FILE= \
bash scripts/finetune/SDWPF_ablation_prompt.sh

{
    for variant in full random uniform_prior no_context; do
        echo "===== ${variant} ====="
        cat "${ABLATION_LOG_DIR}/runs/${variant}/finetune.summary.txt"
    done
} > "${ABLATION_LOG_DIR}/summary.txt"
echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${ABLATION_LOG_DIR}/ablation.env"
echo "[ABLATION] Log directory: ${ABLATION_LOG_DIR}"
echo "[ABLATION] Summary: ${ABLATION_LOG_DIR}/summary.txt"
