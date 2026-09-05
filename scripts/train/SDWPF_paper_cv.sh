#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

# Hyper-parameter selection only.  rolling_holdout keeps the final 20% sealed
# and uses three disjoint validation blocks inside the preceding 10%.
PRED_LEN="${PRED_LEN:-12}"
SPLIT="${SPLIT:-rolling_holdout}"
if [[ "${SPLIT}" != "rolling_holdout" ]]; then
    echo "SDWPF_paper_cv.sh requires SPLIT=rolling_holdout." >&2
    exit 2
fi
FOLDS="${FOLDS:-0 1 2}"
SEEDS="${SEEDS:-2024 2025 2026}"
read -r -a CV_FOLDS <<< "${FOLDS}"
read -r -a CV_SEEDS <<< "${SEEDS}"
if [[ "${#CV_FOLDS[@]}" -eq 0 || "${#CV_SEEDS[@]}" -eq 0 ]]; then
    echo "FOLDS and SEEDS must each contain at least one value." >&2
    exit 2
fi
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-0.000001}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.0001}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.2}"
CV_STAMP="$(date +%Y%m%d_%H%M%S)"
CV_ID="${CV_ID:-cv_h${PRED_LEN}_${SPLIT}_${#CV_FOLDS[@]}fold_${#CV_SEEDS[@]}seed_${CV_STAMP}}"
LOG_PARAMETERS="h${PRED_LEN}_${SPLIT}_folds${FOLDS// /-}_seeds${SEEDS// /-}_blr${FINETUNE_LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_mix${MIX_MSE_WEIGHT}"
sdwpf_log_init "cv" "${LOG_PARAMETERS}" "cv.log" "${CV_ID}"
sdwpf_log_install_exit_trap
CV_LOG_DIR="${SDWPF_LOG_DIR}"
sdwpf_log_capture

{
    echo "TASK=cv"
    echo "CV_ID=${CV_ID}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "SPLIT=${SPLIT}"
    echo "FOLDS=${FOLDS}"
    echo "SEEDS=${SEEDS}"
    echo "FINETUNE_LEARNING_RATE=${FINETUNE_LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${CV_LOG_DIR}/cv.env"

for fold in "${CV_FOLDS[@]}"; do
    for seed in "${CV_SEEDS[@]}"; do
        echo "===== CV fold=${fold} seed=${seed} ====="
        FOLD="${fold}" \
        N_FOLDS=3 \
        SEED="${seed}" \
        SPLIT="${SPLIT}" \
        PRED_LEN="${PRED_LEN}" \
        FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE}" \
        NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
        MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT}" \
        PIPELINE_ID="${CV_ID}_f${fold}_s${seed}" \
        PRETRAIN_RUN_ID= \
        SDWPF_LOG_DIR="${CV_LOG_DIR}/runs/f${fold}_s${seed}" \
        SDWPF_LOG_FILE= \
        bash scripts/train/SDWPF_full_pipeline.sh
    done
done

{
    for fold in "${CV_FOLDS[@]}"; do
        for seed in "${CV_SEEDS[@]}"; do
            echo "===== CV fold=${fold} seed=${seed} ====="
            cat "${CV_LOG_DIR}/runs/f${fold}_s${seed}/summary.txt"
        done
    done
} > "${CV_LOG_DIR}/summary.txt"
echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${CV_LOG_DIR}/cv.env"

echo "[CV] Completed. Select one configuration from validation metrics only."
echo "[CV] Then train the frozen configuration with SPLIT=time_ratio."
echo "[CV] Log directory: ${CV_LOG_DIR}"
echo "[CV] Summary: ${CV_LOG_DIR}/summary.txt"
