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
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.000005}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.2}"
PROMPT_ROUTER="${PROMPT_ROUTER:-hybrid_wiki}"
if [[ "${PROMPT_ROUTER}" == "scene_wiki" ]]; then
    REGIME_LABEL_METHOD="scene_wiki"
else
    REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-trend_quantile}"
fi
if [[ "${PROMPT_ROUTER}" == "hybrid_wiki" ]]; then
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_exception_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_exception_wiki_qwen.npz}"
else
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_regime_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_regime_wiki_qwen.npz}"
fi
SCENE_WIKI_TOP_K="${SCENE_WIKI_TOP_K:-2}"
SCENE_WIKI_TEMPERATURE="${SCENE_WIKI_TEMPERATURE:-0.2}"
SCENE_WIKI_RULE_WEIGHT="${SCENE_WIKI_RULE_WEIGHT:-2.0}"
SCENE_WIKI_PROMPT_GATE_INIT="${SCENE_WIKI_PROMPT_GATE_INIT:--2.2}"
WIKI_LLM_PATH="${WIKI_LLM_PATH:-Qwen/Qwen2.5-0.5B}"
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-auto}"
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}"
CV_STAMP="$(date +%Y%m%d_%H%M%S)"
CV_ID="${CV_ID:-cv_h${PRED_LEN}_${SPLIT}_${#CV_FOLDS[@]}fold_${#CV_SEEDS[@]}seed_${CV_STAMP}}"
LOG_PARAMETERS="h${PRED_LEN}_${SPLIT}_folds${FOLDS// /-}_seeds${SEEDS// /-}_blr${FINETUNE_LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_mix${MIX_MSE_WEIGHT}_${PROMPT_ROUTER}_reg${REGIME_LABEL_METHOD}_rq${REGIME_CALIBRATION_QUANTILE}"
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
    echo "PROMPT_ROUTER=${PROMPT_ROUTER}"
    echo "SCENE_WIKI_CONFIG=${SCENE_WIKI_CONFIG}"
    echo "SCENE_WIKI_EMBEDDINGS=${SCENE_WIKI_EMBEDDINGS}"
    echo "SCENE_WIKI_TOP_K=${SCENE_WIKI_TOP_K}"
    echo "SCENE_WIKI_TEMPERATURE=${SCENE_WIKI_TEMPERATURE}"
    echo "SCENE_WIKI_RULE_WEIGHT=${SCENE_WIKI_RULE_WEIGHT}"
    echo "SCENE_WIKI_PROMPT_GATE_INIT=${SCENE_WIKI_PROMPT_GATE_INIT}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD}"
    echo "REGIME_CALIBRATION_QUANTILE=${REGIME_CALIBRATION_QUANTILE}"
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
        PROMPT_ROUTER="${PROMPT_ROUTER}" \
        SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG}" \
        SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS}" \
        SCENE_WIKI_TOP_K="${SCENE_WIKI_TOP_K}" \
        SCENE_WIKI_TEMPERATURE="${SCENE_WIKI_TEMPERATURE}" \
        SCENE_WIKI_RULE_WEIGHT="${SCENE_WIKI_RULE_WEIGHT}" \
        SCENE_WIKI_PROMPT_GATE_INIT="${SCENE_WIKI_PROMPT_GATE_INIT}" \
        WIKI_LLM_PATH="${WIKI_LLM_PATH}" \
        WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE}" \
        REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD}" \
        REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE}" \
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
python scripts/summarize_sdwpf_cv.py \
    "${CV_LOG_DIR}/summary.txt" \
    --output-dir "${CV_LOG_DIR}" \
    --expected-folds "${FOLDS}" \
    --expected-seeds "${SEEDS}"
echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${CV_LOG_DIR}/cv.env"

echo "[CV] Completed. Select one configuration from validation metrics only."
echo "[CV] Then train the frozen configuration with SPLIT=time_ratio."
echo "[CV] Log directory: ${CV_LOG_DIR}"
echo "[CV] Summary: ${CV_LOG_DIR}/summary.txt"
echo "[CV] Metrics: ${CV_LOG_DIR}/cv_metrics_summary.txt"
