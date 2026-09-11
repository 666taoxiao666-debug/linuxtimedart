#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

# Run only after every hyper-parameter and ablation choice is frozen.
# time_ratio uses 70% train, 10% validation and the final 20% as test.
PRED_LEN="${PRED_LEN:-12}"
SPLIT="${SPLIT:-time_ratio}"
if [[ "${SPLIT}" != "time_ratio" ]]; then
    echo "SDWPF_paper_final.sh requires SPLIT=time_ratio." >&2
    exit 2
fi
FINETUNE_LEARNING_RATE="${FINETUNE_LEARNING_RATE:-0.000001}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.000005}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.2}"
PROMPT_ROUTER="${PROMPT_ROUTER:-scene_wiki}"
if [[ "${PROMPT_ROUTER}" == "scene_wiki" ]]; then
    REGIME_LABEL_METHOD="scene_wiki"
else
    REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-trend_quantile}"
fi
SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_regime_wiki.json}"
SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_regime_wiki_qwen.npz}"
SCENE_WIKI_TOP_K="${SCENE_WIKI_TOP_K:-2}"
SCENE_WIKI_TEMPERATURE="${SCENE_WIKI_TEMPERATURE:-0.2}"
SCENE_WIKI_RULE_WEIGHT="${SCENE_WIKI_RULE_WEIGHT:-2.0}"
SCENE_WIKI_PROMPT_GATE_INIT="${SCENE_WIKI_PROMPT_GATE_INIT:--2.2}"
WIKI_LLM_PATH="${WIKI_LLM_PATH:-Qwen/Qwen2.5-0.5B}"
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-auto}"
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}"
FINAL_STAMP="$(date +%Y%m%d_%H%M%S)"
FINAL_ID="${FINAL_ID:-finaltrain_h${PRED_LEN}_${SPLIT}_3seed_${FINAL_STAMP}}"
LOG_PARAMETERS="h${PRED_LEN}_${SPLIT}_3seed_blr${FINETUNE_LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_mix${MIX_MSE_WEIGHT}_${PROMPT_ROUTER}_reg${REGIME_LABEL_METHOD}_rq${REGIME_CALIBRATION_QUANTILE}"
sdwpf_log_init "final_train" "${LOG_PARAMETERS}" "final_train.log" "${FINAL_ID}"
sdwpf_log_install_exit_trap
FINAL_LOG_DIR="${SDWPF_LOG_DIR}"
sdwpf_log_capture

{
    echo "TASK=final_train"
    echo "FINAL_ID=${FINAL_ID}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "SPLIT=${SPLIT}"
    echo "SEEDS=2024,2025,2026"
    echo "FINETUNE_LEARNING_RATE=${FINETUNE_LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "PROMPT_ROUTER=${PROMPT_ROUTER}"
    echo "SCENE_WIKI_CONFIG=${SCENE_WIKI_CONFIG}"
    echo "SCENE_WIKI_EMBEDDINGS=${SCENE_WIKI_EMBEDDINGS}"
    echo "SCENE_WIKI_PROMPT_GATE_INIT=${SCENE_WIKI_PROMPT_GATE_INIT}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD}"
    echo "REGIME_CALIBRATION_QUANTILE=${REGIME_CALIBRATION_QUANTILE}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${FINAL_LOG_DIR}/final_train.env"

for seed in 2024 2025 2026; do
    echo "===== FINAL TRAIN seed=${seed}; test remains sealed ====="
    SEED="${seed}" \
    FOLD=0 \
    N_FOLDS=1 \
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
    PIPELINE_ID="${FINAL_ID}_s${seed}" \
    PRETRAIN_RUN_ID= \
    SDWPF_LOG_DIR="${FINAL_LOG_DIR}/runs/s${seed}" \
    SDWPF_LOG_FILE= \
    bash scripts/train/SDWPF_full_pipeline.sh
done

{
    for seed in 2024 2025 2026; do
        echo "===== FINAL TRAIN seed=${seed} ====="
        cat "${FINAL_LOG_DIR}/runs/s${seed}/summary.txt"
    done
} > "${FINAL_LOG_DIR}/summary.txt"
echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${FINAL_LOG_DIR}/final_train.env"

echo "[FINAL TRAIN] Three predetermined seed checkpoints are ready."
echo "[FINAL TRAIN] Evaluate each checkpoint exactly once with the command printed above."
echo "[FINAL TRAIN] Log directory: ${FINAL_LOG_DIR}"
echo "[FINAL TRAIN] Summary: ${FINAL_LOG_DIR}/summary.txt"
