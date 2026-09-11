#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

if [[ -z "${FINETUNE_CHECKPOINT:-}" ]]; then
    echo "FINETUNE_CHECKPOINT is required." >&2
    exit 2
fi

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling_holdout}"
PRED_LEN="${PRED_LEN:-12}"
MODEL="${MODEL:-PromptTimeDART}"
PROMPT_ROUTER="${PROMPT_ROUTER:-compositional_wiki}"
if [[ "${PROMPT_ROUTER}" == "scene_wiki" ]]; then
    REGIME_LABEL_METHOD="scene_wiki"
else
    REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-trend_quantile}"
fi
if [[ "${PROMPT_ROUTER}" == "compositional_wiki" ]]; then
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_event_factor_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_event_factor_wiki_qwen.npz}"
elif [[ "${PROMPT_ROUTER}" == "hybrid_wiki" ]]; then
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_exception_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_exception_wiki_qwen.npz}"
else
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_regime_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_regime_wiki_qwen.npz}"
fi
FORECAST_PLOT_POINTS="${FORECAST_PLOT_POINTS:-150}"
RUN_ID="${RUN_ID:-validation_plot_h${PRED_LEN}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

if [[ "${SPLIT}" != "rolling_holdout" ]]; then
    echo "Validation plotting is locked to SPLIT=rolling_holdout." >&2
    exit 2
fi

LOG_PARAMETERS="${MODEL}_h${PRED_LEN}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_pts${FORECAST_PLOT_POINTS}_tid${FORECAST_PLOT_TURBINE_ID:-auto}"
sdwpf_log_init "validation_plot" "${LOG_PARAMETERS}" "validation.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_ENV_FILE="$(sdwpf_log_sidecar env)"
LOG_SUMMARY_FILE="$(sdwpf_log_sidecar summary.txt)"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-${SDWPF_LOG_DIR}/artifacts}"

{
    echo "RUN_ID=${RUN_ID}"
    echo "EVAL_SPLIT=val"
    echo "MODEL=${MODEL}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SPLIT=${SPLIT}"
    echo "SEED=${SEED}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "FINETUNE_CHECKPOINT=${FINETUNE_CHECKPOINT}"
    echo "REPORT_OUTPUT_DIR=${REPORT_OUTPUT_DIR}"
    echo "FORECAST_PLOT_POINTS=${FORECAST_PLOT_POINTS}"
    echo "FORECAST_PLOT_TURBINE_ID=${FORECAST_PLOT_TURBINE_ID:-}"
    echo "FORECAST_PLOT_START=${FORECAST_PLOT_START:-}"
    echo "PROMPT_ROUTER=${PROMPT_ROUTER}"
    echo "SCENE_WIKI_ACTIVATION_THRESHOLD=${SCENE_WIKI_ACTIVATION_THRESHOLD:-0.55}"
    echo "SCENE_WIKI_CONFIDENCE_POWER=${SCENE_WIKI_CONFIDENCE_POWER:-1.0}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_ENV_FILE}"

EVAL_SPLIT=val \
RUN_ID="${RUN_ID}" \
MODEL="${MODEL}" \
FOLD="${FOLD}" \
N_FOLDS="${N_FOLDS}" \
SPLIT="${SPLIT}" \
SEED="${SEED}" \
PRED_LEN="${PRED_LEN}" \
FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT}" \
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR}" \
FORECAST_PLOT_POINTS="${FORECAST_PLOT_POINTS}" \
FORECAST_PLOT_TURBINE_ID="${FORECAST_PLOT_TURBINE_ID:-}" \
FORECAST_PLOT_START="${FORECAST_PLOT_START:-}" \
REGIME_PROMPT="${REGIME_PROMPT:-1}" \
PROMPT_ROUTER="${PROMPT_ROUTER}" \
SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG}" \
SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS}" \
SCENE_WIKI_TOP_K="${SCENE_WIKI_TOP_K:-2}" \
SCENE_WIKI_TEMPERATURE="${SCENE_WIKI_TEMPERATURE:-0.2}" \
SCENE_WIKI_RULE_WEIGHT="${SCENE_WIKI_RULE_WEIGHT:-2.0}" \
SCENE_WIKI_PROMPT_GATE_INIT="${SCENE_WIKI_PROMPT_GATE_INIT:--2.2}" \
SCENE_WIKI_ACTIVATION_THRESHOLD="${SCENE_WIKI_ACTIVATION_THRESHOLD:-0.55}" \
SCENE_WIKI_CONFIDENCE_POWER="${SCENE_WIKI_CONFIDENCE_POWER:-1.0}" \
WIKI_LLM_PATH="${WIKI_LLM_PATH:-Qwen/Qwen2.5-0.5B}" \
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-auto}" \
REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD}" \
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}" \
bash scripts/eval/SDWPF_final_eval.sh 2>&1 | tee "${SDWPF_LOG_FILE}"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_ENV_FILE}"
{
    grep -E '^[0-9]+->[0-9]+ \| original-scale |^  0-4 h |^Detailed val forecast report:|^\[AUDIT\]' \
        "${SDWPF_LOG_FILE}" || true
} > "${LOG_SUMMARY_FILE}"

echo "[VALIDATION_PLOT] Log: ${SDWPF_LOG_FILE}"
echo "[VALIDATION_PLOT] Summary: ${LOG_SUMMARY_FILE}"
echo "[VALIDATION_PLOT] Figure: ${REPORT_OUTPUT_DIR}/forecast_trace.png"
