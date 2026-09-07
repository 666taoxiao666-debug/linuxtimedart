#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

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
N_FOLDS="${N_FOLDS:-1}"
SPLIT="${SPLIT:-time_ratio}"
PRED_LEN="${PRED_LEN:-12}"
MODEL="${MODEL:-PromptTimeDART}"
RUN_ID="${RUN_ID:-final_h${PRED_LEN}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
LOG_PARAMETERS="${MODEL}_h${PRED_LEN}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_pts${FORECAST_PLOT_POINTS:-150}_tid${FORECAST_PLOT_TURBINE_ID:-auto}"
sdwpf_log_init "final_eval" "${LOG_PARAMETERS}" "test.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_ENV_FILE="$(sdwpf_log_sidecar env)"
LOG_SUMMARY_FILE="$(sdwpf_log_sidecar summary.txt)"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-${SDWPF_LOG_DIR}/artifacts}"
FORECAST_PLOT_POINTS="${FORECAST_PLOT_POINTS:-150}"

{
    echo "RUN_ID=${RUN_ID}"
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
    echo "REGIME_PROMPT=${REGIME_PROMPT:-1}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD:-trend_quantile}"
    echo "REGIME_CALIBRATION_QUANTILE=${REGIME_CALIBRATION_QUANTILE:-0.3333333333}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_ENV_FILE}"

RUN_ID="${RUN_ID}" \
MODEL="${MODEL}" \
FOLD="${FOLD}" \
N_FOLDS="${N_FOLDS}" \
SPLIT="${SPLIT}" \
SEED="${SEED}" \
PRED_LEN="${PRED_LEN}" \
CONFIRM_FINAL_EVAL=1 \
FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT}" \
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR}" \
FORECAST_PLOT_POINTS="${FORECAST_PLOT_POINTS}" \
FORECAST_PLOT_TURBINE_ID="${FORECAST_PLOT_TURBINE_ID:-}" \
FORECAST_PLOT_START="${FORECAST_PLOT_START:-}" \
REGIME_PROMPT="${REGIME_PROMPT:-1}" \
REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-trend_quantile}" \
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}" \
bash scripts/eval/SDWPF_final_eval.sh 2>&1 | tee "${SDWPF_LOG_FILE}"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_ENV_FILE}"
{
    grep -E '^[0-9]+->[0-9]+ \| original-scale |^  0-4 h |^  4-16 h |^Detailed forecast report:|^\[AUDIT\]' \
        "${SDWPF_LOG_FILE}" || true
} > "${LOG_SUMMARY_FILE}"
echo "[EVAL] Log: ${SDWPF_LOG_FILE}"
echo "[EVAL] Summary: ${LOG_SUMMARY_FILE}"
