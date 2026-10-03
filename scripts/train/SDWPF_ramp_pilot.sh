#!/usr/bin/env bash
set -euo pipefail

# Fast, validation-only paired pilot: reuse the matched trend pretraining,
# change only the zero-initialized history-ramp residual in fine-tuning.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"
POINTER="outputs/logs/SDWPF/ramp_pilot_latest.txt"

if [[ "${1:-}" == "--status" ]]; then
    [[ -f "${POINTER}" ]] || { echo "No ramp pilot has been launched."; exit 2; }
    result_dir="$(< "${POINTER}")"
    echo "RESULT_DIR=${result_dir}"
    [[ ! -f "${result_dir}/status.env" ]] || cat "${result_dir}/status.env"
    [[ ! -f "${result_dir}/summary.txt" ]] || cat "${result_dir}/summary.txt"
    [[ -f "${result_dir}/summary.txt" ]] || tail -n 30 "${result_dir}/launch.log"
    exit 0
fi
[[ -z "${1:-}" ]] || { echo "Usage: bash $0 [--status]" >&2; exit 2; }

TREND_CV_DIR="${TREND_CV_DIR:-}"
FOLD="${FOLD:-0}"
SEED="${SEED:-2024}"
[[ -f "${TREND_CV_DIR}/cv.env" ]] || {
    echo "TREND_CV_DIR must point to a completed pure-trend CV directory." >&2
    exit 2
}
read_value() { sed -n "s/^${2}=//p" "$1" | tail -n 1; }
[[ "$(read_value "${TREND_CV_DIR}/cv.env" PROMPT_ROUTER)" == "trend" ]] || {
    echo "TREND_CV_DIR must have PROMPT_ROUTER=trend." >&2
    exit 2
}
[[ "$(read_value "${TREND_CV_DIR}/cv.env" PRED_LEN)" == "12" ]] || {
    echo "This paired pilot requires the h12 trend baseline." >&2
    exit 2
}
[[ "${ROBUST_PITCH:-0}" == "0" ]] || {
    echo "Pitch repair changes pretraining data; do not reuse the old checkpoint for that ablation." >&2
    exit 2
}
stage="${TREND_CV_DIR}/runs/f${FOLD}_s${SEED}/pipeline.env"
[[ -f "${stage}" ]] || { echo "Missing matched pipeline: ${stage}" >&2; exit 2; }
PRETRAIN_RUN_ID="$(read_value "${stage}" PRETRAIN_RUN_ID)"
[[ -n "${PRETRAIN_RUN_ID}" ]] || { echo "Missing PRETRAIN_RUN_ID in ${stage}" >&2; exit 2; }

TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
PATIENCE="${PATIENCE:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.000001}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.000005}"
RAMP_LEARNING_RATE="${RAMP_LEARNING_RATE:-0}"
RUN_ID="${RUN_ID:-ramp_h12_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
parameters="h12_f${FOLD}_s${SEED}_ep${TRAIN_EPOCHS}_pat${PATIENCE}_blr${LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_rlr${RAMP_LEARNING_RATE}_rpt0_ramp1"
sdwpf_log_init "ramp_pilot" "${parameters}" "launch.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
result_dir="${SDWPF_LOG_DIR}"
printf '%s\n' "${result_dir}" > "${POINTER}"
sdwpf_log_capture
echo "[RAMP_PILOT] trend_cv=${TREND_CV_DIR} fold=${FOLD} seed=${SEED}"
echo "[RAMP_PILOT] pretrain_run_id=${PRETRAIN_RUN_ID} test_split=sealed"

PROMPT_ROUTER=trend ROBUST_PITCH=0 RAMP_RESIDUAL=1 \
FOLD="${FOLD}" SEED="${SEED}" N_FOLDS=3 SPLIT=rolling_holdout \
PRED_LEN=12 EVAL_STRIDE=12 PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" \
LEARNING_RATE="${LEARNING_RATE}" NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE}" \
RAMP_LEARNING_RATE="${RAMP_LEARNING_RATE}" \
RUN_ID="${RUN_ID}_finetune" \
SDWPF_LOG_DIR="${result_dir}/runs/f${FOLD}_s${SEED}" SDWPF_LOG_FILE= \
bash scripts/finetune/SDWPF_ablation_prompt.sh

{
    echo "TREND_CV_DIR=${TREND_CV_DIR}"
    echo "FOLD=${FOLD} SEED=${SEED} PRED_LEN=12"
    echo "OLD_TREND_BASELINE_FROM=${TREND_CV_DIR}/cv_metrics.csv"
    awk -F, -v fold="${FOLD}" -v seed="${SEED}" \
        'NR==1 || ($1==fold && $2==seed)' "${TREND_CV_DIR}/cv_metrics.csv"
    grep -E '^Epoch:|^Early stopping|^\[AUDIT\] FINETUNE_CHECKPOINT=' \
        "${result_dir}/runs/f${FOLD}_s${SEED}/finetune.log" || true
} > "${result_dir}/summary.txt"
echo "[RAMP_PILOT] Complete: ${result_dir}/summary.txt"
