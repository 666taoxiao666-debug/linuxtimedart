#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling_holdout}"
PRED_LEN="${PRED_LEN:-12}"
EVAL_STRIDE="${EVAL_STRIDE:-${PRED_LEN}}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
BASELINE_MAX_SAMPLES="${BASELINE_MAX_SAMPLES:-500000}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-baselines_h${PRED_LEN}_${EVAL_SPLIT}_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"

if [[ "${EVAL_SPLIT}" != "val" && "${EVAL_SPLIT}" != "test" ]]; then
    echo "EVAL_SPLIT must be val or test." >&2
    exit 2
fi
if [[ "${EVAL_SPLIT}" == "test" && "${CONFIRM_FINAL_EVAL:-0}" != "1" ]]; then
    echo "Test is locked. Use EVAL_SPLIT=val for model selection." >&2
    echo "For the one-time final test only, also set CONFIRM_FINAL_EVAL=1." >&2
    exit 2
fi
if [[ "${EVAL_SPLIT}" == "test" && "${SPLIT}" != "time_ratio" ]]; then
    echo "Final baseline test requires SPLIT=time_ratio; rolling_holdout is CV-only." >&2
    exit 2
fi

LOG_PARAMETERS="h${PRED_LEN}_${EVAL_SPLIT}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_stride${EVAL_STRIDE}_max${BASELINE_MAX_SAMPLES}"
sdwpf_log_init "baseline" "${LOG_PARAMETERS}" "baseline.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_ENV_FILE="$(sdwpf_log_sidecar env)"
LOG_SUMMARY_FILE="$(sdwpf_log_sidecar summary.txt)"

{
    echo "RUN_ID=${RUN_ID}"
    echo "EVAL_SPLIT=${EVAL_SPLIT}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "EVAL_STRIDE=${EVAL_STRIDE}"
    echo "SPLIT=${SPLIT}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SEED=${SEED}"
    echo "BASELINE_MAX_SAMPLES=${BASELINE_MAX_SAMPLES}"
    echo "RATED_POWER=${RATED_POWER}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_ENV_FILE}"

python -u run_baselines.py \
    --task_name finetune \
    --downstream_task forecast \
    --is_training 0 \
    --allow_random_init \
    --root_path ./datasets/ \
    --data_path sdwpf_fixed.csv \
    --model_id SDWPF \
    --model TimeDART \
    --data SDWPF \
    --features MS \
    --target power \
    --freq 10min \
    --input_len 336 \
    --pred_len "${PRED_LEN}" \
    --d_model 128 \
    --d_ff 512 \
    --n_heads 8 \
    --e_layers 2 \
    --patch_len 12 \
    --stride 12 \
    --sdwpf_eval_stride "${EVAL_STRIDE}" \
    --sdwpf_split "${SPLIT}" \
    --sdwpf_fold "${FOLD}" \
    --sdwpf_n_folds "${N_FOLDS}" \
    --eval_split "${EVAL_SPLIT}" \
    --baseline_max_samples "${BASELINE_MAX_SAMPLES}" \
    --rated_power "${RATED_POWER}" \
    --mix_channels \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu "${GPU}" 2>&1 | tee "${SDWPF_LOG_FILE}"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_ENV_FILE}"
{
    grep -E '^Evaluation split:|^Forecast horizon:|^persistence |^daily_persistence |^power_curve |^tree |^Output:' \
        "${SDWPF_LOG_FILE}" || true
} > "${LOG_SUMMARY_FILE}"
echo "[BASELINE] Log: ${SDWPF_LOG_FILE}"
echo "[BASELINE] Summary: ${LOG_SUMMARY_FILE}"
