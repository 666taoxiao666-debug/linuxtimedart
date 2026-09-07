#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

# Ablation: PromptTimeDART with or without a pre-trained checkpoint.
# Do not treat this as the main result unless it beats random-init TimeDART
# on 0-4 h MAE and RMSE across seeds.

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-rolling_holdout}"
N_FOLDS="${N_FOLDS:-3}"
PRED_LEN="${PRED_LEN:-12}"
EVAL_STRIDE="${EVAL_STRIDE:-${PRED_LEN}}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-}"
ALLOW_RANDOM="${ALLOW_RANDOM:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.000001}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.000005}"
PCT_START="${PCT_START:-0.20}"
PATIENCE="${PATIENCE:-2}"
LOSS="${LOSS:-MIXED}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.2}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-original_mae}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
MODEL="${MODEL:-PromptTimeDART}"
CHANNEL_PRIOR="${CHANNEL_PRIOR:-1}"
OP_CONTEXT="${OP_CONTEXT:-1}"
REVIN_KEEP_WIND="${REVIN_KEEP_WIND:-1}"
REGIME_PROMPT="${REGIME_PROMPT:-1}"
REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-trend_quantile}"
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}"
RESIDUAL_GATE_INIT="${RESIDUAL_GATE_INIT:--2.2}"
RUN_ID="${RUN_ID:-prompt_h${PRED_LEN}_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

LOG_PARAMETERS="${MODEL}_h${PRED_LEN}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_blr${LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_${LOSS}_mix${MIX_MSE_WEIGHT}_rg${RESIDUAL_GATE_INIT}_ep${TRAIN_EPOCHS}_pat${PATIENCE}_cp${CHANNEL_PRIOR}_ctx${OP_CONTEXT}_rw${REVIN_KEEP_WIND}_rp${REGIME_PROMPT}_reg${REGIME_LABEL_METHOD}_rq${REGIME_CALIBRATION_QUANTILE}"
sdwpf_log_init "finetune" "${LOG_PARAMETERS}" "finetune.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_ENV_FILE="$(sdwpf_log_sidecar env)"
LOG_SUMMARY_FILE="$(sdwpf_log_sidecar summary.txt)"

{
    echo "TASK=finetune"
    echo "RUN_ID=${RUN_ID}"
    echo "MODEL=${MODEL}"
    echo "PRETRAIN_RUN_ID=${PRETRAIN_RUN_ID}"
    echo "ALLOW_RANDOM=${ALLOW_RANDOM}"
    echo "SPLIT=${SPLIT}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SEED=${SEED}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "EVAL_STRIDE=${EVAL_STRIDE}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "PCT_START=${PCT_START}"
    echo "PATIENCE=${PATIENCE}"
    echo "LOSS=${LOSS}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "EARLY_STOP_METRIC=${EARLY_STOP_METRIC}"
    echo "RESIDUAL_GATE_INIT=${RESIDUAL_GATE_INIT}"
    echo "CHANNEL_PRIOR=${CHANNEL_PRIOR}"
    echo "OP_CONTEXT=${OP_CONTEXT}"
    echo "REVIN_KEEP_WIND=${REVIN_KEEP_WIND}"
    echo "REGIME_PROMPT=${REGIME_PROMPT}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD}"
    echo "REGIME_CALIBRATION_QUANTILE=${REGIME_CALIBRATION_QUANTILE}"
    echo "RATED_POWER=${RATED_POWER}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_ENV_FILE}"

EXTRA=()
if [[ "${ALLOW_RANDOM}" == "1" ]]; then
    EXTRA+=(--pretrain_init none --allow_random_init)
else
    EXTRA+=(--pretrain_init auto)
fi
if [[ "${CHANNEL_PRIOR}" == "1" ]]; then
    EXTRA+=(--channel_prior)
else
    EXTRA+=(--no-channel_prior)
fi
if [[ "${OP_CONTEXT}" == "1" ]]; then
    EXTRA+=(--op_context)
else
    EXTRA+=(--no-op_context)
fi
if [[ "${REVIN_KEEP_WIND}" == "1" ]]; then
    EXTRA+=(--revin_keep_wind)
else
    EXTRA+=(--no-revin_keep_wind)
fi
if [[ "${REGIME_PROMPT}" == "0" ]]; then
    EXTRA+=(--disable_regime_prompt)
fi

COMMAND=(python -u run.py
    --task_name finetune \
    --downstream_task forecast \
    --is_training 1 \
    --root_path ./datasets/ \
    --data_path sdwpf_fixed.csv \
    --model_id SDWPF \
    --model "${MODEL}" \
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
    --d_layers 1 \
    --patch_len 12 \
    --stride 12 \
    --batch_size 32 \
    --eval_batch_size 128 \
    --num_workers 0 \
    --sdwpf_train_stride 6 \
    --sdwpf_eval_stride "${EVAL_STRIDE}" \
    --sdwpf_split "${SPLIT}" \
    --sdwpf_fold "${FOLD}" \
    --sdwpf_n_folds "${N_FOLDS}" \
    --pretrain_run_id "${PRETRAIN_RUN_ID}" \
    --regime_label_method "${REGIME_LABEL_METHOD}" \
    --regime_calibration_quantile "${REGIME_CALIBRATION_QUANTILE}" \
    --mix_channels \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --new_module_learning_rate "${NEW_MODULE_LEARNING_RATE}" \
    --loss "${LOSS}" \
    --mix_mse_weight "${MIX_MSE_WEIGHT}" \
    --early_stop_metric "${EARLY_STOP_METRIC}" \
    --validate_before_training \
    --residual_forecast \
    --zero_init_residual_head \
    --residual_gate_init "${RESIDUAL_GATE_INIT}" \
    --patience "${PATIENCE}" \
    --pct_start "${PCT_START}" \
    --rated_power "${RATED_POWER}" \
    --lradj step \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu "${GPU}" \
    "${EXTRA[@]}")

"${COMMAND[@]}" 2>&1 | tee "${SDWPF_LOG_FILE}"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_ENV_FILE}"
{
    grep -E '^Transferred |^Optimizer groups:|^Epoch:|^Early stopping|^\[AUDIT\] FINETUNE_CHECKPOINT=|^\[INFO\] Test evaluation' \
        "${SDWPF_LOG_FILE}" || true
} > "${LOG_SUMMARY_FILE}"
echo "[FINETUNE] Log: ${SDWPF_LOG_FILE}"
echo "[FINETUNE] Summary: ${LOG_SUMMARY_FILE}"
