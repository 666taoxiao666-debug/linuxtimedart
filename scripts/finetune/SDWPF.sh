#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

# Main SDWPF protocol after the leakage review:
# - TimeDART (no Prompt) from random init; pretrain/Prompt are ablations
# - Channel mixer so wind speed reaches the power head
# - Physics features (yaw, drop Prtv/Itmp), Wspd bypasses instance norm,
#   mixer prior + operating-point context (last/mean wind)
# - Primary SCADA-only horizon: 0-4 h (24 ten-minute steps)
# - Non-overlapping evaluation (one window per forecast horizon)
# - Rolling-origin fold 0 by default
# - Rated power 1500 kW, clipped labels
#
# Example:
#   SEED=2024 FOLD=0 bash scripts/finetune/SDWPF.sh
#   for SEED in 2024 2025 2026; do SEED=$SEED bash scripts/finetune/SDWPF.sh; done

SEED="${SEED:-2024}"
LOSS="${LOSS:-MIXED}"
LEARNING_RATE="${LEARNING_RATE:-0.00003}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.8}"
HORIZON_WEIGHT_END="${HORIZON_WEIGHT_END:-1.0}"
POWER_WEIGHT_ALPHA="${POWER_WEIGHT_ALPHA:-0.0}"
RESIDUAL_FORECAST="${RESIDUAL_FORECAST:-1}"
PATIENCE="${PATIENCE:-4}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling_holdout}"
PRED_LEN="${PRED_LEN:-12}"
EVAL_STRIDE="${EVAL_STRIDE:-${PRED_LEN}}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.0001}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
RESIDUAL_GATE_INIT="${RESIDUAL_GATE_INIT:--2.2}"
RUN_ID="${RUN_ID:-mix_h${PRED_LEN}_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

if [[ "${RESIDUAL_FORECAST}" == "1" ]]; then
    RESIDUAL_ARGS=(--residual_forecast --zero_init_residual_head)
else
    RESIDUAL_ARGS=(--no-residual_forecast)
fi

LOG_PARAMETERS="TimeDART_random_h${PRED_LEN}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_lr${LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_${LOSS}_mix${MIX_MSE_WEIGHT}_rg${RESIDUAL_GATE_INIT}_ep${TRAIN_EPOCHS}_pat${PATIENCE}"
sdwpf_log_init "finetune" "${LOG_PARAMETERS}" "finetune.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_ENV_FILE="$(sdwpf_log_sidecar env)"
LOG_SUMMARY_FILE="$(sdwpf_log_sidecar summary.txt)"

{
    echo "TASK=finetune"
    echo "RUN_ID=${RUN_ID}"
    echo "MODEL=TimeDART"
    echo "INITIALIZATION=random"
    echo "SPLIT=${SPLIT}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SEED=${SEED}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "EVAL_STRIDE=${EVAL_STRIDE}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "LOSS=${LOSS}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "HORIZON_WEIGHT_END=${HORIZON_WEIGHT_END}"
    echo "POWER_WEIGHT_ALPHA=${POWER_WEIGHT_ALPHA}"
    echo "RESIDUAL_FORECAST=${RESIDUAL_FORECAST}"
    echo "RESIDUAL_GATE_INIT=${RESIDUAL_GATE_INIT}"
    echo "PATIENCE=${PATIENCE}"
    echo "RATED_POWER=${RATED_POWER}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_ENV_FILE}"

python -u run.py \
    --task_name finetune \
    --downstream_task forecast \
    --is_training 1 \
    --pretrain_init none \
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
    --mix_channels \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --new_module_learning_rate "${NEW_MODULE_LEARNING_RATE}" \
    --loss "${LOSS}" \
    --huber_delta 1.0 \
    --mix_mse_weight "${MIX_MSE_WEIGHT}" \
    --horizon_weight_end "${HORIZON_WEIGHT_END}" \
    --power_weight_alpha "${POWER_WEIGHT_ALPHA}" \
    --early_stop_metric original_mae \
    "${RESIDUAL_ARGS[@]}" \
    --residual_gate_init "${RESIDUAL_GATE_INIT}" \
    --weight_decay 0.0001 \
    --grad_clip 1.0 \
    --head_dropout 0.1 \
    --patience "${PATIENCE}" \
    --pct_start 0.1 \
    --rated_power "${RATED_POWER}" \
    --lradj step \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu "${GPU}" 2>&1 | tee "${SDWPF_LOG_FILE}"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_ENV_FILE}"
{
    grep -E '^Optimizer groups:|^Epoch:|^Early stopping|^\[AUDIT\] FINETUNE_CHECKPOINT=|^\[INFO\] Test evaluation' \
        "${SDWPF_LOG_FILE}" || true
} > "${LOG_SUMMARY_FILE}"
echo "[FINETUNE] Log: ${SDWPF_LOG_FILE}"
echo "[FINETUNE] Summary: ${LOG_SUMMARY_FILE}"
