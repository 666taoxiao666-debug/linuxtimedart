#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

MODEL="${MODEL:?MODEL must be PatchTST or DLinear}"
case "${MODEL}" in
    PatchTST|DLinear) ;;
    *) echo "MODEL must be PatchTST or DLinear" >&2; exit 2 ;;
esac
SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS=3
SPLIT=rolling_holdout
PRED_LEN=12
INPUT_LEN=336
EVAL_STRIDE=12
TRAIN_EPOCHS="${TRAIN_EPOCHS:?TRAIN_EPOCHS is frozen by the caller}"
PATIENCE="${PATIENCE:?PATIENCE is frozen by the caller}"
LEARNING_RATE="${LEARNING_RATE:?LEARNING_RATE is frozen by the caller}"
LOSS="${LOSS:-MSE}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-frozen_${MODEL}_h12_f${FOLD}_s${SEED}}"
LOG_PARAMETERS="${MODEL}_h12_rolling_holdout_f${FOLD}of3_s${SEED}_lr${LEARNING_RATE}_ep${TRAIN_EPOCHS}_pat${PATIENCE}_${LOSS}"
sdwpf_log_init "strong_baseline" "${LOG_PARAMETERS}" "train.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_DIR="${SDWPF_LOG_DIR}"
ENV_FILE="${LOG_DIR}/baseline.env"

{
    echo "TASK=strong_baseline"
    echo "PROTOCOL_ID=${PROTOCOL_ID:?PROTOCOL_ID is required}"
    echo "PROTOCOL_SHA256=${PROTOCOL_SHA256:?PROTOCOL_SHA256 is required}"
    echo "MODEL=${MODEL}"
    echo "SPLIT=${SPLIT}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SEED=${SEED}"
    echo "INPUT_LEN=${INPUT_LEN}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "EVAL_STRIDE=${EVAL_STRIDE}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "PATIENCE=${PATIENCE}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    echo "LOSS=${LOSS}"
    echo "SEALED_TEST_ACCESSED=0"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${ENV_FILE}"

COMMON=(
    --task_name finetune --downstream_task forecast --is_training 1
    --pretrain_init none --allow_random_init
    --root_path ./datasets/ --data_path sdwpf_fixed.csv
    --model_id SDWPF --model "${MODEL}" --data SDWPF
    --features MS --target power --freq 10min
    --input_len "${INPUT_LEN}" --pred_len "${PRED_LEN}"
    --d_model 128 --d_ff 512 --n_heads 8 --e_layers 2 --d_layers 1
    --patch_len 12 --stride 12 --moving_avg 25 --individual 0
    --batch_size 32 --eval_batch_size 128 --num_workers 0
    --sdwpf_train_stride 6 --sdwpf_eval_stride "${EVAL_STRIDE}"
    --sdwpf_split "${SPLIT}" --sdwpf_fold "${FOLD}" --sdwpf_n_folds "${N_FOLDS}"
    --no-mix_channels --no-residual_forecast
    --train_epochs "${TRAIN_EPOCHS}" --learning_rate "${LEARNING_RATE}"
    --new_module_learning_rate 0 --loss "${LOSS}" --early_stop_metric original_mae
    --weight_decay 0.0001 --grad_clip 1.0 --head_dropout 0.1
    --patience "${PATIENCE}" --pct_start 0.1 --rated_power 1500
    --lradj step --seed "${SEED}" --run_id "${RUN_ID}" --gpu "${GPU}"
)

python -u run.py "${COMMON[@]}" 2>&1 | tee "${LOG_DIR}/train.log"
CHECKPOINT="$(grep -F '[AUDIT] FINETUNE_CHECKPOINT=' "${LOG_DIR}/train.log" | tail -n 1 | sed 's/^.*FINETUNE_CHECKPOINT=//' || true)"
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "Training finished without a valid selected checkpoint" >&2
    exit 3
fi

REPORT_DIR="${LOG_DIR}/artifacts"
python -u run.py \
    --task_name finetune --downstream_task forecast --is_training 0 \
    --finetune_checkpoint "${CHECKPOINT}" \
    --root_path ./datasets/ --data_path sdwpf_fixed.csv \
    --model_id SDWPF --model "${MODEL}" --data SDWPF \
    --features MS --target power --freq 10min \
    --input_len "${INPUT_LEN}" --pred_len "${PRED_LEN}" \
    --d_model 128 --d_ff 512 --n_heads 8 --e_layers 2 --d_layers 1 \
    --patch_len 12 --stride 12 --moving_avg 25 --individual 0 \
    --eval_batch_size 128 --num_workers 0 \
    --sdwpf_eval_stride "${EVAL_STRIDE}" --sdwpf_split "${SPLIT}" \
    --sdwpf_fold "${FOLD}" --sdwpf_n_folds "${N_FOLDS}" \
    --no-mix_channels --no-residual_forecast --rated_power 1500 \
    --seed "${SEED}" --run_id "${RUN_ID}_report" --gpu "${GPU}" \
    --report_split val --report_output_dir "${REPORT_DIR}" 2>&1 | tee "${LOG_DIR}/report.log"

{
    echo "FINETUNE_CHECKPOINT=${CHECKPOINT}"
    echo "REPORT_DIR=${REPORT_DIR}"
    echo "COMPLETED_AT=$(date --iso-8601=seconds)"
} >> "${ENV_FILE}"
grep -E '^Epoch:|^Early stopping|^336->12|^Detailed val forecast report:' "${LOG_DIR}/train.log" "${LOG_DIR}/report.log" > "${LOG_DIR}/summary.txt" || true
echo "[BASELINE] Completed ${MODEL} fold=${FOLD} seed=${SEED}"
echo "[BASELINE] Results: ${LOG_DIR}"
