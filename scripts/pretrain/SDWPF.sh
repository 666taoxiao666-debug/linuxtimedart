#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling_holdout}"
PRED_LEN="${PRED_LEN:-24}"
TRAIN_STRIDE="${TRAIN_STRIDE:-6}"
EVAL_STRIDE="${EVAL_STRIDE:-6}"
PATIENCE="${PATIENCE:-3}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LAMBDA_CE="${LAMBDA_CE:-0.02}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-pretrain_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${PRETRAIN_RUN_ID}}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

COMMAND=(python -u run.py
    --task_name pretrain \
    --downstream_task forecast \
    --root_path ./datasets/ \
    --data_path sdwpf_fixed.csv \
    --model_id SDWPF \
    --model PromptTimeDART \
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
    --sdwpf_train_stride "${TRAIN_STRIDE}" \
    --sdwpf_eval_stride "${EVAL_STRIDE}" \
    --sdwpf_split "${SPLIT}" \
    --sdwpf_fold "${FOLD}" \
    --sdwpf_n_folds "${N_FOLDS}" \
    --pretrain_run_id "${PRETRAIN_RUN_ID}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --learning_rate "${LEARNING_RATE}" \
    --lambda_ce "${LAMBDA_CE}" \
    --rated_power "${RATED_POWER}" \
    --lr_decay 0.95 \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu "${GPU}")

if [[ -n "${SDWPF_LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname "${SDWPF_LOG_FILE}")"
    "${COMMAND[@]}" 2>&1 | tee "${SDWPF_LOG_FILE}"
else
    "${COMMAND[@]}"
fi
