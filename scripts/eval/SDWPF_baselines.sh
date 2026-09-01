#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-rolling}"
EVAL_STRIDE="${EVAL_STRIDE:-96}"
RUN_ID="${RUN_ID:-baselines_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

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
    --pred_len 96 \
    --d_model 128 \
    --d_ff 512 \
    --n_heads 8 \
    --e_layers 2 \
    --patch_len 12 \
    --stride 12 \
    --sdwpf_eval_stride "${EVAL_STRIDE}" \
    --sdwpf_split "${SPLIT}" \
    --sdwpf_fold "${FOLD}" \
    --mix_channels \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu 0
