#!/usr/bin/env bash
set -euo pipefail

# Ablation: PromptTimeDART with or without a pre-trained checkpoint.
# Do not treat this as the main result unless it beats random-init TimeDART
# on 0-4 h MAE and RMSE across seeds.

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-rolling}"
EVAL_STRIDE="${EVAL_STRIDE:-96}"
ALLOW_RANDOM="${ALLOW_RANDOM:-0}"
RUN_ID="${RUN_ID:-prompt_ablation_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

EXTRA=()
if [[ "${ALLOW_RANDOM}" == "1" ]]; then
    EXTRA+=(--allow_random_init)
fi

python -u run.py \
    --task_name finetune \
    --downstream_task forecast \
    --is_training 1 \
    --root_path ./datasets/ \
    --data_path sdwpf_fixed.csv \
    --model_id SDWPF \
    --model PromptTimeDART \
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
    --mix_channels \
    --train_epochs 20 \
    --learning_rate 0.00003 \
    --loss MIXED \
    --mix_mse_weight 0.8 \
    --early_stop_metric mae \
    --residual_forecast \
    --no-zero_init_residual_head \
    --patience 4 \
    --lradj step \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu 0 \
    "${EXTRA[@]}"
