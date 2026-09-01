#!/usr/bin/env bash
set -euo pipefail

# Ablation: PromptTimeDART with or without a pre-trained checkpoint.
# Do not treat this as the main result unless it beats random-init TimeDART
# on 0-4 h MAE and RMSE across seeds.

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-rolling}"
N_FOLDS="${N_FOLDS:-3}"
PRED_LEN="${PRED_LEN:-24}"
EVAL_STRIDE="${EVAL_STRIDE:-${PRED_LEN}}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-}"
ALLOW_RANDOM="${ALLOW_RANDOM:-0}"
RUN_ID="${RUN_ID:-prompt_h${PRED_LEN}_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"

EXTRA=()
if [[ "${ALLOW_RANDOM}" == "1" ]]; then
    EXTRA+=(--pretrain_init none --allow_random_init)
else
    EXTRA+=(--pretrain_init auto)
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
