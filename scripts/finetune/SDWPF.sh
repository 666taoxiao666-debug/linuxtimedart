#!/usr/bin/env bash
set -euo pipefail

# Defaults form the clean residual-vs-absolute ablation. Override from shell,
# e.g. LOSS=MIXED HORIZON_WEIGHT_END=1.5 POWER_WEIGHT_ALPHA=0.5 bash ...
SEED="${SEED:-2024}"
LOSS="${LOSS:-MSE}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.8}"
HORIZON_WEIGHT_END="${HORIZON_WEIGHT_END:-1.0}"
POWER_WEIGHT_ALPHA="${POWER_WEIGHT_ALPHA:-0.0}"
RESIDUAL_FORECAST="${RESIDUAL_FORECAST:-1}"

if [[ "${RESIDUAL_FORECAST}" == "1" ]]; then
    RESIDUAL_ARGS=(--residual_forecast --zero_init_residual_head)
else
    RESIDUAL_ARGS=(--no-residual_forecast)
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
    --enc_in 10 \
    --dec_in 10 \
    --c_out 10 \
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
    --sdwpf_eval_stride 6 \
    --train_epochs 20 \
    --learning_rate "${LEARNING_RATE}" \
    --loss "${LOSS}" \
    --huber_delta 1.0 \
    --mix_mse_weight "${MIX_MSE_WEIGHT}" \
    --horizon_weight_end "${HORIZON_WEIGHT_END}" \
    --power_weight_alpha "${POWER_WEIGHT_ALPHA}" \
    --early_stop_metric mse \
    "${RESIDUAL_ARGS[@]}" \
    --weight_decay 0.0001 \
    --grad_clip 1.0 \
    --head_dropout 0.1 \
    --patience 4 \
    --pct_start 0.1 \
    --lradj step \
    --seed "${SEED}" \
    --gpu 0