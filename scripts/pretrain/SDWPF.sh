#!/usr/bin/env bash
set -euo pipefail

python -u run.py \
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
    --learning_rate 0.0001 \
    --lr_decay 0.95 \
    --gpu 0
