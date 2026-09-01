#!/usr/bin/env bash
set -euo pipefail

# Deliberately separate from training.  This guard makes repeated test-set
# inspection an explicit action rather than an automatic side effect.
if [[ "${CONFIRM_FINAL_EVAL:-0}" != "1" ]]; then
    echo "Refusing to read the test split. Set CONFIRM_FINAL_EVAL=1 for the final evaluation." >&2
    exit 2
fi
if [[ -z "${FINETUNE_CHECKPOINT:-}" ]]; then
    echo "FINETUNE_CHECKPOINT must point to the selected validation checkpoint." >&2
    exit 2
fi

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-3}"
SPLIT="${SPLIT:-rolling}"
PRED_LEN="${PRED_LEN:-24}"
EVAL_STRIDE="${EVAL_STRIDE:-${PRED_LEN}}"
RESIDUAL_GATE_INIT="${RESIDUAL_GATE_INIT:--4.0}"
MODEL="${MODEL:-PromptTimeDART}"
RUN_ID="${RUN_ID:-final_h${PRED_LEN}_${SPLIT}_f${FOLD}_s${SEED}}"

python -u run.py \
    --task_name finetune \
    --downstream_task forecast \
    --is_training 0 \
    --finetune_checkpoint "${FINETUNE_CHECKPOINT}" \
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
    --eval_batch_size 128 \
    --num_workers 0 \
    --sdwpf_eval_stride "${EVAL_STRIDE}" \
    --sdwpf_split "${SPLIT}" \
    --sdwpf_fold "${FOLD}" \
    --sdwpf_n_folds "${N_FOLDS}" \
    --mix_channels \
    --residual_forecast \
    --no-zero_init_residual_head \
    --residual_gate_init "${RESIDUAL_GATE_INIT}" \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu 0
