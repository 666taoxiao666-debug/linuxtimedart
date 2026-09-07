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
N_FOLDS="${N_FOLDS:-1}"
SPLIT="${SPLIT:-time_ratio}"
PRED_LEN="${PRED_LEN:-12}"
EVAL_STRIDE="${EVAL_STRIDE:-${PRED_LEN}}"
RESIDUAL_GATE_INIT="${RESIDUAL_GATE_INIT:--2.2}"
MODEL="${MODEL:-PromptTimeDART}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-final_h${PRED_LEN}_${SPLIT}_f${FOLD}_s${SEED}}"
REPORT_OUTPUT_DIR="${REPORT_OUTPUT_DIR:-}"
FORECAST_PLOT_POINTS="${FORECAST_PLOT_POINTS:-150}"
FORECAST_PLOT_TURBINE_ID="${FORECAST_PLOT_TURBINE_ID:-}"
FORECAST_PLOT_START="${FORECAST_PLOT_START:-}"
REGIME_PROMPT="${REGIME_PROMPT:-1}"
REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-trend_quantile}"
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ "${SPLIT}" != "time_ratio" ]]; then
    echo "Final paper evaluation requires SPLIT=time_ratio so CV checkpoints cannot read the sealed holdout." >&2
    exit 2
fi

PLOT_ARGS=(--forecast_plot_points "${FORECAST_PLOT_POINTS}")
PLOT_ARGS+=(--regime_label_method "${REGIME_LABEL_METHOD}")
PLOT_ARGS+=(--regime_calibration_quantile "${REGIME_CALIBRATION_QUANTILE}")
if [[ -n "${FORECAST_PLOT_TURBINE_ID}" ]]; then
    PLOT_ARGS+=(--forecast_plot_turbine_id "${FORECAST_PLOT_TURBINE_ID}")
fi
if [[ "${REGIME_PROMPT}" == "0" ]]; then
    PLOT_ARGS+=(--disable_regime_prompt)
fi
if [[ -n "${FORECAST_PLOT_START}" ]]; then
    PLOT_ARGS+=(--forecast_plot_start "${FORECAST_PLOT_START}")
fi

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
    --zero_init_residual_head \
    --residual_gate_init "${RESIDUAL_GATE_INIT}" \
    --rated_power "${RATED_POWER}" \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu "${GPU}" \
    --report_output_dir "${REPORT_OUTPUT_DIR}" \
    "${PLOT_ARGS[@]}"
