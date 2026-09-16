#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_wiki.sh"

if [[ -z "${FINETUNE_CHECKPOINT:-}" ]]; then
    echo "FINETUNE_CHECKPOINT must point to the selected validation checkpoint." >&2
    exit 2
fi

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
N_FOLDS="${N_FOLDS:-1}"
SPLIT="${SPLIT:-time_ratio}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
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
PROMPT_ROUTER="${PROMPT_ROUTER:-compositional_wiki}"
if [[ "${PROMPT_ROUTER}" == "scene_wiki" ]]; then
    REGIME_LABEL_METHOD="scene_wiki"
else
    REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-trend_quantile}"
fi
if [[ "${PROMPT_ROUTER}" == "compositional_wiki" ]]; then
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_event_factor_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_event_factor_wiki_qwen.npz}"
elif [[ "${PROMPT_ROUTER}" == "hybrid_wiki" ]]; then
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_exception_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_exception_wiki_qwen.npz}"
else
    SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_regime_wiki.json}"
    SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_regime_wiki_qwen.npz}"
fi
SCENE_WIKI_TOP_K="${SCENE_WIKI_TOP_K:-2}"
SCENE_WIKI_TEMPERATURE="${SCENE_WIKI_TEMPERATURE:-0.2}"
SCENE_WIKI_RULE_WEIGHT="${SCENE_WIKI_RULE_WEIGHT:-2.0}"
SCENE_WIKI_PROMPT_GATE_INIT="${SCENE_WIKI_PROMPT_GATE_INIT:--2.2}"
SCENE_WIKI_ACTIVATION_THRESHOLD="${SCENE_WIKI_ACTIVATION_THRESHOLD:-0.55}"
SCENE_WIKI_CONFIDENCE_POWER="${SCENE_WIKI_CONFIDENCE_POWER:-1.0}"
WIKI_LLM_PATH="${WIKI_LLM_PATH:-Qwen/Qwen2.5-0.5B}"
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-auto}"
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

sdwpf_wiki_prepare

case "${EVAL_SPLIT}" in
    test)
        if [[ "${CONFIRM_FINAL_EVAL:-0}" != "1" ]]; then
            echo "Refusing to read the test split. Set CONFIRM_FINAL_EVAL=1 only for final evaluation." >&2
            exit 2
        fi
        if [[ "${SPLIT}" != "time_ratio" ]]; then
            echo "Final paper evaluation requires SPLIT=time_ratio so CV checkpoints cannot read the sealed holdout." >&2
            exit 2
        fi
        ;;
    val)
        if [[ "${SPLIT}" != "rolling_holdout" ]]; then
            echo "Interim validation figures require SPLIT=rolling_holdout." >&2
            exit 2
        fi
        ;;
    *)
        echo "EVAL_SPLIT must be val or test." >&2
        exit 2
        ;;
esac

PLOT_ARGS=(--forecast_plot_points "${FORECAST_PLOT_POINTS}")
if [[ "${WIKI_DIAGNOSTIC:-0}" == "1" ]]; then
    PLOT_ARGS+=(--wiki_diagnostic)
    if [[ "${WIKI_DIAGNOSTIC_SINGLE_EVENTS:-0}" == "1" ]]; then
        PLOT_ARGS+=(--wiki_diagnostic_single_events)
    fi
fi
PLOT_ARGS+=(--prompt_router "${PROMPT_ROUTER}")
PLOT_ARGS+=(--scene_wiki_config "${SCENE_WIKI_CONFIG}")
PLOT_ARGS+=(--scene_wiki_embeddings "${SCENE_WIKI_EMBEDDINGS}")
PLOT_ARGS+=(--scene_wiki_top_k "${SCENE_WIKI_TOP_K}")
PLOT_ARGS+=(--scene_wiki_temperature "${SCENE_WIKI_TEMPERATURE}")
PLOT_ARGS+=(--scene_wiki_rule_weight "${SCENE_WIKI_RULE_WEIGHT}")
PLOT_ARGS+=(--scene_wiki_prompt_gate_init "${SCENE_WIKI_PROMPT_GATE_INIT}")
PLOT_ARGS+=(--scene_wiki_activation_threshold "${SCENE_WIKI_ACTIVATION_THRESHOLD}")
PLOT_ARGS+=(--scene_wiki_confidence_power "${SCENE_WIKI_CONFIDENCE_POWER}")
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
    --report_split "${EVAL_SPLIT}" \
    "${PLOT_ARGS[@]}"
