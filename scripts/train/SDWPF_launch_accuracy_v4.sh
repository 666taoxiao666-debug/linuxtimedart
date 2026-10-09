#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh

POINTER="outputs/logs/SDWPF/accuracy_v4_latest.txt"
PROTOCOL_FILE="configs/sdwpf_accuracy_protocol_v4.json"

case "${1:-}" in
    --status)
        [[ -f "${POINTER}" ]] || { echo "No accuracy-v4 run has been launched."; exit 2; }
        directory="$(< "${POINTER}")"
        [[ -n "${directory}" && -d "${directory}" ]] || {
            echo "Accuracy-v4 result directory is missing: ${directory}" >&2
            exit 2
        }
        echo "RESULT_DIR=${directory}"
        [[ ! -f "${directory}/status.env" ]] || cat "${directory}/status.env"
        [[ ! -f "${directory}/accuracy_v4.env" ]] || cat "${directory}/accuracy_v4.env"
        if [[ -f "${directory}/validation/artifacts/metrics.json" ]]; then
            status_python="$(command -v python || command -v python3)"
            "${status_python}" - "${directory}/validation/artifacts/metrics.json" <<'PY'
import json
import sys

values = json.load(open(sys.argv[1], encoding="utf-8"))["original"]
for key in (
    "mae", "rmse", "r2", "persistence_mae", "persistence_rmse",
    "persistence_r2", "mae_skill_vs_persistence_pct",
    "rmse_skill_vs_persistence_pct",
):
    print(f"{key.upper()}={values[key]:.9f}")
PY
        elif [[ -f "${directory}/finetune.log" ]]; then
            tail -n 45 "${directory}/finetune.log"
        elif [[ -f "${directory}/launch.log" ]]; then
            tail -n 45 "${directory}/launch.log"
        fi
        exit 0
        ;;
    --worker)
        _SDWPF_LOG_OWNS_DIR=1
        sdwpf_log_install_exit_trap
        SDWPF_LOG_FILE="${SDWPF_LOG_DIR}/finetune.log" \
        bash scripts/finetune/SDWPF_ablation_prompt.sh

        checkpoint="$(sed -n 's/^\[AUDIT\] FINETUNE_CHECKPOINT=//p' "${SDWPF_LOG_DIR}/finetune.log" | tail -n 1)"
        [[ -n "${checkpoint}" && -f "${checkpoint}" ]] || {
            echo "Could not resolve the selected fine-tuning checkpoint." >&2
            exit 2
        }
        validation_dir="${SDWPF_LOG_DIR}/validation"
        validation_log="${validation_dir}/validation.log"
        mkdir -p "${validation_dir}"
        FINETUNE_CHECKPOINT="${checkpoint}" \
        REPORT_OUTPUT_DIR="${validation_dir}/artifacts" \
        SDWPF_LOG_DIR="${validation_dir}" \
        SDWPF_LOG_FILE="${validation_log}" \
        FORECAST_PLOT_POINTS=300 \
        bash scripts/eval/SDWPF_logged_validation_plot.sh
        exit 0
        ;;
    ""|--foreground) ;;
    *) echo "Usage: bash $0 [--status|--foreground]" >&2; exit 2 ;;
esac

[[ -f "${PROTOCOL_FILE}" ]] || { echo "Missing protocol: ${PROTOCOL_FILE}" >&2; exit 2; }
PROTOCOL_SHA256="$(sha256sum "${PROTOCOL_FILE}" | awk '{print $1}')"
TREND_CV_DIR="${TREND_CV_DIR:-outputs/logs/SDWPF/20260914/001_cv_trend_h12_rolling_holdout_folds0-1-2_seeds2024-2025-202_h36d88ba5e218}"
FOLD="${FOLD:-1}"
SEED="${SEED:-2024}"
PRED_LEN="${PRED_LEN:-12}"
source_env="${TREND_CV_DIR}/runs/f${FOLD}_s${SEED}/finetune.env"
[[ -f "${source_env}" ]] || { echo "Missing paired trend run: ${source_env}" >&2; exit 2; }
PRETRAIN_RUN_ID="$(sed -n 's/^PRETRAIN_RUN_ID=//p' "${source_env}" | head -n 1)"
[[ -n "${PRETRAIN_RUN_ID}" ]] || { echo "Missing PRETRAIN_RUN_ID in ${source_env}" >&2; exit 2; }

export FOLD SEED PRED_LEN PRETRAIN_RUN_ID
export N_FOLDS=3 SPLIT=rolling_holdout
export MODEL=PromptTimeDART PROMPT_ROUTER=trend
export SCENE_WIKI_CONFIG=configs/wind_regime_wiki.json
export SCENE_WIKI_EMBEDDINGS=outputs/wiki/wind_regime_wiki_qwen.npz
export WIKI_LLM_PATH="${WIKI_LLM_PATH:-outputs/model_cache/Qwen2.5-0.5B}"
export WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-cuda:0}"
export ALLOW_RANDOM=0 FREEZE_NON_UTILITY=0 UTILITY_WIKI=0
export CHANNEL_PRIOR=1 OP_CONTEXT=1 REVIN_KEEP_WIND=1 REGIME_PROMPT=1
export TRAIN_EPOCHS=8 PATIENCE=4 PCT_START=0.20
export LEARNING_RATE=0.000001 NEW_MODULE_LEARNING_RATE=0.000005
export LOSS=MIXED MIX_MSE_WEIGHT=0.5
export HORIZON_WEIGHT_END=1.5 POWER_WEIGHT_ALPHA=0.5
export EARLY_STOP_METRIC=original_mae_rmse_ratio
export RESIDUAL_GATE_INIT=-2.2
export RUN_ID="accuracy_v4_h${PRED_LEN}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
export HF_HUB_OFFLINE=1

python -c 'import torch; print("PyTorch:", torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable"'
parameters="h${PRED_LEN}_f${FOLD}_s${SEED}_mix${MIX_MSE_WEIGHT}_hw${HORIZON_WEIGHT_END}_pw${POWER_WEIGHT_ALPHA}_selbalanced_ep${TRAIN_EPOCHS}_pat${PATIENCE}"
sdwpf_log_init accuracy_v4 "${parameters}" launch.log "${RUN_ID}"
{
    echo "TASK=accuracy_v4"
    echo "RUN_ID=${RUN_ID}"
    echo "PROTOCOL_FILE=${PROTOCOL_FILE}"
    echo "PROTOCOL_SHA256=${PROTOCOL_SHA256}"
    echo "TREND_CV_DIR=${TREND_CV_DIR}"
    echo "PRETRAIN_RUN_ID=${PRETRAIN_RUN_ID}"
    echo "FOLD=${FOLD}"
    echo "SEED=${SEED}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "LOSS=${LOSS}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "HORIZON_WEIGHT_END=${HORIZON_WEIGHT_END}"
    echo "POWER_WEIGHT_ALPHA=${POWER_WEIGHT_ALPHA}"
    echo "EARLY_STOP_METRIC=${EARLY_STOP_METRIC}"
    echo "SEALED_TEST_ACCESSED=0"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${SDWPF_LOG_DIR}/accuracy_v4.env"
printf '%s\n' "${SDWPF_LOG_DIR}" > "${POINTER}"

if [[ "${1:-}" == "--foreground" ]]; then
    bash scripts/train/SDWPF_launch_accuracy_v4.sh --worker
else
    nohup bash scripts/train/SDWPF_launch_accuracy_v4.sh --worker \
        > "${SDWPF_LOG_DIR}/launch.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "${SDWPF_LOG_DIR}/launcher.pid"
    echo "PID=$!"
    echo "Check: bash scripts/train/SDWPF_launch_accuracy_v4.sh --status"
fi
