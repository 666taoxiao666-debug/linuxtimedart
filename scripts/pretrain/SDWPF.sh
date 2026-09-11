#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_wiki.sh"

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
LAMBDA_SCENE_CE="${LAMBDA_SCENE_CE:-0.02}"
LAMBDA_EVENT_BCE="${LAMBDA_EVENT_BCE:-0.02}"
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
REGIME_CALIBRATION_SAMPLES="${REGIME_CALIBRATION_SAMPLES:-50000}"
REGIME_MIN_CLASS_FRACTION="${REGIME_MIN_CLASS_FRACTION:-0.05}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-pretrain_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${PRETRAIN_RUN_ID}}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

sdwpf_wiki_prepare

LOG_PARAMETERS="h${PRED_LEN}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_lr${LEARNING_RATE}_lce${LAMBDA_CE}_lebce${LAMBDA_EVENT_BCE}_${PROMPT_ROUTER}_reg${REGIME_LABEL_METHOD}_rq${REGIME_CALIBRATION_QUANTILE}_ep${TRAIN_EPOCHS}_pat${PATIENCE}"
sdwpf_log_init "pretrain" "${LOG_PARAMETERS}" "pretrain.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_ENV_FILE="$(sdwpf_log_sidecar env)"
LOG_SUMMARY_FILE="$(sdwpf_log_sidecar summary.txt)"

{
    echo "TASK=pretrain"
    echo "RUN_ID=${RUN_ID}"
    echo "PRETRAIN_RUN_ID=${PRETRAIN_RUN_ID}"
    echo "SPLIT=${SPLIT}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SEED=${SEED}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "TRAIN_STRIDE=${TRAIN_STRIDE}"
    echo "EVAL_STRIDE=${EVAL_STRIDE}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    echo "LAMBDA_CE=${LAMBDA_CE}"
    echo "LAMBDA_SCENE_CE=${LAMBDA_SCENE_CE}"
    echo "LAMBDA_EVENT_BCE=${LAMBDA_EVENT_BCE}"
    echo "PROMPT_ROUTER=${PROMPT_ROUTER}"
    echo "SCENE_WIKI_CONFIG=${SCENE_WIKI_CONFIG}"
    echo "SCENE_WIKI_EMBEDDINGS=${SCENE_WIKI_EMBEDDINGS}"
    echo "SCENE_WIKI_TOP_K=${SCENE_WIKI_TOP_K}"
    echo "SCENE_WIKI_TEMPERATURE=${SCENE_WIKI_TEMPERATURE}"
    echo "SCENE_WIKI_RULE_WEIGHT=${SCENE_WIKI_RULE_WEIGHT}"
    echo "SCENE_WIKI_PROMPT_GATE_INIT=${SCENE_WIKI_PROMPT_GATE_INIT}"
    echo "SCENE_WIKI_ACTIVATION_THRESHOLD=${SCENE_WIKI_ACTIVATION_THRESHOLD}"
    echo "SCENE_WIKI_CONFIDENCE_POWER=${SCENE_WIKI_CONFIDENCE_POWER}"
    echo "WIKI_LLM_PATH=${WIKI_LLM_PATH}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD}"
    echo "REGIME_CALIBRATION_QUANTILE=${REGIME_CALIBRATION_QUANTILE}"
    echo "REGIME_CALIBRATION_SAMPLES=${REGIME_CALIBRATION_SAMPLES}"
    echo "REGIME_MIN_CLASS_FRACTION=${REGIME_MIN_CLASS_FRACTION}"
    echo "PATIENCE=${PATIENCE}"
    echo "RATED_POWER=${RATED_POWER}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_ENV_FILE}"

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
    --lambda_scene_ce "${LAMBDA_SCENE_CE}" \
    --lambda_event_bce "${LAMBDA_EVENT_BCE}" \
    --prompt_router "${PROMPT_ROUTER}" \
    --scene_wiki_config "${SCENE_WIKI_CONFIG}" \
    --scene_wiki_embeddings "${SCENE_WIKI_EMBEDDINGS}" \
    --scene_wiki_top_k "${SCENE_WIKI_TOP_K}" \
    --scene_wiki_temperature "${SCENE_WIKI_TEMPERATURE}" \
    --scene_wiki_rule_weight "${SCENE_WIKI_RULE_WEIGHT}" \
    --scene_wiki_prompt_gate_init "${SCENE_WIKI_PROMPT_GATE_INIT}" \
    --scene_wiki_activation_threshold "${SCENE_WIKI_ACTIVATION_THRESHOLD}" \
    --scene_wiki_confidence_power "${SCENE_WIKI_CONFIDENCE_POWER}" \
    --regime_label_method "${REGIME_LABEL_METHOD}" \
    --regime_calibration_quantile "${REGIME_CALIBRATION_QUANTILE}" \
    --regime_calibration_samples "${REGIME_CALIBRATION_SAMPLES}" \
    --regime_min_class_fraction "${REGIME_MIN_CLASS_FRACTION}" \
    --rated_power "${RATED_POWER}" \
    --lr_decay 0.95 \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu "${GPU}")

"${COMMAND[@]}" 2>&1 | tee "${SDWPF_LOG_FILE}"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_ENV_FILE}"
{
    grep -E '^Epoch:|^Legacy Wiki |^Event Wiki |^Validation loss decreased|^Pretrain early stopping|^\[AUDIT\] (Run manifest:|REGIME_CALIBRATION=|SCENE_WIKI_CALIBRATION=|EVENT_FACTOR_CALIBRATION=)' \
        "${SDWPF_LOG_FILE}" || true
} > "${LOG_SUMMARY_FILE}"
echo "[PRETRAIN] Log: ${SDWPF_LOG_FILE}"
echo "[PRETRAIN] Summary: ${LOG_SUMMARY_FILE}"
