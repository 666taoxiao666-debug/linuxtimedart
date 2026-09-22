#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_wiki.sh"

# Ablation: PromptTimeDART with or without a pre-trained checkpoint.
# Do not treat this as the main result unless it beats random-init TimeDART
# on 0-4 h MAE and RMSE across seeds.

SEED="${SEED:-2024}"
FOLD="${FOLD:-0}"
SPLIT="${SPLIT:-rolling_holdout}"
N_FOLDS="${N_FOLDS:-3}"
PRED_LEN="${PRED_LEN:-12}"
EVAL_STRIDE="${EVAL_STRIDE:-${PRED_LEN}}"
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-}"
OVERLAY_CHECKPOINT="${OVERLAY_CHECKPOINT:-}"
FREEZE_NON_UTILITY="${FREEZE_NON_UTILITY:-0}"
ALLOW_RANDOM="${ALLOW_RANDOM:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.000001}"
NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.000005}"
PCT_START="${PCT_START:-0.20}"
PATIENCE="${PATIENCE:-2}"
LOSS="${LOSS:-MIXED}"
MIX_MSE_WEIGHT="${MIX_MSE_WEIGHT:-0.2}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-original_mae}"
RATED_POWER="${RATED_POWER:-1500}"
GPU="${GPU:-0}"
MODEL="${MODEL:-PromptTimeDART}"
CHANNEL_PRIOR="${CHANNEL_PRIOR:-1}"
OP_CONTEXT="${OP_CONTEXT:-1}"
REVIN_KEEP_WIND="${REVIN_KEEP_WIND:-1}"
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
UTILITY_WIKI="${UTILITY_WIKI:-0}"
UTILITY_ADAPTER_MODE="${UTILITY_ADAPTER_MODE:-legacy}"
UTILITY_CALIBRATION_FRACTION="${UTILITY_CALIBRATION_FRACTION:-0.2}"
UTILITY_EVENT_MAX_SCALE="${UTILITY_EVENT_MAX_SCALE:-0.5}"
UTILITY_COMPOSITION_MAX_SCALE="${UTILITY_COMPOSITION_MAX_SCALE:-0.25}"
UTILITY_LOSS_WEIGHT="${UTILITY_LOSS_WEIGHT:-0.1}"
UTILITY_DECISION_LOSS_WEIGHT="${UTILITY_DECISION_LOSS_WEIGHT:-0.2}"
UTILITY_CANDIDATE_LOSS_WEIGHT="${UTILITY_CANDIDATE_LOSS_WEIGHT:-0.1}"
UTILITY_RANKING_LOSS_WEIGHT="${UTILITY_RANKING_LOSS_WEIGHT:-0.1}"
UTILITY_RANKING_MARGIN="${UTILITY_RANKING_MARGIN:-0.01}"
UTILITY_ADAPTER_WARMUP_EPOCHS="${UTILITY_ADAPTER_WARMUP_EPOCHS:-0}"
UTILITY_LEARNING_RATE="${UTILITY_LEARNING_RATE:-0.00003}"
UTILITY_GATE_TEMPERATURE="${UTILITY_GATE_TEMPERATURE:-0.25}"
UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.02}"
UTILITY_TARGET_EPS="${UTILITY_TARGET_EPS:-0.05}"
UTILITY_INTERVENTION_FLOOR="${UTILITY_INTERVENTION_FLOOR:-0.5}"
WIKI_LLM_PATH="${WIKI_LLM_PATH:-Qwen/Qwen2.5-0.5B}"
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-auto}"
REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-0.3333333333}"
RESIDUAL_GATE_INIT="${RESIDUAL_GATE_INIT:--2.2}"
RUN_ID="${RUN_ID:-prompt_h${PRED_LEN}_${SPLIT}_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

sdwpf_wiki_prepare

LOG_PARAMETERS="${MODEL}_h${PRED_LEN}_${SPLIT}_f${FOLD}of${N_FOLDS}_s${SEED}_blr${LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_${LOSS}_mix${MIX_MSE_WEIGHT}_rg${RESIDUAL_GATE_INIT}_ep${TRAIN_EPOCHS}_pat${PATIENCE}_cp${CHANNEL_PRIOR}_ctx${OP_CONTEXT}_rw${REVIN_KEEP_WIND}_rp${REGIME_PROMPT}_${PROMPT_ROUTER}_uw${UTILITY_WIKI}_freeze${FREEZE_NON_UTILITY}_warm${UTILITY_ADAPTER_WARMUP_EPOCHS}_ulr${UTILITY_LEARNING_RATE}_ulw${UTILITY_LOSS_WEIGHT}_udlw${UTILITY_DECISION_LOSS_WEIGHT}_uclw${UTILITY_CANDIDATE_LOSS_WEIGHT}_urlw${UTILITY_RANKING_LOSS_WEIGHT}_ugt${UTILITY_GATE_TEMPERATURE}_umg${UTILITY_MIN_GAIN}_uif${UTILITY_INTERVENTION_FLOOR}_reg${REGIME_LABEL_METHOD}_rq${REGIME_CALIBRATION_QUANTILE}"
LOG_PARAMETERS="adapter${UTILITY_ADAPTER_MODE}_calfrac${UTILITY_CALIBRATION_FRACTION}_ecap${UTILITY_EVENT_MAX_SCALE}_ccap${UTILITY_COMPOSITION_MAX_SCALE}_${LOG_PARAMETERS}"
sdwpf_log_init "finetune" "${LOG_PARAMETERS}" "finetune.log" "${RUN_ID}"
sdwpf_log_install_exit_trap
LOG_ENV_FILE="$(sdwpf_log_sidecar env)"
LOG_SUMMARY_FILE="$(sdwpf_log_sidecar summary.txt)"

{
    echo "TASK=finetune"
    echo "RUN_ID=${RUN_ID}"
    echo "MODEL=${MODEL}"
    echo "PRETRAIN_RUN_ID=${PRETRAIN_RUN_ID}"
    echo "OVERLAY_CHECKPOINT=${OVERLAY_CHECKPOINT}"
    echo "FREEZE_NON_UTILITY=${FREEZE_NON_UTILITY}"
    echo "ALLOW_RANDOM=${ALLOW_RANDOM}"
    echo "SPLIT=${SPLIT}"
    echo "FOLD=${FOLD}"
    echo "N_FOLDS=${N_FOLDS}"
    echo "SEED=${SEED}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "EVAL_STRIDE=${EVAL_STRIDE}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "PCT_START=${PCT_START}"
    echo "PATIENCE=${PATIENCE}"
    echo "LOSS=${LOSS}"
    echo "MIX_MSE_WEIGHT=${MIX_MSE_WEIGHT}"
    echo "EARLY_STOP_METRIC=${EARLY_STOP_METRIC}"
    echo "RESIDUAL_GATE_INIT=${RESIDUAL_GATE_INIT}"
    echo "CHANNEL_PRIOR=${CHANNEL_PRIOR}"
    echo "OP_CONTEXT=${OP_CONTEXT}"
    echo "REVIN_KEEP_WIND=${REVIN_KEEP_WIND}"
    echo "REGIME_PROMPT=${REGIME_PROMPT}"
    echo "PROMPT_ROUTER=${PROMPT_ROUTER}"
    echo "SCENE_WIKI_CONFIG=${SCENE_WIKI_CONFIG}"
    echo "SCENE_WIKI_EMBEDDINGS=${SCENE_WIKI_EMBEDDINGS}"
    echo "SCENE_WIKI_TOP_K=${SCENE_WIKI_TOP_K}"
    echo "SCENE_WIKI_TEMPERATURE=${SCENE_WIKI_TEMPERATURE}"
    echo "SCENE_WIKI_RULE_WEIGHT=${SCENE_WIKI_RULE_WEIGHT}"
    echo "SCENE_WIKI_PROMPT_GATE_INIT=${SCENE_WIKI_PROMPT_GATE_INIT}"
    echo "SCENE_WIKI_ACTIVATION_THRESHOLD=${SCENE_WIKI_ACTIVATION_THRESHOLD}"
    echo "SCENE_WIKI_CONFIDENCE_POWER=${SCENE_WIKI_CONFIDENCE_POWER}"
    echo "UTILITY_WIKI=${UTILITY_WIKI}"
    echo "UTILITY_ADAPTER_MODE=${UTILITY_ADAPTER_MODE}"
    echo "UTILITY_CALIBRATION_FRACTION=${UTILITY_CALIBRATION_FRACTION}"
    echo "UTILITY_EVENT_MAX_SCALE=${UTILITY_EVENT_MAX_SCALE}"
    echo "UTILITY_COMPOSITION_MAX_SCALE=${UTILITY_COMPOSITION_MAX_SCALE}"
    echo "UTILITY_LOSS_WEIGHT=${UTILITY_LOSS_WEIGHT}"
    echo "UTILITY_DECISION_LOSS_WEIGHT=${UTILITY_DECISION_LOSS_WEIGHT}"
    echo "UTILITY_CANDIDATE_LOSS_WEIGHT=${UTILITY_CANDIDATE_LOSS_WEIGHT}"
    echo "UTILITY_RANKING_LOSS_WEIGHT=${UTILITY_RANKING_LOSS_WEIGHT}"
    echo "UTILITY_RANKING_MARGIN=${UTILITY_RANKING_MARGIN}"
    echo "UTILITY_ADAPTER_WARMUP_EPOCHS=${UTILITY_ADAPTER_WARMUP_EPOCHS}"
    echo "UTILITY_LEARNING_RATE=${UTILITY_LEARNING_RATE}"
    echo "UTILITY_GATE_TEMPERATURE=${UTILITY_GATE_TEMPERATURE}"
    echo "UTILITY_MIN_GAIN=${UTILITY_MIN_GAIN}"
    echo "UTILITY_TARGET_EPS=${UTILITY_TARGET_EPS}"
    echo "UTILITY_INTERVENTION_FLOOR=${UTILITY_INTERVENTION_FLOOR}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD}"
    echo "REGIME_CALIBRATION_QUANTILE=${REGIME_CALIBRATION_QUANTILE}"
    echo "RATED_POWER=${RATED_POWER}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${LOG_ENV_FILE}"

EXTRA=()
if [[ "${ALLOW_RANDOM}" == "1" ]]; then
    EXTRA+=(--pretrain_init none --allow_random_init)
else
    EXTRA+=(--pretrain_init auto)
fi
if [[ -n "${OVERLAY_CHECKPOINT}" ]]; then
    [[ -f "${OVERLAY_CHECKPOINT}" ]] || {
        echo "OVERLAY_CHECKPOINT does not exist: ${OVERLAY_CHECKPOINT}" >&2
        exit 2
    }
    EXTRA+=(--overlay_checkpoint "${OVERLAY_CHECKPOINT}")
fi
if [[ "${FREEZE_NON_UTILITY}" == "1" ]]; then
    EXTRA+=(--freeze_non_utility)
fi
if [[ "${CHANNEL_PRIOR}" == "1" ]]; then
    EXTRA+=(--channel_prior)
else
    EXTRA+=(--no-channel_prior)
fi
if [[ "${OP_CONTEXT}" == "1" ]]; then
    EXTRA+=(--op_context)
else
    EXTRA+=(--no-op_context)
fi
if [[ "${REVIN_KEEP_WIND}" == "1" ]]; then
    EXTRA+=(--revin_keep_wind)
else
    EXTRA+=(--no-revin_keep_wind)
fi
if [[ "${REGIME_PROMPT}" == "0" ]]; then
    EXTRA+=(--disable_regime_prompt)
fi
if [[ "${UTILITY_WIKI}" == "1" ]]; then
    EXTRA+=(
        --utility_wiki
        --utility_adapter_mode "${UTILITY_ADAPTER_MODE}"
        --utility_calibration_fraction "${UTILITY_CALIBRATION_FRACTION}"
        --utility_event_max_scale "${UTILITY_EVENT_MAX_SCALE}"
        --utility_composition_max_scale "${UTILITY_COMPOSITION_MAX_SCALE}"
        --utility_loss_weight "${UTILITY_LOSS_WEIGHT}"
        --utility_decision_loss_weight "${UTILITY_DECISION_LOSS_WEIGHT}"
        --utility_candidate_loss_weight "${UTILITY_CANDIDATE_LOSS_WEIGHT}"
        --utility_ranking_loss_weight "${UTILITY_RANKING_LOSS_WEIGHT}"
        --utility_ranking_margin "${UTILITY_RANKING_MARGIN}"
        --utility_adapter_warmup_epochs "${UTILITY_ADAPTER_WARMUP_EPOCHS}"
        --utility_learning_rate "${UTILITY_LEARNING_RATE}"
        --utility_gate_temperature "${UTILITY_GATE_TEMPERATURE}"
        --utility_min_gain "${UTILITY_MIN_GAIN}"
        --utility_target_eps "${UTILITY_TARGET_EPS}"
        --utility_intervention_floor "${UTILITY_INTERVENTION_FLOOR}"
    )
fi

COMMAND=(python -u run.py
    --task_name finetune \
    --downstream_task forecast \
    --is_training 1 \
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
    --batch_size 32 \
    --eval_batch_size 128 \
    --num_workers 0 \
    --sdwpf_train_stride 6 \
    --sdwpf_eval_stride "${EVAL_STRIDE}" \
    --sdwpf_split "${SPLIT}" \
    --sdwpf_fold "${FOLD}" \
    --sdwpf_n_folds "${N_FOLDS}" \
    --pretrain_run_id "${PRETRAIN_RUN_ID}" \
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
    --mix_channels \
    --train_epochs "${TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --new_module_learning_rate "${NEW_MODULE_LEARNING_RATE}" \
    --loss "${LOSS}" \
    --mix_mse_weight "${MIX_MSE_WEIGHT}" \
    --early_stop_metric "${EARLY_STOP_METRIC}" \
    --validate_before_training \
    --residual_forecast \
    --zero_init_residual_head \
    --residual_gate_init "${RESIDUAL_GATE_INIT}" \
    --patience "${PATIENCE}" \
    --pct_start "${PCT_START}" \
    --rated_power "${RATED_POWER}" \
    --lradj step \
    --seed "${SEED}" \
    --run_id "${RUN_ID}" \
    --gpu "${GPU}" \
    "${EXTRA[@]}")

"${COMMAND[@]}" 2>&1 | tee "${SDWPF_LOG_FILE}"

echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${LOG_ENV_FILE}"
{
    grep -E '^Transferred |^Optimizer groups:|^Epoch:|^Early stopping|^\[AUDIT\] FINETUNE_CHECKPOINT=|^\[INFO\] Test evaluation' \
        "${SDWPF_LOG_FILE}" || true
} > "${LOG_SUMMARY_FILE}"
echo "[FINETUNE] Log: ${SDWPF_LOG_FILE}"
echo "[FINETUNE] Summary: ${LOG_SUMMARY_FILE}"
