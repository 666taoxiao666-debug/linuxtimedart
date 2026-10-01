#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh

POINTER="outputs/logs/SDWPF/wiki_oof_latest.txt"
if [[ "${1:-}" == "--status" ]]; then
    [[ -f "${POINTER}" ]] || { echo "No Wiki OOF build has been launched."; exit 2; }
    RESULT_DIR="$(< "${POINTER}")"
    echo "RESULT_DIR=${RESULT_DIR}"
    [[ ! -f "${RESULT_DIR}/status.env" ]] || cat "${RESULT_DIR}/status.env"
    [[ ! -f "${RESULT_DIR}/oof.env" ]] || cat "${RESULT_DIR}/oof.env"
    exit 0
fi
if [[ -n "${1:-}" ]]; then
    echo "Usage: FOLD=0 SEED=2024 TREND_CV_DIR=... bash $0 [--status]" >&2
    exit 2
fi

FOLD="${FOLD:-0}"
SEED="${SEED:-2024}"
TREND_CV_DIR="${TREND_CV_DIR:-}"
[[ -n "${TREND_CV_DIR}" && -f "${TREND_CV_DIR}/cv.env" ]] || {
    echo "TREND_CV_DIR must be the matched, completed trend CV directory." >&2
    exit 2
}
WIKI_LLM_PATH="${WIKI_LLM_PATH:-outputs/model_cache/Qwen2.5-0.5B}"
SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-configs/wind_event_factor_wiki.json}"
SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-outputs/wiki/wind_event_factor_wiki_qwen.npz}"

sdwpf_log_init "wiki_oof" "f${FOLD}_s${SEED}_forward_train_only" "oof.log" "wiki_oof_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
sdwpf_log_install_exit_trap
RESULT_DIR="${SDWPF_LOG_DIR}"
printf '%s\n' "${RESULT_DIR}" > "${POINTER}"
sdwpf_log_capture

PLAN="${RESULT_DIR}/oof_plan.json"
EVIDENCE="${RESULT_DIR}/train_oof_wiki_candidates.json"
python -u scripts/generate_wind_wiki_oof_evidence.py plan \
    --trend-cv-dir "${TREND_CV_DIR}" --fold "${FOLD}" --seed "${SEED}" \
    --output "${PLAN}"
read -r INNER_TRAIN_RATIO INNER_VAL_RATIO PRED_LEN < <(
    python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["inner_train_ratio"], p["inner_val_ratio"], p["pred_len"])' "${PLAN}"
)
{
    echo "TASK=wiki_oof"
    echo "FOLD=${FOLD}"
    echo "SEED=${SEED}"
    echo "TREND_CV_DIR=${TREND_CV_DIR}"
    echo "INNER_TRAIN_RATIO=${INNER_TRAIN_RATIO}"
    echo "INNER_VAL_RATIO=${INNER_VAL_RATIO}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "PLAN=${PLAN}"
    echo "EVIDENCE=${EVIDENCE}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${RESULT_DIR}/oof.env"

echo "[OOF] Stage 1/3: train the source only before the evidence interval"
SPLIT=time_ratio FOLD="${FOLD}" SEED="${SEED}" PRED_LEN="${PRED_LEN}" \
SDWPF_TRAIN_RATIO="${INNER_TRAIN_RATIO}" SDWPF_VAL_RATIO="${INNER_VAL_RATIO}" \
PROMPT_ROUTER=compositional_wiki UTILITY_WIKI=0 \
SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG}" SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS}" \
WIKI_LLM_PATH="${WIKI_LLM_PATH}" \
PIPELINE_ID="wiki_oof_source_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)" \
SDWPF_LOG_DIR="${RESULT_DIR}/source" SDWPF_LOG_FILE= \
bash scripts/train/SDWPF_full_pipeline.sh

read_value() {
    sed -n "s/^${2}=//p" "${1}" | tail -n 1
}
SOURCE_ENV="${RESULT_DIR}/source/pipeline.env"
PRETRAIN_RUN_ID="$(read_value "${SOURCE_ENV}" PRETRAIN_RUN_ID)"
SOURCE_CHECKPOINT="$(read_value "${SOURCE_ENV}" FINETUNE_CHECKPOINT)"
[[ -n "${PRETRAIN_RUN_ID}" && -f "${SOURCE_CHECKPOINT}" ]] || {
    echo "OOF source pretraining or static checkpoint was not produced." >&2
    exit 3
}

echo "[OOF] Stage 2/3: train event experts on the same earlier interval"
SPLIT=time_ratio FOLD="${FOLD}" SEED="${SEED}" PRED_LEN="${PRED_LEN}" \
SDWPF_TRAIN_RATIO="${INNER_TRAIN_RATIO}" SDWPF_VAL_RATIO="${INNER_VAL_RATIO}" \
PROMPT_ROUTER=compositional_wiki UTILITY_WIKI=1 UTILITY_FACTORIZED=1 \
UTILITY_ADAPTER_MODE=calibrated_evidence FREEZE_NON_UTILITY=1 \
UTILITY_ADAPTER_WARMUP_EPOCHS="${UTILITY_ADAPTER_WARMUP_EPOCHS:-3}" \
UTILITY_INTERVENTION_FLOOR=1 \
UTILITY_LEARNING_RATE="${UTILITY_LEARNING_RATE:-0.0001}" \
UTILITY_HARM_LOSS_WEIGHT="${UTILITY_HARM_LOSS_WEIGHT:-0.5}" \
UTILITY_DECISION_LOSS_WEIGHT="${UTILITY_DECISION_LOSS_WEIGHT:-0.7}" \
UTILITY_RANKING_LOSS_WEIGHT="${UTILITY_RANKING_LOSS_WEIGHT:-0.3}" \
UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.003}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" PATIENCE="${PATIENCE:-7}" \
PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID}" OVERLAY_CHECKPOINT="${SOURCE_CHECKPOINT}" \
SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG}" SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS}" \
WIKI_LLM_PATH="${WIKI_LLM_PATH}" \
RUN_ID="wiki_oof_factorized_f${FOLD}_s${SEED}_$(date +%Y%m%d_%H%M%S)" \
SDWPF_LOG_DIR="${RESULT_DIR}/factorized" \
SDWPF_LOG_FILE="${RESULT_DIR}/factorized/finetune.log" \
bash scripts/finetune/SDWPF_ablation_prompt.sh

BEST_CHECKPOINT="$(grep -F '[AUDIT] FINETUNE_CHECKPOINT=' "${RESULT_DIR}/factorized/finetune.log" | tail -n 1 | sed 's/^.*FINETUNE_CHECKPOINT=//')"
CHECKPOINT="${BEST_CHECKPOINT%/*}/checkpoint_last.pth"
[[ -f "${CHECKPOINT}" ]] || {
    echo "Trained factorized checkpoint_last.pth is missing: ${CHECKPOINT}" >&2
    exit 3
}

echo "[OOF] Stage 3/3: collect only later, unseen original-train targets"
python -u scripts/generate_wind_wiki_oof_evidence.py collect \
    --plan "${PLAN}" --checkpoint "${CHECKPOINT}" \
    --base-config "${SCENE_WIKI_CONFIG}" --output "${EVIDENCE}"
echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}" >> "${RESULT_DIR}/oof.env"
echo "OOF_CHECKPOINT=${CHECKPOINT}" >> "${RESULT_DIR}/oof.env"
echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${RESULT_DIR}/oof.env"
echo "[OOF] EVIDENCE=${EVIDENCE}"
