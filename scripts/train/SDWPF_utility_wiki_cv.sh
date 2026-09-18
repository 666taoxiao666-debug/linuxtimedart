#!/usr/bin/env bash
set -euo pipefail

# Evidence-Conditioned Utility-Adaptive Wiki (ECUA-Wiki).
# With SOURCE_CV_DIR, reuse its fold/seed-matched compositional-Wiki pretraining
# and run fine-tuning only. Without it, run the complete pretrain + fine-tune CV.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

export PROMPT_ROUTER="compositional_wiki"
export UTILITY_WIKI=1
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-8}"
export PATIENCE="${PATIENCE:-3}"
export LEARNING_RATE="${LEARNING_RATE:-0.000001}"
export NEW_MODULE_LEARNING_RATE="${NEW_MODULE_LEARNING_RATE:-0.000005}"
export UTILITY_LOSS_WEIGHT="${UTILITY_LOSS_WEIGHT:-0.2}"
export UTILITY_CANDIDATE_LOSS_WEIGHT="${UTILITY_CANDIDATE_LOSS_WEIGHT:-0.1}"
export UTILITY_RANKING_LOSS_WEIGHT="${UTILITY_RANKING_LOSS_WEIGHT:-0.1}"
export UTILITY_RANKING_MARGIN="${UTILITY_RANKING_MARGIN:-0.01}"
export UTILITY_LEARNING_RATE="${UTILITY_LEARNING_RATE:-0.00003}"
export UTILITY_GATE_TEMPERATURE="${UTILITY_GATE_TEMPERATURE:-0.25}"
export UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.0}"
export UTILITY_TARGET_EPS="${UTILITY_TARGET_EPS:-0.05}"
export UTILITY_INTERVENTION_FLOOR="${UTILITY_INTERVENTION_FLOOR:-0.5}"
export PRED_LEN="${PRED_LEN:-12}"
export FOLDS="${FOLDS:-0}"
export SEEDS="${SEEDS:-2024}"

if [[ -z "${SOURCE_CV_DIR:-}" ]]; then
    echo "[UTILITY-WIKI] SOURCE_CV_DIR is unset; running pretraining + fine-tuning."
    bash scripts/train/SDWPF_paper_cv.sh
    exit 0
fi

read_value() {
    sed -n "s/^${2}=//p" "$1" | tail -n 1
}

SOURCE_ENV="${SOURCE_CV_DIR}/cv.env"
[[ -f "${SOURCE_ENV}" ]] || {
    echo "SOURCE_CV_DIR has no cv.env: ${SOURCE_CV_DIR}" >&2
    exit 2
}
[[ "$(read_value "${SOURCE_ENV}" PROMPT_ROUTER)" == "compositional_wiki" ]] || {
    echo "SOURCE_CV_DIR must be a compositional_wiki CV run." >&2
    exit 2
}
[[ "$(read_value "${SOURCE_ENV}" SPLIT)" == "rolling_holdout" ]] || {
    echo "SOURCE_CV_DIR must use rolling_holdout." >&2
    exit 2
}

export SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_CONFIG)}"
export SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_EMBEDDINGS)}"
export SCENE_WIKI_TOP_K="${SCENE_WIKI_TOP_K:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_TOP_K)}"
export SCENE_WIKI_TEMPERATURE="${SCENE_WIKI_TEMPERATURE:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_TEMPERATURE)}"
export SCENE_WIKI_RULE_WEIGHT="${SCENE_WIKI_RULE_WEIGHT:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_RULE_WEIGHT)}"
export SCENE_WIKI_PROMPT_GATE_INIT="${SCENE_WIKI_PROMPT_GATE_INIT:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_PROMPT_GATE_INIT)}"
export SCENE_WIKI_ACTIVATION_THRESHOLD="${SCENE_WIKI_ACTIVATION_THRESHOLD:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_ACTIVATION_THRESHOLD)}"
export SCENE_WIKI_CONFIDENCE_POWER="${SCENE_WIKI_CONFIDENCE_POWER:-$(read_value "${SOURCE_ENV}" SCENE_WIKI_CONFIDENCE_POWER)}"
export REGIME_LABEL_METHOD="${REGIME_LABEL_METHOD:-$(read_value "${SOURCE_ENV}" REGIME_LABEL_METHOD)}"
export REGIME_CALIBRATION_QUANTILE="${REGIME_CALIBRATION_QUANTILE:-$(read_value "${SOURCE_ENV}" REGIME_CALIBRATION_QUANTILE)}"

for required in SCENE_WIKI_CONFIG SCENE_WIKI_EMBEDDINGS SCENE_WIKI_TOP_K SCENE_WIKI_TEMPERATURE SCENE_WIKI_RULE_WEIGHT SCENE_WIKI_PROMPT_GATE_INIT SCENE_WIKI_ACTIVATION_THRESHOLD SCENE_WIKI_CONFIDENCE_POWER REGIME_LABEL_METHOD REGIME_CALIBRATION_QUANTILE; do
    [[ -n "${!required}" ]] || {
        echo "Missing source parameter: ${required}" >&2
        exit 2
    }
done
[[ -f "${SCENE_WIKI_CONFIG}" && -f "${SCENE_WIKI_EMBEDDINGS}" ]] || {
    echo "Frozen source Wiki config/embedding file is missing." >&2
    exit 2
}

read -r -a CV_FOLDS <<< "${FOLDS}"
read -r -a CV_SEEDS <<< "${SEEDS}"
CV_ID="${CV_ID:-utility_reuse_h${PRED_LEN}_rolling_holdout_${#CV_FOLDS[@]}fold_${#CV_SEEDS[@]}seed_$(date +%Y%m%d_%H%M%S)}"
LOG_PARAMETERS="reuse_h${PRED_LEN}_folds${FOLDS// /-}_seeds${SEEDS// /-}_ep${TRAIN_EPOCHS}_pat${PATIENCE}_blr${LEARNING_RATE}_nlr${NEW_MODULE_LEARNING_RATE}_ulr${UTILITY_LEARNING_RATE}_ulw${UTILITY_LOSS_WEIGHT}_uclw${UTILITY_CANDIDATE_LOSS_WEIGHT}_urlw${UTILITY_RANKING_LOSS_WEIGHT}_ugt${UTILITY_GATE_TEMPERATURE}_umg${UTILITY_MIN_GAIN}_uif${UTILITY_INTERVENTION_FLOOR}"
sdwpf_log_init "cv_utility" "${LOG_PARAMETERS}" "cv.log" "${CV_ID}"
sdwpf_log_install_exit_trap
CV_LOG_DIR="${SDWPF_LOG_DIR}"
sdwpf_log_capture

{
    echo "TASK=cv_utility_finetune_only"
    echo "CV_ID=${CV_ID}"
    echo "SOURCE_CV_DIR=${SOURCE_CV_DIR}"
    echo "PRED_LEN=${PRED_LEN}"
    echo "SPLIT=rolling_holdout"
    echo "FOLDS=${FOLDS}"
    echo "SEEDS=${SEEDS}"
    echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
    echo "PATIENCE=${PATIENCE}"
    echo "LEARNING_RATE=${LEARNING_RATE}"
    echo "NEW_MODULE_LEARNING_RATE=${NEW_MODULE_LEARNING_RATE}"
    echo "PROMPT_ROUTER=compositional_wiki"
    echo "SCENE_WIKI_CONFIG=${SCENE_WIKI_CONFIG}"
    echo "SCENE_WIKI_EMBEDDINGS=${SCENE_WIKI_EMBEDDINGS}"
    echo "SCENE_WIKI_TOP_K=${SCENE_WIKI_TOP_K}"
    echo "SCENE_WIKI_TEMPERATURE=${SCENE_WIKI_TEMPERATURE}"
    echo "SCENE_WIKI_RULE_WEIGHT=${SCENE_WIKI_RULE_WEIGHT}"
    echo "SCENE_WIKI_PROMPT_GATE_INIT=${SCENE_WIKI_PROMPT_GATE_INIT}"
    echo "SCENE_WIKI_ACTIVATION_THRESHOLD=${SCENE_WIKI_ACTIVATION_THRESHOLD}"
    echo "SCENE_WIKI_CONFIDENCE_POWER=${SCENE_WIKI_CONFIDENCE_POWER}"
    echo "REGIME_LABEL_METHOD=${REGIME_LABEL_METHOD}"
    echo "REGIME_CALIBRATION_QUANTILE=${REGIME_CALIBRATION_QUANTILE}"
    echo "UTILITY_WIKI=1"
    echo "UTILITY_LOSS_WEIGHT=${UTILITY_LOSS_WEIGHT}"
    echo "UTILITY_CANDIDATE_LOSS_WEIGHT=${UTILITY_CANDIDATE_LOSS_WEIGHT}"
    echo "UTILITY_RANKING_LOSS_WEIGHT=${UTILITY_RANKING_LOSS_WEIGHT}"
    echo "UTILITY_RANKING_MARGIN=${UTILITY_RANKING_MARGIN}"
    echo "UTILITY_LEARNING_RATE=${UTILITY_LEARNING_RATE}"
    echo "UTILITY_GATE_TEMPERATURE=${UTILITY_GATE_TEMPERATURE}"
    echo "UTILITY_MIN_GAIN=${UTILITY_MIN_GAIN}"
    echo "UTILITY_TARGET_EPS=${UTILITY_TARGET_EPS}"
    echo "UTILITY_INTERVENTION_FLOOR=${UTILITY_INTERVENTION_FLOOR}"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${CV_LOG_DIR}/cv.env"

for fold in "${CV_FOLDS[@]}"; do
    for seed in "${CV_SEEDS[@]}"; do
        source_stage="${SOURCE_CV_DIR}/runs/f${fold}_s${seed}/pipeline.env"
        [[ -f "${source_stage}" ]] || {
            echo "Missing ${source_stage}" >&2
            exit 2
        }
        pretrain_id="$(read_value "${source_stage}" PRETRAIN_RUN_ID)"
        [[ -n "${pretrain_id}" ]] || {
            echo "Missing PRETRAIN_RUN_ID in ${source_stage}" >&2
            exit 2
        }
        run_dir="${CV_LOG_DIR}/runs/f${fold}_s${seed}"
        mkdir -p "${run_dir}"
        echo "===== CV fold=${fold} seed=${seed} ====="
        FOLD="${fold}" N_FOLDS=3 SEED="${seed}" SPLIT=rolling_holdout \
        PRED_LEN="${PRED_LEN}" EVAL_STRIDE="${PRED_LEN}" PRETRAIN_RUN_ID="${pretrain_id}" \
        RUN_ID="${CV_ID}_f${fold}_s${seed}_finetune" \
        SDWPF_LOG_DIR="${run_dir}" SDWPF_LOG_FILE="${run_dir}/finetune.log" \
        bash scripts/finetune/SDWPF_ablation_prompt.sh
    done
done

{
    for fold in "${CV_FOLDS[@]}"; do
        for seed in "${CV_SEEDS[@]}"; do
            echo "===== CV fold=${fold} seed=${seed} ====="
            grep -E '^Epoch:|^Early stopping|^\[AUDIT\] FINETUNE_CHECKPOINT=' \
                "${CV_LOG_DIR}/runs/f${fold}_s${seed}/finetune.log" || true
        done
    done
} > "${CV_LOG_DIR}/summary.txt"

python scripts/summarize_sdwpf_cv.py "${CV_LOG_DIR}/summary.txt" \
    --output-dir "${CV_LOG_DIR}" \
    --expected-folds "${FOLDS}" \
    --expected-seeds "${SEEDS}"
echo "COMPLETED_AT=$(date --iso-8601=seconds)" >> "${CV_LOG_DIR}/cv.env"
echo "[UTILITY-WIKI] Fine-tuning-only CV complete: ${CV_LOG_DIR}"
