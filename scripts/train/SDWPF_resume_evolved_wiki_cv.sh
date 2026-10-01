#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Resume a failed evolved-Wiki CV after pretraining has completed. The original
# OOF evidence, evolved Wiki, embeddings, and pretrained checkpoint are reused.
WIKI_DIR="${WIKI_DIR:-$(< outputs/wiki/evolved/latest.txt)}"
TREND_CV_DIR="${TREND_CV_DIR:-}"
[[ -d "${WIKI_DIR}" && -f "${WIKI_DIR}/source_cv.launch.log" ]] || {
    echo "WIKI_DIR has no previously launched source CV: ${WIKI_DIR}" >&2
    exit 2
}
[[ -n "${TREND_CV_DIR}" && -f "${TREND_CV_DIR}/cv.env" ]] || {
    echo "Set TREND_CV_DIR to the original, fold-matched trend CV directory." >&2
    exit 2
}
SOURCE_CV_DIR="$(sed -n 's/^\[LOG\] Directory: //p' "${WIKI_DIR}/source_cv.launch.log" | head -n 1)"
[[ -n "${SOURCE_CV_DIR}" && -f "${SOURCE_CV_DIR}/cv.env" ]] || {
    echo "Could not resolve the failed source CV directory." >&2
    exit 2
}
read_value() {
    sed -n "s/^${2}=//p" "${1}" | tail -n 1
}
SOURCE_ENV="${SOURCE_CV_DIR}/cv.env"
[[ "$(read_value "${SOURCE_ENV}" PROMPT_ROUTER)" == "compositional_wiki" ]] || {
    echo "The source CV must use compositional_wiki." >&2
    exit 2
}
[[ "$(read_value "${SOURCE_ENV}" SPLIT)" == "rolling_holdout" ]] || {
    echo "The source CV must use rolling_holdout." >&2
    exit 2
}
FOLDS="$(read_value "${SOURCE_ENV}" FOLDS)"
SEEDS="$(read_value "${SOURCE_ENV}" SEEDS)"
PRED_LEN="$(read_value "${SOURCE_ENV}" PRED_LEN)"
SCENE_WIKI_CONFIG="$(read_value "${SOURCE_ENV}" SCENE_WIKI_CONFIG)"
SCENE_WIKI_EMBEDDINGS="$(read_value "${SOURCE_ENV}" SCENE_WIKI_EMBEDDINGS)"
[[ -f "${SCENE_WIKI_CONFIG}" && -f "${SCENE_WIKI_EMBEDDINGS}" ]] || {
    echo "The frozen evolved Wiki config or embedding bundle is missing." >&2
    exit 2
}
read -r -a CV_FOLDS <<< "${FOLDS}"
read -r -a CV_SEEDS <<< "${SEEDS}"
[[ "${#CV_FOLDS[@]}" -gt 0 && "${#CV_SEEDS[@]}" -gt 0 ]] || {
    echo "The source CV fold/seed matrix is empty." >&2
    exit 2
}

exec > >(tee -a "${WIKI_DIR}/resume.log") 2>&1
echo "[RESUME] Wiki=${WIKI_DIR} source_cv=${SOURCE_CV_DIR} trend_cv=${TREND_CV_DIR}"
for fold in "${CV_FOLDS[@]}"; do
    for seed in "${CV_SEEDS[@]}"; do
        run_dir="${SOURCE_CV_DIR}/runs/f${fold}_s${seed}"
        stage="${run_dir}/pipeline.env"
        [[ -f "${stage}" && -f "${run_dir}/pretrain.log" ]] || {
            echo "Missing completed pretraining record for fold=${fold} seed=${seed}" >&2
            exit 3
        }
        pretrain_dir="$(sed -n 's/^>>>>>>> start pre-training: //p' "${run_dir}/pretrain.log" | head -n 1)"
        [[ -n "${pretrain_dir}" && -f "${pretrain_dir}/ckpt_best.pth" ]] || {
            echo "Pretrained ckpt_best.pth is missing for fold=${fold} seed=${seed}" >&2
            exit 3
        }
        pretrain_id="$(read_value "${stage}" PRETRAIN_RUN_ID)"
        [[ -n "${pretrain_id}" ]] || {
            echo "PRETRAIN_RUN_ID is missing from ${stage}" >&2
            exit 3
        }
        checkpoint="$(read_value "${stage}" FINETUNE_CHECKPOINT)"
        if [[ -n "${checkpoint}" && -f "${checkpoint}" ]]; then
            echo "[RESUME] Reusing completed fine-tuning fold=${fold} seed=${seed}: ${checkpoint}"
            finetune_log="${run_dir}/finetune.log"
        else
            echo "[RESUME] Fine-tuning fold=${fold} seed=${seed}; pretraining is NOT rerun"
            finetune_log="${run_dir}/finetune_resume.log"
            FOLD="${fold}" SEED="${seed}" N_FOLDS=3 SPLIT=rolling_holdout \
            PRED_LEN="${PRED_LEN}" EVAL_STRIDE="${PRED_LEN}" \
            SDWPF_TRAIN_RATIO="$(read_value "${stage}" SDWPF_TRAIN_RATIO)" \
            SDWPF_VAL_RATIO="$(read_value "${stage}" SDWPF_VAL_RATIO)" \
            PRETRAIN_RUN_ID="${pretrain_id}" ALLOW_RANDOM=0 \
            PROMPT_ROUTER=compositional_wiki \
            SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG}" \
            SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS}" \
            SCENE_WIKI_TOP_K="$(read_value "${SOURCE_ENV}" SCENE_WIKI_TOP_K)" \
            SCENE_WIKI_TEMPERATURE="$(read_value "${SOURCE_ENV}" SCENE_WIKI_TEMPERATURE)" \
            SCENE_WIKI_RULE_WEIGHT="$(read_value "${SOURCE_ENV}" SCENE_WIKI_RULE_WEIGHT)" \
            SCENE_WIKI_PROMPT_GATE_INIT="$(read_value "${SOURCE_ENV}" SCENE_WIKI_PROMPT_GATE_INIT)" \
            SCENE_WIKI_ACTIVATION_THRESHOLD="$(read_value "${SOURCE_ENV}" SCENE_WIKI_ACTIVATION_THRESHOLD)" \
            SCENE_WIKI_CONFIDENCE_POWER="$(read_value "${SOURCE_ENV}" SCENE_WIKI_CONFIDENCE_POWER)" \
            REGIME_LABEL_METHOD="$(read_value "${SOURCE_ENV}" REGIME_LABEL_METHOD)" \
            REGIME_CALIBRATION_QUANTILE="$(read_value "${SOURCE_ENV}" REGIME_CALIBRATION_QUANTILE)" \
            WIKI_LLM_PATH="$(read_value "${stage}" WIKI_LLM_PATH)" \
            UTILITY_WIKI=0 TRAIN_EPOCHS="$(read_value "${stage}" FINETUNE_EPOCHS)" \
            LEARNING_RATE="$(read_value "${SOURCE_ENV}" FINETUNE_LEARNING_RATE)" \
            NEW_MODULE_LEARNING_RATE="$(read_value "${SOURCE_ENV}" NEW_MODULE_LEARNING_RATE)" \
            MIX_MSE_WEIGHT="$(read_value "${SOURCE_ENV}" MIX_MSE_WEIGHT)" \
            PATIENCE="$(read_value "${stage}" FINETUNE_PATIENCE)" \
            PCT_START="$(read_value "${stage}" FINETUNE_PCT_START)" \
            LOSS="$(read_value "${stage}" LOSS)" \
            EARLY_STOP_METRIC="$(read_value "${stage}" EARLY_STOP_METRIC)" \
            RATED_POWER="$(read_value "${stage}" RATED_POWER)" \
            RUN_ID="$(read_value "${stage}" PIPELINE_ID)_finetune" \
            SDWPF_LOG_DIR="${run_dir}" SDWPF_LOG_FILE="${finetune_log}" \
            bash scripts/finetune/SDWPF_ablation_prompt.sh
            checkpoint="$(grep -F '[AUDIT] FINETUNE_CHECKPOINT=' "${finetune_log}" | tail -n 1 | sed 's/^.*FINETUNE_CHECKPOINT=//')"
            [[ -n "${checkpoint}" && -f "${checkpoint}" ]] || {
                echo "Resumed fine-tuning produced no checkpoint: ${finetune_log}" >&2
                exit 3
            }
            echo "FINETUNE_CHECKPOINT=${checkpoint}" >> "${stage}"
            echo "RESUMED_COMPLETED_AT=$(date --iso-8601=seconds)" >> "${stage}"
        fi
        {
            echo "PIPELINE_ID=$(read_value "${stage}" PIPELINE_ID)"
            echo "FOLD=${fold} SEED=${seed} SPLIT=rolling_holdout PRED_LEN=${PRED_LEN}"
            grep -E '^Epoch:|^Pretrain early stopping' "${run_dir}/pretrain.log" || true
            grep -E '^Transferred |^Optimizer groups:|^Epoch:|^Early stopping|^\[AUDIT\] FINETUNE_CHECKPOINT=' "${finetune_log}" || true
        } > "${run_dir}/summary.txt"
    done
done

{
    for fold in "${CV_FOLDS[@]}"; do
        for seed in "${CV_SEEDS[@]}"; do
            echo "===== CV fold=${fold} seed=${seed} ====="
            cat "${SOURCE_CV_DIR}/runs/f${fold}_s${seed}/summary.txt"
        done
    done
} > "${SOURCE_CV_DIR}/summary.txt"
python scripts/summarize_sdwpf_cv.py "${SOURCE_CV_DIR}/summary.txt" \
    --output-dir "${SOURCE_CV_DIR}" --expected-folds "${FOLDS}" --expected-seeds "${SEEDS}"
echo "RESUMED_COMPLETED_AT=$(date --iso-8601=seconds)" >> "${SOURCE_ENV}"
python -m utils.sdwpf_logging finish --log-dir "${SOURCE_CV_DIR}" --exit-code 0
printf '%s\n' "${SOURCE_CV_DIR}" > "${WIKI_DIR}/source_cv_dir.txt"
echo "[RESUME] Source CV completed without rerunning pretraining: ${SOURCE_CV_DIR}"

echo "[RESUME] Stage 2/2: factorized selective residual fine-tuning"
SCENE_WIKI_CONFIG="${SCENE_WIKI_CONFIG}" \
SCENE_WIKI_EMBEDDINGS="${SCENE_WIKI_EMBEDDINGS}" \
SOURCE_CV_DIR="${SOURCE_CV_DIR}" TREND_CV_DIR="${TREND_CV_DIR}" \
UTILITY_WIKI=1 UTILITY_FACTORIZED=1 \
UTILITY_ADAPTER_MODE=calibrated_evidence FREEZE_NON_UTILITY=1 \
UTILITY_ADAPTER_WARMUP_EPOCHS="${UTILITY_ADAPTER_WARMUP_EPOCHS:-3}" \
UTILITY_INTERVENTION_FLOOR=1 \
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" PATIENCE="${PATIENCE:-7}" \
UTILITY_LEARNING_RATE="${UTILITY_LEARNING_RATE:-0.0001}" \
PRED_LEN="${PRED_LEN}" FOLDS="${FOLDS}" SEEDS="${SEEDS}" \
bash scripts/train/SDWPF_utility_wiki_cv.sh 2>&1 | tee "${WIKI_DIR}/factorized_cv.resume.launch.log"

FACTORIZED_CV_DIR="$(sed -n 's/^\[LOG\] Directory: //p' "${WIKI_DIR}/factorized_cv.resume.launch.log" | head -n 1)"
[[ -n "${FACTORIZED_CV_DIR}" && -f "${FACTORIZED_CV_DIR}/cv_metrics.csv" ]] || {
    echo "Factorized CV metrics directory was not produced." >&2
    exit 3
}
printf '%s\n' "${FACTORIZED_CV_DIR}" > "${WIKI_DIR}/factorized_cv_dir.txt"
python scripts/summarize_hierarchical_wiki.py \
    --wiki-dir "${FACTORIZED_CV_DIR}" --trend-dir "${TREND_CV_DIR}"
echo "[RESUME] Paired result: ${FACTORIZED_CV_DIR}/wiki_vs_trend.txt"
