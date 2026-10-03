#!/usr/bin/env bash
set -euo pipefail

# Run the still-unmeasured rolling folds sequentially: each fold builds its own
# train-only OOF evidence, and the underlying launcher uses a shared latest.txt.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TREND_CV_DIR="${TREND_CV_DIR:-outputs/logs/SDWPF/20260914/001_cv_trend_h12_rolling_holdout_folds0-1-2_seeds2024-2025-202_h36d88ba5e218}"
WIKI_LLM_PATH="${WIKI_LLM_PATH:-outputs/model_cache/Qwen2.5-0.5B}"
SEEDS="${SEEDS:-2024}"
PRED_LEN="${PRED_LEN:-12}"
FOLDS_TO_RUN="${FOLDS_TO_RUN:-1 2}"

[[ -f "${TREND_CV_DIR}/cv.env" ]] || { echo "Missing trend CV: ${TREND_CV_DIR}" >&2; exit 2; }
[[ -f "${WIKI_LLM_PATH}/config.json" ]] || { echo "Missing local Wiki encoder: ${WIKI_LLM_PATH}" >&2; exit 2; }

read -r -a folds <<< "${FOLDS_TO_RUN}"
[[ "${#folds[@]}" -gt 0 ]] || { echo "FOLDS_TO_RUN is empty" >&2; exit 2; }
for fold in "${folds[@]}"; do
    [[ "${fold}" =~ ^[0-2]$ ]] || { echo "Invalid fold: ${fold}" >&2; exit 2; }
done

echo "[EVOLVED-REMAINING] Started $(date --iso-8601=seconds) commit=$(git rev-parse HEAD) folds=${FOLDS_TO_RUN} seeds=${SEEDS}"
for fold in "${folds[@]}"; do
    echo "[EVOLVED-REMAINING] START fold=${fold} $(date --iso-8601=seconds)"
    TREND_CV_DIR="${TREND_CV_DIR}" WIKI_LLM_PATH="${WIKI_LLM_PATH}" \
        FOLDS="${fold}" SEEDS="${SEEDS}" PRED_LEN="${PRED_LEN}" \
        bash scripts/train/SDWPF_evolved_wiki_cv.sh
    wiki_dir="$(< outputs/wiki/evolved/latest.txt)"
    result_dir="$(< "${wiki_dir}/factorized_cv_dir.txt")"
    [[ -f "${result_dir}/wiki_vs_trend.txt" ]] || { echo "Missing paired result for fold ${fold}" >&2; exit 3; }
    echo "[EVOLVED-REMAINING] DONE fold=${fold} result=${result_dir} $(date --iso-8601=seconds)"
done
echo "[EVOLVED-REMAINING] COMPLETED $(date --iso-8601=seconds)"
