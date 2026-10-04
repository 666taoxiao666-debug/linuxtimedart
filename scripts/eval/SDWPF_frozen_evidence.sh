#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ROOT="${1:?Usage: bash scripts/eval/SDWPF_frozen_evidence.sh BENCHMARK_DIR}"
[[ -d "${ROOT}" ]] || { echo "Benchmark directory does not exist: ${ROOT}" >&2; exit 2; }
PROTOCOL_CONFIG="${ROOT}/frozen_protocol.json"
[[ -f "${PROTOCOL_CONFIG}" ]] || { echo "Missing frozen protocol snapshot" >&2; exit 2; }
python scripts/check_sdwpf_frozen_protocol.py --config "${PROTOCOL_CONFIG}" >/dev/null

json_value() {
    python - "$PROTOCOL_CONFIG" "$1" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}
read_value() {
    sed -n "s/^$2=//p" "$1" | tail -n 1
}

TREND_CV_DIR="$(json_value frozen_internal_references.trend_cv_dir)"
WIKI_CV_DIR="$(json_value frozen_internal_references.wiki_cv_dir)"
GPU="${GPU:-0}"
for directory in "${TREND_CV_DIR}" "${WIKI_CV_DIR}"; do
    [[ -d "${directory}" ]] || { echo "Frozen reference is missing: ${directory}" >&2; exit 2; }
done
mkdir -p "${ROOT}/evidence/reports" "${ROOT}/evidence/paired"

for seed in 2024 2025 2026; do
    for fold in 0 1 2; do
        run="f${fold}_s${seed}"
        trend_report="${ROOT}/evidence/reports/trend/${run}"
        wiki_report="${ROOT}/evidence/reports/wiki/${run}"
        trend_stage="${TREND_CV_DIR}/runs/${run}/pipeline.env"
        wiki_log="${WIKI_CV_DIR}/runs/${run}/finetune.log"
        [[ -f "${trend_stage}" ]] || { echo "Missing ${trend_stage}" >&2; exit 2; }
        [[ -f "${wiki_log}" ]] || { echo "Missing ${wiki_log}" >&2; exit 2; }
        trend_checkpoint="$(read_value "${trend_stage}" FINETUNE_CHECKPOINT)"
        wiki_checkpoint="$(grep -F '[AUDIT] FINETUNE_CHECKPOINT=' "${wiki_log}" | tail -n 1 | sed 's/^.*FINETUNE_CHECKPOINT=//')"
        [[ -f "${trend_checkpoint}" ]] || { echo "Missing trend checkpoint: ${trend_checkpoint}" >&2; exit 2; }
        [[ -f "${wiki_checkpoint}" ]] || { echo "Missing Wiki checkpoint: ${wiki_checkpoint}" >&2; exit 2; }

        if [[ ! -f "${trend_report}/predictions.npz" ]]; then
            mkdir -p "${trend_report}"
            FINETUNE_CHECKPOINT="${trend_checkpoint}" MODEL=PromptTimeDART \
            PROMPT_ROUTER=trend UTILITY_WIKI=0 REGIME_LABEL_METHOD=trend_quantile \
            FOLD="${fold}" N_FOLDS=3 SEED="${seed}" SPLIT=rolling_holdout \
            PRED_LEN=12 EVAL_STRIDE=12 EVAL_SPLIT=val GPU="${GPU}" \
            REPORT_OUTPUT_DIR="${trend_report}" \
            bash scripts/eval/SDWPF_final_eval.sh > "${trend_report}/report.log" 2>&1
        fi
        if [[ ! -f "${wiki_report}/predictions.npz" ]]; then
            mkdir -p "${wiki_report}"
            FINETUNE_CHECKPOINT="${wiki_checkpoint}" MODEL=PromptTimeDART \
            PROMPT_ROUTER=compositional_wiki UTILITY_WIKI=1 UTILITY_FACTORIZED=1 \
            UTILITY_ADAPTER_MODE=calibrated_evidence UTILITY_EVENT_MAX_SCALE=0.2 \
            UTILITY_COMPOSITION_MAX_SCALE=0.05 UTILITY_GATE_TEMPERATURE=0.05 \
            UTILITY_MIN_GAIN=0.003 UTILITY_INTERVENTION_FLOOR=1 \
            REGIME_LABEL_METHOD=trend_quantile FOLD="${fold}" N_FOLDS=3 SEED="${seed}" \
            SPLIT=rolling_holdout PRED_LEN=12 EVAL_STRIDE=12 EVAL_SPLIT=val GPU="${GPU}" \
            REPORT_OUTPUT_DIR="${wiki_report}" \
            bash scripts/eval/SDWPF_final_eval.sh > "${wiki_report}/report.log" 2>&1
        fi
        if [[ ! -f "${wiki_report}/diagnostic/intervention_summary.csv" ]]; then
            mkdir -p "${wiki_report}/diagnostic"
            FINETUNE_CHECKPOINT="${wiki_checkpoint}" MODEL=PromptTimeDART \
            PROMPT_ROUTER=compositional_wiki UTILITY_WIKI=1 UTILITY_FACTORIZED=1 \
            UTILITY_ADAPTER_MODE=calibrated_evidence UTILITY_EVENT_MAX_SCALE=0.2 \
            UTILITY_COMPOSITION_MAX_SCALE=0.05 UTILITY_GATE_TEMPERATURE=0.05 \
            UTILITY_MIN_GAIN=0.003 UTILITY_INTERVENTION_FLOOR=1 \
            REGIME_LABEL_METHOD=trend_quantile WIKI_DIAGNOSTIC=1 WIKI_DIAGNOSTIC_SINGLE_EVENTS=1 \
            FOLD="${fold}" N_FOLDS=3 SEED="${seed}" SPLIT=rolling_holdout \
            PRED_LEN=12 EVAL_STRIDE=12 EVAL_SPLIT=val GPU="${GPU}" \
            REPORT_OUTPUT_DIR="${wiki_report}/diagnostic" \
            bash scripts/eval/SDWPF_final_eval.sh > "${wiki_report}/diagnostic/report.log" 2>&1
        fi
        paired="${ROOT}/evidence/paired/wiki_vs_trend/${run}"
        if [[ ! -f "${paired}/paired_statistics.json" ]]; then
            python scripts/compare_paired_forecasts.py \
                --candidate-dir "${wiki_report}" --reference-dir "${trend_report}" \
                --output-dir "${paired}" --block-length 12 --replicates 5000 --seed 2024
        fi

        for model in PatchTST DLinear; do
            candidate="${ROOT}/runs/deep/${model}/${run}/artifacts"
            [[ -f "${candidate}/predictions.npz" ]] || { echo "Missing deep baseline report: ${candidate}" >&2; exit 2; }
            paired="${ROOT}/evidence/paired/${model}_vs_trend/${run}"
            if [[ ! -f "${paired}/paired_statistics.json" ]]; then
                python scripts/compare_paired_forecasts.py \
                    --candidate-dir "${candidate}" --reference-dir "${trend_report}" \
                    --output-dir "${paired}" --block-length 12 --replicates 5000 --seed 2024
            fi
        done
    done
done

python scripts/summarize_frozen_evidence.py "${ROOT}"
echo "[FROZEN EVIDENCE] Paired statistics and Wiki intervention audit complete: ${ROOT}/evidence"
