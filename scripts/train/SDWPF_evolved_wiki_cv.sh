#!/usr/bin/env bash
set -euo pipefail

# Offline lifecycle followed by fold-matched source and factorized CV stages.
# Each fold needs its own train_oof evidence because rolling train cutoffs differ.
# Validation/test evidence is rejected by evolve_wind_event_wiki.py.
POINTER="outputs/wiki/evolved/latest.txt"
if [[ "${1:-}" == "--status" ]]; then
    [[ -f "${POINTER}" ]] || { echo "No evolved Wiki CV has been launched."; exit 2; }
    WIKI_DIR="$(< "${POINTER}")"
    echo "WIKI_DIR=${WIKI_DIR}"
    for stage in source_cv factorized_cv; do
        pointer_file="${WIKI_DIR}/${stage}_dir.txt"
        if [[ -f "${pointer_file}" ]]; then
            stage_dir="$(< "${pointer_file}")"
            echo "${stage^^}_DIR=${stage_dir}"
            [[ ! -f "${stage_dir}/status.env" ]] || cat "${stage_dir}/status.env"
            [[ ! -f "${stage_dir}/cv_metrics_summary.txt" ]] || cat "${stage_dir}/cv_metrics_summary.txt"
            [[ ! -f "${stage_dir}/wiki_vs_trend.txt" ]] || cat "${stage_dir}/wiki_vs_trend.txt"
        else
            echo "${stage}: pending"
        fi
    done
    exit 0
fi
if [[ -n "${1:-}" ]]; then
    echo "Usage: bash $0 [--status]" >&2
    exit 2
fi
if [[ -z "${TREND_CV_DIR:-}" || ! -f "${TREND_CV_DIR}/cv.env" ]]; then
    echo "TREND_CV_DIR must point to the matched trend CV directory." >&2
    exit 2
fi

PRED_LEN="${PRED_LEN:-12}"
FOLDS="${FOLDS:-0}"
SEEDS="${SEEDS:-2024}"
WIKI_LLM_PATH="${WIKI_LLM_PATH:-outputs/model_cache/Qwen2.5-0.5B}"
read -r -a REQUESTED_FOLDS <<< "${FOLDS}"
if [[ "${#REQUESTED_FOLDS[@]}" -ne 1 ]]; then
    echo "One evidence JSON must not be silently reused across different fold train cutoffs. Run this script once per fold with matched train_oof evidence." >&2
    exit 2
fi
if [[ -z "${EVIDENCE:-}" ]]; then
    read -r -a REQUESTED_SEEDS <<< "${SEEDS}"
    [[ "${#REQUESTED_SEEDS[@]}" -gt 0 ]] || {
        echo "SEEDS must contain at least one seed." >&2
        exit 2
    }
    echo "[EVOLVED-WIKI] No EVIDENCE supplied; building real forward-OOF evidence for fold ${REQUESTED_FOLDS[0]}"
    FOLD="${REQUESTED_FOLDS[0]}" SEED="${OOF_SEED:-${REQUESTED_SEEDS[0]}}" \
    TREND_CV_DIR="${TREND_CV_DIR}" \
    WIKI_LLM_PATH="${WIKI_LLM_PATH}" \
    bash scripts/train/SDWPF_build_wiki_oof_evidence.sh
    OOF_DIR="$(< outputs/logs/SDWPF/wiki_oof_latest.txt)"
    EVIDENCE="${OOF_DIR}/train_oof_wiki_candidates.json"
fi
[[ -f "${EVIDENCE}" ]] || { echo "Missing OOF evidence: ${EVIDENCE}" >&2; exit 2; }
EVIDENCE_FOLD="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("sdwpf_fold", ""))' "${EVIDENCE}")"
if [[ "${EVIDENCE_FOLD}" != "${REQUESTED_FOLDS[0]}" ]]; then
    echo "Evidence sdwpf_fold=${EVIDENCE_FOLD:-missing} does not match FOLDS=${FOLDS}." >&2
    exit 2
fi
EVIDENCE_HORIZON="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("pred_len", ""))' "${EVIDENCE}")"
if [[ "${EVIDENCE_HORIZON}" != "${PRED_LEN}" ]]; then
    echo "Evidence pred_len=${EVIDENCE_HORIZON:-missing} does not match PRED_LEN=${PRED_LEN}." >&2
    exit 2
fi

UTILITY_INIT_CHECKPOINT=""
UTILITY_INIT_CHECKPOINT_SHA256=""
if [[ "${WIKI_UTILITY_WARM_START:-0}" == "1" ]]; then
    read -r -a REQUESTED_SEEDS <<< "${SEEDS}"
    [[ "${#REQUESTED_SEEDS[@]}" -eq 1 ]] || {
        echo "WIKI_UTILITY_WARM_START requires exactly one seed because OOF provenance is seed specific." >&2
        exit 2
    }
    UTILITY_INIT_CHECKPOINT="$(python -c 'import json,sys; print((json.load(open(sys.argv[1], encoding="utf-8")).get("provenance") or {}).get("model_checkpoint", ""))' "${EVIDENCE}")"
    UTILITY_INIT_CHECKPOINT_SHA256="$(python -c 'import json,sys; print((json.load(open(sys.argv[1], encoding="utf-8")).get("provenance") or {}).get("model_checkpoint_sha256", ""))' "${EVIDENCE}")"
    [[ -f "${UTILITY_INIT_CHECKPOINT}" ]] || {
        echo "OOF provenance model checkpoint is missing: ${UTILITY_INIT_CHECKPOINT}" >&2
        exit 2
    }
    [[ "${UTILITY_INIT_CHECKPOINT_SHA256}" =~ ^[0-9a-f]{64}$ ]] || {
        echo "OOF provenance lacks a valid model_checkpoint_sha256." >&2
        exit 2
    }
    ACTUAL_UTILITY_INIT_SHA256="$(sha256sum "${UTILITY_INIT_CHECKPOINT}" | awk '{print $1}')"
    [[ "${ACTUAL_UTILITY_INIT_SHA256}" == "${UTILITY_INIT_CHECKPOINT_SHA256}" ]] || {
        echo "OOF provenance checkpoint SHA-256 mismatch." >&2
        exit 2
    }
    echo "[EVOLVED-WIKI] Audited utility warm-start: ${UTILITY_INIT_CHECKPOINT}"
fi

WIKI_VERSION="${WIKI_VERSION:-$(date +%Y%m%d_%H%M%S)}"
WIKI_DIR="${WIKI_DIR:-outputs/wiki/evolved/${WIKI_VERSION}}"
WIKI_CONFIG="${WIKI_CONFIG:-${WIKI_DIR}/wind_event_factor_wiki.json}"
WIKI_BUNDLE="${WIKI_BUNDLE:-${WIKI_DIR}/wind_event_factor_wiki_qwen.npz}"
WIKI_AUDIT="${WIKI_AUDIT:-${WIKI_DIR}/lifecycle_audit.json}"
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-auto}"

mkdir -p "${WIKI_DIR}"
printf '%s\n' "${WIKI_DIR}" > "${POINTER}"

MACRO_GUARD_ARGS=()
if [[ "${WIKI_MACRO_TRANSFER_GUARD:-0}" == "1" ]]; then
    MACRO_GUARD_ARGS+=(--macro_transfer_guard)
fi
TEMPORAL_GUARD_ARGS=()
if [[ "${WIKI_TEMPORAL_STABILITY_GUARD:-0}" == "1" ]]; then
    [[ -f "${TEMPORAL_EVIDENCE:-}" ]] || {
        echo "TEMPORAL_EVIDENCE must point to a train-OOF temporal report." >&2
        exit 2
    }
    TEMPORAL_GUARD_ARGS+=(--temporal_stability_guard --temporal_evidence "${TEMPORAL_EVIDENCE}")
fi

python -u scripts/evolve_wind_event_wiki.py \
    --base_config "${BASE_WIKI_CONFIG:-configs/wind_event_factor_wiki.json}" \
    --evidence "${EVIDENCE}" \
    --output_config "${WIKI_CONFIG}" \
    --audit_output "${WIKI_AUDIT}" \
    --min_source_turbines "${MIN_SOURCE_TURBINES:-3}" \
    --min_total_windows "${MIN_TOTAL_WINDOWS:-300}" \
    --min_positive_turbine_fraction "${MIN_POSITIVE_TURBINE_FRACTION:-0.6666666667}" \
    --min_transfer_lcb "${MIN_TRANSFER_LCB:-0.0}" \
    --target_utility "${TARGET_UTILITY:-0.02}" \
    --z_value "${UTILITY_Z_VALUE:-1.645}" \
    --forget_half_life_steps "${FORGET_HALF_LIFE_STEPS:-100000}" \
    --retire_weight "${RETIRE_WEIGHT:-0.10}" \
    --max_merged_insights "${MAX_MERGED_INSIGHTS:-3}" \
    "${MACRO_GUARD_ARGS[@]}" \
    "${TEMPORAL_GUARD_ARGS[@]}"

python -u scripts/build_wind_regime_wiki.py \
    --config "${WIKI_CONFIG}" \
    --output "${WIKI_BUNDLE}" \
    --llm_path "${WIKI_LLM_PATH}" \
    --device "${WIKI_BUILD_DEVICE}"

echo "[EVOLVED-WIKI] Frozen lifecycle artifacts: ${WIKI_DIR}"
echo "[EVOLVED-WIKI] Stage 1/2: matched pretraining and static Wiki reference"
SCENE_WIKI_CONFIG="${WIKI_CONFIG}" \
SCENE_WIKI_EMBEDDINGS="${WIKI_BUNDLE}" \
WIKI_LLM_PATH="${WIKI_LLM_PATH}" \
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE}" \
PROMPT_ROUTER=compositional_wiki UTILITY_WIKI=0 UTILITY_FACTORIZED=0 \
UTILITY_ADAPTER_MODE=legacy FREEZE_NON_UTILITY=0 \
PRED_LEN="${PRED_LEN}" FOLDS="${FOLDS}" SEEDS="${SEEDS}" \
bash scripts/train/SDWPF_paper_cv.sh 2>&1 | tee "${WIKI_DIR}/source_cv.launch.log"

SOURCE_CV_DIR="$(sed -n 's/^\[LOG\] Directory: //p' "${WIKI_DIR}/source_cv.launch.log" | head -n 1)"
if [[ -z "${SOURCE_CV_DIR}" || ! -f "${SOURCE_CV_DIR}/cv.env" ]]; then
    echo "Could not resolve matched source CV directory." >&2
    exit 3
fi
printf '%s\n' "${SOURCE_CV_DIR}" > "${WIKI_DIR}/source_cv_dir.txt"

echo "[EVOLVED-WIKI] Stage 2/2: factorized selective residual fine-tuning"
SCENE_WIKI_CONFIG="${WIKI_CONFIG}" \
SCENE_WIKI_EMBEDDINGS="${WIKI_BUNDLE}" \
SOURCE_CV_DIR="${SOURCE_CV_DIR}" TREND_CV_DIR="${TREND_CV_DIR}" \
UTILITY_INIT_CHECKPOINT="${UTILITY_INIT_CHECKPOINT}" \
UTILITY_INIT_CHECKPOINT_SHA256="${UTILITY_INIT_CHECKPOINT_SHA256}" \
UTILITY_WIKI=1 UTILITY_FACTORIZED=1 \
UTILITY_ADAPTER_MODE=calibrated_evidence FREEZE_NON_UTILITY=1 \
UTILITY_ADAPTER_WARMUP_EPOCHS="${UTILITY_ADAPTER_WARMUP_EPOCHS:-3}" \
UTILITY_INTERVENTION_FLOOR=1 \
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" PATIENCE="${PATIENCE:-7}" \
UTILITY_LEARNING_RATE="${UTILITY_LEARNING_RATE:-0.0001}" \
PRED_LEN="${PRED_LEN}" FOLDS="${FOLDS}" SEEDS="${SEEDS}" \
bash scripts/train/SDWPF_utility_wiki_cv.sh 2>&1 | tee "${WIKI_DIR}/factorized_cv.launch.log"

FACTORIZED_CV_DIR="$(sed -n 's/^\[LOG\] Directory: //p' "${WIKI_DIR}/factorized_cv.launch.log" | head -n 1)"
if [[ -z "${FACTORIZED_CV_DIR}" || ! -f "${FACTORIZED_CV_DIR}/cv_metrics.csv" ]]; then
    echo "Could not resolve factorized CV metrics directory." >&2
    exit 3
fi
printf '%s\n' "${FACTORIZED_CV_DIR}" > "${WIKI_DIR}/factorized_cv_dir.txt"
python scripts/summarize_hierarchical_wiki.py \
    --wiki-dir "${FACTORIZED_CV_DIR}" --trend-dir "${TREND_CV_DIR}"
echo "[EVOLVED-WIKI] Matched source CV: ${SOURCE_CV_DIR}"
echo "[EVOLVED-WIKI] Factorized CV: ${FACTORIZED_CV_DIR}"
echo "[EVOLVED-WIKI] Paired result: ${FACTORIZED_CV_DIR}/wiki_vs_trend.txt"
