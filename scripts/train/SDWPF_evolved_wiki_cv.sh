#!/usr/bin/env bash
set -euo pipefail

# Offline lifecycle stage followed by the existing leakage-safe 3-fold x 3-seed
# protocol.  EVIDENCE must contain train_oof statistics; validation/test evidence
# is rejected by evolve_wind_event_wiki.py.
if [[ -z "${EVIDENCE:-}" ]]; then
    echo "EVIDENCE must point to a train_oof Wiki candidate JSON file." >&2
    exit 2
fi

WIKI_VERSION="${WIKI_VERSION:-$(date +%Y%m%d_%H%M%S)}"
WIKI_DIR="${WIKI_DIR:-outputs/wiki/evolved/${WIKI_VERSION}}"
WIKI_CONFIG="${WIKI_CONFIG:-${WIKI_DIR}/wind_event_factor_wiki.json}"
WIKI_BUNDLE="${WIKI_BUNDLE:-${WIKI_DIR}/wind_event_factor_wiki_qwen.npz}"
WIKI_AUDIT="${WIKI_AUDIT:-${WIKI_DIR}/lifecycle_audit.json}"
WIKI_LLM_PATH="${WIKI_LLM_PATH:-Qwen/Qwen2.5-0.5B}"
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE:-auto}"

mkdir -p "${WIKI_DIR}"

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
    --max_merged_insights "${MAX_MERGED_INSIGHTS:-3}"

python -u scripts/build_wind_regime_wiki.py \
    --config "${WIKI_CONFIG}" \
    --output "${WIKI_BUNDLE}" \
    --llm_path "${WIKI_LLM_PATH}" \
    --device "${WIKI_BUILD_DEVICE}"

echo "[EVOLVED-WIKI] Frozen lifecycle artifacts: ${WIKI_DIR}"
SCENE_WIKI_CONFIG="${WIKI_CONFIG}" \
SCENE_WIKI_EMBEDDINGS="${WIKI_BUNDLE}" \
WIKI_LLM_PATH="${WIKI_LLM_PATH}" \
WIKI_BUILD_DEVICE="${WIKI_BUILD_DEVICE}" \
bash scripts/train/SDWPF_paper_cv.sh
