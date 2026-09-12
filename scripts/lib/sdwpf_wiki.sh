#!/usr/bin/env bash

# Shared offline preparation for the SDWPF semantic prompt Wiki.
sdwpf_wiki_prepare() {
    if [[ "${PROMPT_ROUTER}" != "scene_wiki" \
        && "${PROMPT_ROUTER}" != "hybrid_wiki" \
        && "${PROMPT_ROUTER}" != "compositional_wiki" ]]; then
        return 0
    fi
    if [[ -f "${SCENE_WIKI_EMBEDDINGS}" ]] \
        && python -u scripts/check_wind_regime_wiki.py \
            --config "${SCENE_WIKI_CONFIG}" \
            --bundle "${SCENE_WIKI_EMBEDDINGS}"; then
        return 0
    fi
    echo "[WIKI] Embedding bundle is absent or stale; rebuilding frozen anchors."
    python -u scripts/build_wind_regime_wiki.py \
        --config "${SCENE_WIKI_CONFIG}" \
        --output "${SCENE_WIKI_EMBEDDINGS}" \
        --llm_path "${WIKI_LLM_PATH}" \
        --device "${WIKI_BUILD_DEVICE}"
}
