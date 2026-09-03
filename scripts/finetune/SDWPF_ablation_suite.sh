#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PRETRAIN_RUN_ID:-}" ]]; then
    echo "PRETRAIN_RUN_ID must identify the matched fold/seed checkpoint." >&2
    exit 2
fi

COMMON_ID="${ABLATION_ID:-ablation_h${PRED_LEN:-12}_${SPLIT:-rolling_holdout}_f${FOLD:-0}_s${SEED:-2024}_$(date +%Y%m%d_%H%M%S)}"

echo "[ABLATION 1/4] full pretrained method"
RUN_ID="${COMMON_ID}_full" \
ALLOW_RANDOM=0 CHANNEL_PRIOR=1 OP_CONTEXT=1 REVIN_KEEP_WIND=1 \
bash scripts/finetune/SDWPF_ablation_prompt.sh

echo "[ABLATION 2/4] matched random initialization"
RUN_ID="${COMMON_ID}_random" \
ALLOW_RANDOM=1 CHANNEL_PRIOR=1 OP_CONTEXT=1 REVIN_KEEP_WIND=1 \
bash scripts/finetune/SDWPF_ablation_prompt.sh

echo "[ABLATION 3/4] pretrained with uniform channel initialization"
RUN_ID="${COMMON_ID}_uniform_prior" \
ALLOW_RANDOM=0 CHANNEL_PRIOR=0 OP_CONTEXT=1 REVIN_KEEP_WIND=1 \
bash scripts/finetune/SDWPF_ablation_prompt.sh

echo "[ABLATION 4/4] pretrained without operating-point context"
RUN_ID="${COMMON_ID}_no_context" \
ALLOW_RANDOM=0 CHANNEL_PRIOR=1 OP_CONTEXT=0 REVIN_KEEP_WIND=1 \
bash scripts/finetune/SDWPF_ablation_prompt.sh
