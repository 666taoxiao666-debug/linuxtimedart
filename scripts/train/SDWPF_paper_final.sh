#!/usr/bin/env bash
set -euo pipefail

# Run only after every hyper-parameter and ablation choice is frozen.
# time_ratio uses 70% train, 10% validation and the final 20% as test.
PRED_LEN="${PRED_LEN:-12}"
for seed in 2024 2025 2026; do
    echo "===== FINAL TRAIN seed=${seed}; test remains sealed ====="
    SEED="${seed}" \
    FOLD=0 \
    N_FOLDS=1 \
    SPLIT=time_ratio \
    PRED_LEN="${PRED_LEN}" \
    PIPELINE_ID="final_h${PRED_LEN}_time_ratio_s${seed}_$(date +%Y%m%d_%H%M%S)" \
    bash scripts/train/SDWPF_full_pipeline.sh
done

echo "[FINAL TRAIN] Three predetermined seed checkpoints are ready."
echo "[FINAL TRAIN] Evaluate each checkpoint exactly once with the command printed above."
