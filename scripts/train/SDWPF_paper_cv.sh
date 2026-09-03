#!/usr/bin/env bash
set -euo pipefail

# Hyper-parameter selection only.  rolling_holdout keeps the final 20% sealed
# and uses three disjoint validation blocks inside the preceding 10%.
for fold in 0 1 2; do
    for seed in 2024 2025 2026; do
        echo "===== CV fold=${fold} seed=${seed} ====="
        FOLD="${fold}" \
        SEED="${seed}" \
        SPLIT=rolling_holdout \
        PRED_LEN="${PRED_LEN:-12}" \
        bash scripts/train/SDWPF_full_pipeline.sh
    done
done

echo "[CV] Completed. Select one configuration from validation metrics only."
echo "[CV] Then train the frozen configuration with SPLIT=time_ratio."
