#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Paired intervention experiment: reuse the same pretrain and frozen trend
# checkpoints, but train the gate to penalize harmful event/horizon actions.
# All labels for this objective come from the original training partition.
export UTILITY_LAUNCH_TASK=event_gain_wiki
export UTILITY_HARM_LOSS_WEIGHT="${UTILITY_HARM_LOSS_WEIGHT:-0.5}"
export UTILITY_DECISION_LOSS_WEIGHT="${UTILITY_DECISION_LOSS_WEIGHT:-0.7}"
export UTILITY_RANKING_LOSS_WEIGHT="${UTILITY_RANKING_LOSS_WEIGHT:-0.3}"
export UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.003}"
export FOLDS="${FOLDS:-0 1 2}"
export SEEDS="${SEEDS:-2024 2025 2026}"
exec bash scripts/train/SDWPF_launch_factorized_wiki.sh "$@"
