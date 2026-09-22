#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Original train only: earlier target windows fit corrections, later ones fit
# the soft gate. Validation selects hard-routing checkpoints; test is untouched.
export UTILITY_ADAPTER_MODE=calibrated_evidence
export UTILITY_CALIBRATION_FRACTION="${UTILITY_CALIBRATION_FRACTION:-0.2}"
export UTILITY_EVENT_MAX_SCALE="${UTILITY_EVENT_MAX_SCALE:-0.2}"
export UTILITY_COMPOSITION_MAX_SCALE="${UTILITY_COMPOSITION_MAX_SCALE:-0.05}"
export UTILITY_ADAPTER_WARMUP_EPOCHS="${UTILITY_ADAPTER_WARMUP_EPOCHS:-2}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-6}" PATIENCE="${PATIENCE:-3}"
export UTILITY_LEARNING_RATE="${UTILITY_LEARNING_RATE:-0.00003}"
export UTILITY_GATE_TEMPERATURE="${UTILITY_GATE_TEMPERATURE:-0.05}"
export UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.001}"
exec bash scripts/train/SDWPF_launch_hierarchical_wiki.sh "$@"
