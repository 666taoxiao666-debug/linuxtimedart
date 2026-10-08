#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Protocol v2: change only the Wiki decision layer.  Physical evidence and the
# original expected-gain gate remain mandatory; a separately supervised
# train-only risk head may veto candidates at evaluation with a fixed 0.5 cut.
export UTILITY_LAUNCH_TASK=risk_veto_wiki
export UTILITY_PROTOCOL_FILE="configs/sdwpf_wiki_risk_veto_protocol_v2.json"
export UTILITY_PROTOCOL_SHA256="$(sha256sum "${UTILITY_PROTOCOL_FILE}" | awk '{print $1}')"
export UTILITY_HARM_VETO=1
export UTILITY_HARM_THRESHOLD="${UTILITY_HARM_THRESHOLD:-0.5}"
export UTILITY_HARM_CLASSIFIER_WEIGHT="${UTILITY_HARM_CLASSIFIER_WEIGHT:-1.0}"
export UTILITY_HARM_LOSS_WEIGHT="${UTILITY_HARM_LOSS_WEIGHT:-0.5}"
export UTILITY_DECISION_LOSS_WEIGHT="${UTILITY_DECISION_LOSS_WEIGHT:-0.7}"
export UTILITY_RANKING_LOSS_WEIGHT="${UTILITY_RANKING_LOSS_WEIGHT:-0.3}"
export UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.003}"
export FOLDS="${FOLDS:-1}"
export SEEDS="${SEEDS:-2024}"

echo "[PROTOCOL] ${UTILITY_PROTOCOL_FILE} sha256=${UTILITY_PROTOCOL_SHA256}"

exec bash scripts/train/SDWPF_launch_factorized_wiki.sh "$@"
