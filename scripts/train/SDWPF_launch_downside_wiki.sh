#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Protocol v3: replace the binary hard veto with a continuous train-only
# severity estimate.  Routing uses expected gain minus expected positive
# excess error; physical evidence and the existing abstention threshold remain
# mandatory.  The fixed unit penalty has no validation-tuned cutoff.
export UTILITY_LAUNCH_TASK=downside_wiki
export UTILITY_PROTOCOL_FILE="configs/sdwpf_wiki_downside_protocol_v3.json"
export UTILITY_PROTOCOL_SHA256="$(sha256sum "${UTILITY_PROTOCOL_FILE}" | awk '{print $1}')"
export UTILITY_HARM_VETO=0
export UTILITY_HARM_CLASSIFIER_WEIGHT=0
export UTILITY_DOWNSIDE_GUARD=1
export UTILITY_DOWNSIDE_WEIGHT="${UTILITY_DOWNSIDE_WEIGHT:-1.0}"
export UTILITY_DOWNSIDE_LOSS_WEIGHT="${UTILITY_DOWNSIDE_LOSS_WEIGHT:-1.0}"
export UTILITY_HARM_LOSS_WEIGHT="${UTILITY_HARM_LOSS_WEIGHT:-0.5}"
export UTILITY_DECISION_LOSS_WEIGHT="${UTILITY_DECISION_LOSS_WEIGHT:-0.7}"
export UTILITY_RANKING_LOSS_WEIGHT="${UTILITY_RANKING_LOSS_WEIGHT:-0.3}"
export UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.003}"
export FOLDS="${FOLDS:-1}"
export SEEDS="${SEEDS:-2024}"

echo "[PROTOCOL] ${UTILITY_PROTOCOL_FILE} sha256=${UTILITY_PROTOCOL_SHA256}"

exec bash scripts/train/SDWPF_launch_factorized_wiki.sh "$@"
