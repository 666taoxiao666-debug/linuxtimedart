#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/certified_fallback_latest.txt"
if [[ "${1:-}" == --status ]]; then
  [[ -s "${POINTER}" ]] || { echo "STATUS=NOT_STARTED"; exit 0; }
  RESULT_DIR="$(<"${POINTER}")"
  printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
  [[ ! -f "${RESULT_DIR}/status.json" ]] || cat "${RESULT_DIR}/status.json"
  [[ ! -f "${RESULT_DIR}/progress.json" ]] || cat "${RESULT_DIR}/progress.json"
  exit 0
fi
[[ $# -le 1 && ( "${1:-}" == "" || "${1:-}" == --resume ) ]] || {
  echo "Usage: bash scripts/train/Kelmarsh_launch_certified_fallback.sh [--status|--resume]" >&2; exit 2;
}
mkdir -p outputs/logs/Kelmarsh
exec 9>outputs/logs/Kelmarsh/certified_fallback_launch.lock
flock -n 9 || { echo "[BUSY] Certified fallback already running/queued" >&2; exit 2; }
if [[ "${1:-}" == --resume ]]; then
  [[ -s "${POINTER}" ]] || { echo "No v7 experiment to resume" >&2; exit 2; }
  RESULT_DIR="$(<"${POINTER}")"
  [[ -d "${RESULT_DIR}" ]] || exit 2
else
  [[ ! -s "${POINTER}" ]] || { echo "Existing v7 experiment; use --status or --resume" >&2; exit 2; }
  DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
  mkdir -p "${DATE_DIR}"
  INDEX=1
  while true; do
    printf -v NUMBER '%03d' "${INDEX}"
    RESULT_DIR="${DATE_DIR}/${NUMBER}_certified_fallback_h2_f1_s2024_ep20"
    if mkdir "${RESULT_DIR}" 2>/dev/null; then break; fi
    INDEX=$((INDEX+1))
  done
  printf '%s\n' "${RESULT_DIR}" > "${POINTER}"
fi
printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf '%s\n' "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
git rev-parse HEAD >> "${RESULT_DIR}/source_commits.log"
"${PYTHON}" -u scripts/train_kelmarsh_certified_fallback.py \
  --protocol configs/kelmarsh_certified_fallback_protocol_v7.json --output "${RESULT_DIR}" \
  2>&1 | tee -a "${RESULT_DIR}/launch.log"
