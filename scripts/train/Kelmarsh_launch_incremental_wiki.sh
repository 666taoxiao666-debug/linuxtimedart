#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/incremental_wiki_latest.txt"
if [[ "${1:-}" == --status ]]; then
  [[ -s "${POINTER}" ]] || { echo "STATUS=NOT_STARTED"; exit 0; }
  RESULT_DIR="$(<"${POINTER}")"
  printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
  [[ ! -f "${RESULT_DIR}/status.json" ]] || cat "${RESULT_DIR}/status.json"
  [[ ! -f "${RESULT_DIR}/progress.json" ]] || cat "${RESULT_DIR}/progress.json"
  exit 0
fi
[[ $# -le 1 && ( "${1:-}" == "" || "${1:-}" == --resume ) ]] || {
  echo "Usage: bash scripts/train/Kelmarsh_launch_incremental_wiki.sh [--status|--resume]" >&2; exit 2;
}
mkdir -p outputs/logs/Kelmarsh
exec 9>outputs/logs/Kelmarsh/incremental_wiki_launch.lock
flock -n 9 || { echo "[BUSY] Incremental Wiki already running/queued; do not duplicate" >&2; exit 2; }
if [[ "${1:-}" == --resume ]]; then
  [[ -s "${POINTER}" ]] || { echo "No incremental run to resume" >&2; exit 2; }
  RESULT_DIR="$(<"${POINTER}")"
  [[ -d "${RESULT_DIR}" ]] || exit 2
else
  if [[ -s "${POINTER}" ]]; then
    echo "An incremental experiment already exists; use --status or --resume, not a new duplicate" >&2
    exit 2
  fi
  DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
  mkdir -p "${DATE_DIR}"
  INDEX=1
  while true; do
    printf -v NUMBER '%03d' "${INDEX}"
    RESULT_DIR="${DATE_DIR}/${NUMBER}_incremental_wiki_h2_f1_s2024_ep20"
    if mkdir "${RESULT_DIR}" 2>/dev/null; then break; fi
    INDEX=$((INDEX+1))
  done
  printf '%s\n' "${RESULT_DIR}" > "${POINTER}"
fi
printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf '%s\n' "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
git rev-parse HEAD >> "${RESULT_DIR}/source_commits.log"
"${PYTHON}" -u scripts/train_kelmarsh_incremental_wiki.py \
  --protocol configs/kelmarsh_incremental_wiki_protocol_v6.json --output "${RESULT_DIR}" \
  2>&1 | tee -a "${RESULT_DIR}/launch.log"
