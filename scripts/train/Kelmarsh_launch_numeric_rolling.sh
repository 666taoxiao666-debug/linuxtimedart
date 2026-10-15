#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/numeric_rolling_latest.txt"
if [[ "${1:-}" == --status && $# -eq 1 ]]; then
  [[ -s "${POINTER}" ]] || { echo "STATUS=NOT_STARTED"; exit 0; }
  RESULT_DIR="$(<"${POINTER}")"
  printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
  [[ ! -f "${RESULT_DIR}/status.json" ]] || cat "${RESULT_DIR}/status.json"
  [[ ! -f "${RESULT_DIR}/progress.json" ]] || cat "${RESULT_DIR}/progress.json"
  exit 0
fi
RESUME=0
RESOURCE_ARGS=()
for ARG in "$@"; do
  case "${ARG}" in
    --resume) RESUME=1 ;;
    --shared) RESOURCE_ARGS=() ;;
    --idle-only) RESOURCE_ARGS=(--idle-only) ;;
    *) echo "Usage: bash scripts/train/Kelmarsh_launch_numeric_rolling.sh [--status|--resume] [--shared|--idle-only]" >&2; exit 2 ;;
  esac
done
mkdir -p outputs/logs/Kelmarsh
exec 9>outputs/logs/Kelmarsh/numeric_rolling_launch.lock
flock -n 9 || { echo "[BUSY] Numerical rolling suite already running" >&2; exit 2; }
if [[ "${RESUME}" -eq 1 ]]; then
  [[ -s "${POINTER}" ]] || { echo "No rolling run to resume" >&2; exit 2; }
  RESULT_DIR="$(<"${POINTER}")"
  [[ -d "${RESULT_DIR}" ]] || exit 2
else
  [[ ! -s "${POINTER}" ]] || { echo "Existing run; use --resume or --status" >&2; exit 2; }
  DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
  mkdir -p "${DATE_DIR}"
  INDEX=1
  while true; do
    printf -v NUMBER '%03d' "${INDEX}"
    RESULT_DIR="${DATE_DIR}/${NUMBER}_numeric_rolling_h2_f1_s2024_o70-80-90_ep20"
    if mkdir "${RESULT_DIR}" 2>/dev/null; then break; fi
    INDEX=$((INDEX+1))
  done
  printf '%s\n' "${RESULT_DIR}" > "${POINTER}"
fi
printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf '%s\n' "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
git rev-parse HEAD >> "${RESULT_DIR}/source_commits.log"
"${PYTHON}" -u scripts/train_kelmarsh_numeric_rolling.py \
  --protocol configs/kelmarsh_numeric_rolling_protocol_v12.json --output "${RESULT_DIR}" \
  "${RESOURCE_ARGS[@]}" 2>&1 | tee -a "${RESULT_DIR}/launch.log"
