#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/numeric_state_latest.txt"
if [[ "${1:-}" == --status && $# -eq 1 ]]; then
  [[ -s "${POINTER}" ]] || { echo "STATUS=NOT_STARTED"; exit 0; }
  RESULT_DIR="$(<"${POINTER}")"
  printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
  [[ ! -f "${RESULT_DIR}/status.json" ]] || cat "${RESULT_DIR}/status.json"
  [[ ! -f "${RESULT_DIR}/progress.json" ]] || cat "${RESULT_DIR}/progress.json"
  exit 0
fi
RESUME=0
if [[ $# -gt 1 ]]; then echo "Only --resume or --status accepted" >&2; exit 2; fi
if [[ $# -eq 1 ]]; then
  [[ "$1" == --resume ]] || { echo "Only --resume or --status accepted" >&2; exit 2; }
  RESUME=1
fi
mkdir -p outputs/logs/Kelmarsh
exec 9>outputs/logs/Kelmarsh/numeric_state_launch.lock
flock -n 9 || { echo "[BUSY] State suite already running" >&2; exit 2; }
if [[ "${RESUME}" -eq 1 ]]; then
  [[ -s "${POINTER}" ]] || { echo "No state run to resume" >&2; exit 2; }
  RESULT_DIR="$(<"${POINTER}")"
  [[ -d "${RESULT_DIR}" ]] || exit 2
else
  [[ ! -s "${POINTER}" ]] || { echo "Existing run; use --resume or --status" >&2; exit 2; }
  DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
  mkdir -p "${DATE_DIR}"
  INDEX=1
  while true; do
    printf -v NUMBER '%03d' "${INDEX}"
    RESULT_DIR="${DATE_DIR}/${NUMBER}_numeric_state_h2_f1_s2024_shrunk"
    if mkdir "${RESULT_DIR}" 2>/dev/null; then break; fi
    INDEX=$((INDEX+1))
  done
  printf '%s\n' "${RESULT_DIR}" > "${POINTER}"
fi
printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf '%s\n' "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
git rev-parse HEAD >> "${RESULT_DIR}/source_commits.log"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
"${PYTHON}" -u scripts/train_kelmarsh_numeric_state.py \
  --protocol configs/kelmarsh_numeric_state_protocol_v18.json --output "${RESULT_DIR}" \
  2>&1 | tee -a "${RESULT_DIR}/launch.log"
