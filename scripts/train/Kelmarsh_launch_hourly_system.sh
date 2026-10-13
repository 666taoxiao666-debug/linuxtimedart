#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/hourly_system_latest.txt"
case "${1:-}" in
  --status)
    exec "${PYTHON}" scripts/train_kelmarsh_hourly.py --status
    ;;
  --resume)
    [[ -s "${POINTER}" ]] || { echo "No prior Kelmarsh result pointer" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Prior result directory missing" >&2; exit 2; }
    ;;
  "")
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    INDEX=1
    while true; do
      printf -v NUMBER '%03d' "${INDEX}"
      RESULT_DIR="${DATE_DIR}/${NUMBER}_hourly_system_h2_f0-1-2_s2024-2025-2026"
      if mkdir "${RESULT_DIR}" 2>/dev/null; then break; fi
      INDEX=$((INDEX+1))
    done
    ;;
  *)
    echo "Usage: bash scripts/train/Kelmarsh_launch_hourly_system.sh [--status|--resume]" >&2
    exit 2
    ;;
esac
echo "RESULT_DIR=${RESULT_DIR}"
if [[ "${1:-}" == "--resume" ]]; then
  git rev-parse HEAD >> "${RESULT_DIR}/resume_source_commits.log"
fi
if [[ ! -f "${RESULT_DIR}/source_commit.txt" ]]; then
  git rev-parse HEAD > "${RESULT_DIR}/source_commit.txt"
fi
echo "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
"${PYTHON}" -u scripts/train_kelmarsh_hourly.py \
  --protocol configs/kelmarsh_hourly_protocol_v1.json \
  --output "${RESULT_DIR}" --gpu "${GPU:-0}" 2>&1 | tee -a "${RESULT_DIR}/launch.log"
