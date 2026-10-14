#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/contextual_expert_latest.txt"
case "${1:-}" in
  --status)
    [[ -s "${POINTER}" ]] || { echo "STATUS=NOT_STARTED"; exit 0; }
    RESULT_DIR="$(<"${POINTER}")"
    printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
    [[ ! -f "${RESULT_DIR}/status.json" ]] || cat "${RESULT_DIR}/status.json"
    [[ ! -f "${RESULT_DIR}/progress.json" ]] || cat "${RESULT_DIR}/progress.json"
    exit 0
    ;;
  --resume)
    [[ -s "${POINTER}" ]] || { echo "No contextual experiment to resume" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Prior result directory missing" >&2; exit 2; }
    ;;
  "")
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    INDEX=1
    while true; do
      printf -v NUMBER '%03d' "${INDEX}"
      RESULT_DIR="${DATE_DIR}/${NUMBER}_contextual_expert_h2_f0-1-2_s2024_ep20"
      if mkdir "${RESULT_DIR}" 2>/dev/null; then break; fi
      INDEX=$((INDEX+1))
    done
    printf '%s\n' "${RESULT_DIR}" > "${POINTER}"
    ;;
  *) echo "Usage: bash scripts/train/Kelmarsh_launch_contextual_expert.sh [--status|--resume]" >&2; exit 2 ;;
esac
printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf '%s\n' "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
git rev-parse HEAD >> "${RESULT_DIR}/source_commits.log"
"${PYTHON}" -u scripts/train_kelmarsh_contextual_expert.py --protocol configs/kelmarsh_contextual_expert_protocol_v4.json --output "${RESULT_DIR}" --gpu "${GPU:-0}" 2>&1 | tee -a "${RESULT_DIR}/launch.log"
