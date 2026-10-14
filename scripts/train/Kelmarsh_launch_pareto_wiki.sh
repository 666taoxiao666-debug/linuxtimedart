#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/pareto_wiki_latest.txt"
case "${1:-}" in
  --status)
    [[ -s "${POINTER}" ]] || { echo "STATUS=NOT_STARTED"; exit 0; }
    RESULT_DIR="$(<"${POINTER}")"
    printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
    [[ ! -f "${RESULT_DIR}/status.env" ]] || cat "${RESULT_DIR}/status.env"
    [[ ! -f "${RESULT_DIR}/progress.json" ]] || cat "${RESULT_DIR}/progress.json"
    exit 0
    ;;
  --resume|--evaluate)
    [[ -s "${POINTER}" ]] || { echo "No prior Pareto Wiki run" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Prior directory missing" >&2; exit 2; }
    ;;
  "")
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    INDEX=1
    while true; do
      printf -v NUMBER '%03d' "${INDEX}"
      RESULT_DIR="${DATE_DIR}/${NUMBER}_pareto_wiki_h2_f0-1-2_s2024_frozen_base"
      if mkdir "${RESULT_DIR}" 2>/dev/null; then break; fi
      INDEX=$((INDEX+1))
    done
    printf '%s\n' "${RESULT_DIR}" > "${POINTER}"
    ;;
  *) echo "Usage: bash scripts/train/Kelmarsh_launch_pareto_wiki.sh [--status|--resume|--evaluate]" >&2; exit 2 ;;
esac
printf 'RESULT_DIR=%s\n' "${RESULT_DIR}"
printf '%s\n' "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
git rev-parse HEAD >> "${RESULT_DIR}/source_commits.log"
ARGUMENTS=(--protocol configs/kelmarsh_pareto_protocol_v2.json --output "${RESULT_DIR}" --gpu "${GPU:-0}")
if [[ "${1:-}" == "--evaluate" ]]; then
  # Separate, explicit frozen outer evaluation. Default remains train-only.
  touch "${RESULT_DIR}/outer_evaluation_requested"
fi
if [[ -f "${RESULT_DIR}/outer_evaluation_requested" ]]; then
  ARGUMENTS+=(--outer-validation)
fi
"${PYTHON}" -u scripts/eval_kelmarsh_pareto_wiki.py "${ARGUMENTS[@]}" 2>&1 | tee -a "${RESULT_DIR}/launch.log"
