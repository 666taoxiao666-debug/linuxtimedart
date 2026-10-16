#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/hourly_24h4h_latest.txt"
case "${1:-}" in
  --status) exec "${PYTHON}" scripts/train_kelmarsh_24h4h.py --status ;;
  --resume)
    [[ -s "${POINTER}" ]] || { echo "No prior 24h4h result pointer" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Prior result directory missing" >&2; exit 2; }
    ;;
  "")
    mkdir -p outputs/logs/Kelmarsh
    if [[ -s "${POINTER}" ]]; then
      echo "24h4h experiment already registered; use --status or --resume, never duplicate" >&2
      exit 2
    fi
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    RESULT_DIR="$(mktemp -d "${DATE_DIR}/24h4h_f0-1-2_s2024_XXXXXX")"
    ;;
  *) echo "Usage: bash scripts/train/Kelmarsh_launch_24h4h.sh [--status|--resume]" >&2; exit 2 ;;
esac
echo "RESULT_DIR=${RESULT_DIR}"
if [[ ! -f "${RESULT_DIR}/source_commit.txt" ]]; then
  git rev-parse HEAD > "${RESULT_DIR}/source_commit.txt"
fi
echo "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
"${PYTHON}" -u scripts/train_kelmarsh_24h4h.py --output "${RESULT_DIR}" --gpu "${GPU:-0}" 2>&1 | tee -a "${RESULT_DIR}/launch.log"
