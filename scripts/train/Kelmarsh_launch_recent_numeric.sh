#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
POINTER="outputs/logs/Kelmarsh/hourly_recent_latest.txt"
case "${1:-}" in
  --status) exec "${PYTHON}" scripts/train_kelmarsh_recent_numeric.py --status ;;
  --resume)
    [[ -s "${POINTER}" ]] || { echo "No v20 pointer" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Prior result directory missing" >&2; exit 2; }
    ;;
  "")
    mkdir -p outputs/logs/Kelmarsh
    [[ ! -s "${POINTER}" ]] || { echo "Existing experiment; use --status or --resume" >&2; exit 2; }
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    RESULT_DIR="$(mktemp -d "${DATE_DIR}/recent24h4h_f0-1-2_s2024_XXXXXX")"
    ;;
  *) echo "Usage: bash scripts/train/Kelmarsh_launch_recent_numeric.sh [--status|--resume]" >&2; exit 2 ;;
esac
echo "RESULT_DIR=${RESULT_DIR}"
[[ -f "${RESULT_DIR}/source_commit.txt" ]] || git rev-parse HEAD > "${RESULT_DIR}/source_commit.txt"
echo "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
"${PYTHON}" -u scripts/train_kelmarsh_recent_numeric.py --output "${RESULT_DIR}" --gpu "${GPU:-0}" 2>&1 | tee -a "${RESULT_DIR}/launch.log"
