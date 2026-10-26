#!/usr/bin/env bash
# Generic launcher for frozen Kelmarsh protocol controllers (v29 onward; reuse it instead of adding a launcher per version).
#   preflight only:   bash scripts/train/Kelmarsh_launch_frozen_protocol.sh <controller.py> --preflight-only [--detach]
#   first run:        bash scripts/train/Kelmarsh_launch_frozen_protocol.sh <controller.py> [--detach]
#   status:           bash scripts/train/Kelmarsh_launch_frozen_protocol.sh <controller.py> --status
#   resume:           bash scripts/train/Kelmarsh_launch_frozen_protocol.sh <controller.py> --resume [--detach]
#   report only:      bash scripts/train/Kelmarsh_launch_frozen_protocol.sh <controller.py> --report-only
# e.g. <controller.py> = scripts/train_kelmarsh_operating_state.py (v29). The controller must provide --print-tag, --status,
# --preflight-only and --output DIR [--report-only] --gpu N, and keep its pointer at outputs/logs/Kelmarsh/<tag>_latest.txt.
# A real fit refuses to start without a passed --preflight-only record for the exact code/protocol (checked by the
# controller). Own resources only: 2 CPU threads; never stops, pauses or renices other processes and never changes
# device, clock, MPS or network settings.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

usage() { echo "Usage: $0 <controller.py> [--status|--resume|--preflight-only|--report-only] [--detach]" >&2; }
[[ $# -ge 1 ]] || { usage; exit 2; }
SCRIPT="$1"; shift
[[ -f "${SCRIPT}" && "${SCRIPT}" == scripts/train_kelmarsh_*.py ]] || { echo "Unknown controller: ${SCRIPT}" >&2; exit 2; }
MODE="run"; DETACH=0; PASS=(); MODE_COUNT=0
for arg in "$@"; do
  case "${arg}" in
    --status|--resume|--preflight-only|--report-only) MODE_COUNT=$((MODE_COUNT + 1)) ;;
  esac
  case "${arg}" in
    --status) MODE="status" ;;
    --resume) MODE="resume"; PASS+=("${arg}") ;;
    --preflight-only) MODE="preflight"; PASS+=("${arg}") ;;
    --report-only) MODE="report"; PASS+=("${arg}") ;;
    --detach) DETACH=1 ;;
    *) usage; exit 2 ;;
  esac
done
[[ "${MODE_COUNT}" -le 1 ]] || { echo "Choose exactly one operation mode" >&2; exit 2; }
mkdir -p outputs/logs/Kelmarsh

if [[ "${MODE}" == "status" ]]; then
  exec "${PYTHON}" "${SCRIPT}" --status
fi

TAG="$("${PYTHON}" "${SCRIPT}" --print-tag | tail -n 1)"
[[ "${TAG}" =~ ^[a-z0-9_]+$ ]] || { echo "Invalid controller tag: ${TAG}" >&2; exit 2; }
POINTER="outputs/logs/Kelmarsh/${TAG}_latest.txt"
LAUNCH_LOCK="outputs/logs/Kelmarsh/${TAG}.launch.lock"

if [[ "${DETACH}" == "1" ]]; then
  mkdir -p outputs/logs/Kelmarsh/launcher
  LOG="outputs/logs/Kelmarsh/launcher/${TAG}_${MODE}_$(date +%Y%m%d_%H%M%S).log"
  nohup setsid bash "${SELF}" "${SCRIPT}" ${PASS[@]+"${PASS[@]}"} > "${LOG}" 2>&1 < /dev/null &
  echo "DETACHED_PID=$! LOG=${LOG}"
  exit 0
fi

# One launcher per tag at a time (the Python controller additionally holds its own flock).
exec 9>>"${LAUNCH_LOCK}"
if ! flock -n 9; then
  echo "[BUSY] another ${TAG} launcher holds ${LAUNCH_LOCK}; nothing started" >&2
  exit 3
fi

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON}" -u "${SCRIPT}" --preflight-only
  exit $?
fi

case "${MODE}" in
  resume|report)
    [[ -s "${POINTER}" ]] || { echo "No ${TAG} pointer; nothing to resume" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Registered ${TAG} directory missing: ${RESULT_DIR}" >&2; exit 2; }
    ;;
  run)
    if [[ -s "${POINTER}" ]]; then
      echo "Existing ${TAG} run $(<"${POINTER}"); use --status or --resume (no duplicate study)" >&2
      exit 2
    fi
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    RESULT_DIR="$(mktemp -d "${DATE_DIR}/${TAG}_XXXXXX")"
    ;;
esac
echo "RESULT_DIR=${RESULT_DIR}"
[[ -f "${RESULT_DIR}/source_commit.txt" ]] || git rev-parse HEAD > "${RESULT_DIR}/source_commit.txt"
echo "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
EXTRA=()
[[ "${MODE}" == "report" ]] && EXTRA+=(--report-only)
{
  echo "[LAUNCH] $(date -u +%Y-%m-%dT%H:%M:%SZ) tag=${TAG} mode=${MODE} host=$(hostname) commit=$(git rev-parse HEAD)"
  echo "[LAUNCH] command: ${PYTHON} -u ${SCRIPT} --output ${RESULT_DIR} --gpu ${GPU:-0} ${EXTRA[*]:-}"
} | tee -a "${RESULT_DIR}/launch.log"
set +e
"${PYTHON}" -u "${SCRIPT}" --output "${RESULT_DIR}" --gpu "${GPU:-0}" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "${RESULT_DIR}/launch.log"
CODE=${PIPESTATUS[0]}
set -e
echo "[LAUNCHER_EXIT] $(date -u +%Y-%m-%dT%H:%M:%SZ) code=${CODE}" | tee -a "${RESULT_DIR}/launch.log"
exit "${CODE}"
