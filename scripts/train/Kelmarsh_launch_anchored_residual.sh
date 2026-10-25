#!/usr/bin/env bash
# v26 Kelmarsh anchored-residual launcher (frozen v20 DLinear + zero-initialised TSMixerBasicSmall residual, A0-A4).
#   preflight only:   bash scripts/train/Kelmarsh_launch_anchored_residual.sh --preflight-only [--detach]
#   first run:        bash scripts/train/Kelmarsh_launch_anchored_residual.sh [--detach]
#   status:           bash scripts/train/Kelmarsh_launch_anchored_residual.sh --status
#   resume:           bash scripts/train/Kelmarsh_launch_anchored_residual.sh --resume [--detach]
#   report only:      bash scripts/train/Kelmarsh_launch_anchored_residual.sh --report-only
# A real fit refuses to start without a passed --preflight-only record for the exact code/protocol.
# Own resources only: 2 CPU threads; GPU use is decided per fold/seed pair by the controller. Never stops,
# pauses or renices other processes and never changes device, clock, MPS or network settings.
set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON="${KELMARSH_PYTHON:-python}"
SCRIPT="scripts/train_kelmarsh_anchored_residual.py"
POINTER="outputs/logs/Kelmarsh/anchored_residual_latest.txt"
LAUNCH_LOCK="outputs/logs/Kelmarsh/anchored_residual.launch.lock"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

usage() { echo "Usage: $0 [--status|--resume|--preflight-only|--report-only] [--detach]" >&2; }
MODE="run"; DETACH=0; PASS=()
for arg in "$@"; do
  case "${arg}" in
    --status) MODE="status" ;;
    --resume) MODE="resume"; PASS+=("${arg}") ;;
    --preflight-only) MODE="preflight"; PASS+=("${arg}") ;;
    --report-only) MODE="report"; PASS+=("${arg}") ;;
    --detach) DETACH=1 ;;
    *) usage; exit 2 ;;
  esac
done
mkdir -p outputs/logs/Kelmarsh

if [[ "${MODE}" == "status" ]]; then
  exec "${PYTHON}" "${SCRIPT}" --status
fi

if [[ "${DETACH}" == "1" ]]; then
  mkdir -p outputs/logs/Kelmarsh/launcher
  LOG="outputs/logs/Kelmarsh/launcher/anchored_residual_${MODE}_$(date +%Y%m%d_%H%M%S).log"
  nohup setsid bash "${SELF}" ${PASS[@]+"${PASS[@]}"} > "${LOG}" 2>&1 < /dev/null &
  echo "DETACHED_PID=$! LOG=${LOG}"
  exit 0
fi

# One launcher at a time (the Python controller additionally holds its own flock).
exec 9>>"${LAUNCH_LOCK}"
if ! flock -n 9; then
  echo "[BUSY] another v26 launcher holds ${LAUNCH_LOCK}; nothing started" >&2
  exit 3
fi

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON}" -u "${SCRIPT}" --preflight-only
  exit $?
fi

case "${MODE}" in
  resume|report)
    [[ -s "${POINTER}" ]] || { echo "No v26 pointer; nothing to resume" >&2; exit 2; }
    RESULT_DIR="$(<"${POINTER}")"
    [[ -d "${RESULT_DIR}" ]] || { echo "Registered v26 directory missing: ${RESULT_DIR}" >&2; exit 2; }
    ;;
  run)
    if [[ -s "${POINTER}" ]]; then
      echo "Existing v26 run $(<"${POINTER}"); use --status or --resume (no duplicate study)" >&2
      exit 2
    fi
    DATE_DIR="outputs/logs/Kelmarsh/$(date +%Y%m%d)"
    mkdir -p "${DATE_DIR}"
    RESULT_DIR="$(mktemp -d "${DATE_DIR}/anchored_residual_v26_f0-1-2_XXXXXX")"
    ;;
esac
echo "RESULT_DIR=${RESULT_DIR}"
[[ -f "${RESULT_DIR}/source_commit.txt" ]] || git rev-parse HEAD > "${RESULT_DIR}/source_commit.txt"
echo "${BASHPID}" > "${RESULT_DIR}/launcher.pid"
EXTRA=()
[[ "${MODE}" == "report" ]] && EXTRA+=(--report-only)
{
  echo "[LAUNCH] $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=${MODE} host=$(hostname) commit=$(git rev-parse HEAD)"
  echo "[LAUNCH] command: ${PYTHON} -u ${SCRIPT} --output ${RESULT_DIR} --gpu ${GPU:-0} ${EXTRA[*]:-}"
} | tee -a "${RESULT_DIR}/launch.log"
set +e
"${PYTHON}" -u "${SCRIPT}" --output "${RESULT_DIR}" --gpu "${GPU:-0}" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "${RESULT_DIR}/launch.log"
CODE=${PIPESTATUS[0]}
set -e
echo "[LAUNCHER_EXIT] $(date -u +%Y-%m-%dT%H:%M:%SZ) code=${CODE}" | tee -a "${RESULT_DIR}/launch.log"
exit "${CODE}"
