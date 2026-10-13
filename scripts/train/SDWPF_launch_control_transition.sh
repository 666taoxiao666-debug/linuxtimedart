#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh
pointer=outputs/logs/SDWPF/control_transition_latest.txt
if [[ "${1:-}" == --status ]]; then
    [[ -f "${pointer}" ]] || { echo "No control-transition pilot launched."; exit 0; }
    directory="$(< "${pointer}")"
    echo "RESULT_DIR=${directory}"
    for name in status.env progress.json summary.txt; do
        [[ ! -f "${directory}/${name}" ]] || cat "${directory}/${name}"
    done
    exit 0
fi
[[ "${1:-}" == --start || "${1:-}" == --resume ]] && [[ "$#" == 1 ]] || {
    echo "Usage: bash $0 --start|--resume|--status" >&2; exit 2;
}
command -v python >/dev/null || { echo "Activate the timedart environment first." >&2; exit 2; }
mkdir -p outputs/logs/SDWPF
exec 9> outputs/logs/SDWPF/control_transition_launcher.lock
flock -n 9 || { echo "Existing launcher; no duplicate started." >&2; exit 3; }
source_pointer=outputs/logs/SDWPF/residual_drift_audit_latest.txt
[[ -f "${source_pointer}" ]] || { echo "Verified original-train OOF context missing." >&2; exit 2; }
drift="$(< "${source_pointer}")"
if [[ -f "${pointer}" ]]; then
    previous="$(< "${pointer}")"
    [[ "${1}" == --resume ]] || { echo "Existing pilot; use --status or --resume, not a second run." >&2; exit 3; }
    previous_pid="$(cat "${previous}/launcher.pid" 2>/dev/null || true)"
    if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Existing PID=${previous_pid}; no duplicate started." >&2; exit 3
    fi
fi
# CPU-only; defer even CPU work while an unrelated GPU training job is busy.
if command -v nvidia-smi >/dev/null; then
    gpu_jobs="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
    [[ -z "${gpu_jobs}" ]] || { echo "GPU training is busy; defer this pilot." >&2; exit 3; }
fi
if [[ "${1}" == --resume ]]; then
    [[ -f "${pointer}" ]] || { echo "No pilot to resume." >&2; exit 2; }
    SDWPF_LOG_DIR="$(< "${pointer}")"
else
    sdwpf_log_init control_transition "h12_f1_s2024_2x2_control_transition_trainonly_cpu2" launch.log "control_transition_v15_$(date +%Y%m%d_%H%M%S)"
    printf '%s\n' "${SDWPF_LOG_DIR}" > "${pointer}"
fi
nohup env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
    nice -n 10 python -u scripts/train_sdwpf_control_transition.py --result-dir "${SDWPF_LOG_DIR}" \
    --drift-dir "${drift}" --protocol configs/sdwpf_control_transition_protocol_v15.json \
    >> "${SDWPF_LOG_DIR}/launch.log" 2>&1 < /dev/null 9>&- &
printf '%s\n' "$!" > "${SDWPF_LOG_DIR}/launcher.pid"
echo "RESULT_DIR=${SDWPF_LOG_DIR}"
echo "PID=$!"
echo "Check: bash scripts/train/SDWPF_launch_control_transition.sh --status"
