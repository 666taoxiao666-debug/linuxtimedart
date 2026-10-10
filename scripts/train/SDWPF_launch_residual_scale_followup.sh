#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh
pointer=outputs/logs/SDWPF/residual_scale_followup_latest.txt
if [[ "${1:-}" == --status ]]; then
    [[ -f "${pointer}" ]] || { echo "No residual scale follow-up launched."; exit 2; }
    directory="$(< "${pointer}")"
    echo "RESULT_DIR=${directory}"
    for filename in status.env progress.json; do
        [[ ! -f "${directory}/${filename}" ]] || cat "${directory}/${filename}"
    done
    [[ ! -f "${directory}/result.json" ]] || echo "REPORT=${directory}/result.json"
    exit 0
fi
[[ -z "${1:-}" || "${1:-}" == --resume ]] || { echo "Usage: bash $0 [--status|--resume]" >&2; exit 2; }
mkdir -p outputs/logs/SDWPF
exec 9> outputs/logs/SDWPF/residual_scale_followup_launcher.lock
flock -n 9 || { echo "Another follow-up launcher is active." >&2; exit 3; }
[[ -f outputs/logs/SDWPF/residual_scale_latest.txt ]] || { echo "Completed v7 calibration missing." >&2; exit 2; }
calibration="$(< outputs/logs/SDWPF/residual_scale_latest.txt)"
[[ -f "${calibration}/result.json" ]] || { echo "Calibration not complete." >&2; exit 2; }
if [[ -f "${pointer}" ]]; then
    previous="$(< "${pointer}")"
    previous_pid="$(cat "${previous}/launcher.pid" 2>/dev/null || true)"
    if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Existing follow-up PID=${previous_pid}; no duplicate started." >&2; exit 3
    fi
    [[ "${1:-}" == --resume ]] || { echo "Existing follow-up; inspect --status or --resume." >&2; exit 3; }
fi
if [[ "${1:-}" == --resume ]]; then
    [[ -f "${pointer}" ]] || { echo "No follow-up to resume." >&2; exit 2; }
    SDWPF_LOG_DIR="$(< "${pointer}")"
    if [[ -f "${SDWPF_LOG_DIR}/forecast_accuracy_overview.png" ]] && grep -q '^STATUS=COMPLETED$' "${SDWPF_LOG_DIR}/status.env"; then
        echo "Follow-up already complete; nothing started."; exit 0
    fi
else
    sdwpf_log_init residual_scale_followup "h12_f1_s2024_frozen_cpu_outerval" launch.log "residual_scale_followup_$(date +%Y%m%d_%H%M%S)"
    printf '%s\n' "${SDWPF_LOG_DIR}" > "${pointer}"
fi
nohup env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
    nice -n 10 python -u scripts/confirm_sdwpf_residual_scale.py --result-dir "${SDWPF_LOG_DIR}" \
    --calibration-dir "${calibration}" >> "${SDWPF_LOG_DIR}/launch.log" 2>&1 < /dev/null 9>&- &
printf '%s\n' "$!" > "${SDWPF_LOG_DIR}/launcher.pid"
echo "RESULT_DIR=${SDWPF_LOG_DIR}"
echo "PID=$!"
echo "Check: bash scripts/train/SDWPF_launch_residual_scale_followup.sh --status"
