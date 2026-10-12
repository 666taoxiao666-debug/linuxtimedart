#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh
pointer=outputs/logs/SDWPF/knowledge_transport_latest.txt
if [[ "${1:-}" == --status ]]; then
    [[ -f "${pointer}" ]] || { echo "No knowledge transport pilot launched."; exit 2; }
    directory="$(< "${pointer}")"
    echo "RESULT_DIR=${directory}"
    for filename in status.env progress.json result.json; do
        [[ ! -f "${directory}/${filename}" ]] || cat "${directory}/${filename}"
    done
    exit 0
fi
[[ -z "${1:-}" || "${1:-}" == --resume ]] || { echo "Usage: bash $0 [--status|--resume]" >&2; exit 2; }
mkdir -p outputs/logs/SDWPF
exec 9> outputs/logs/SDWPF/knowledge_transport_launcher.lock
flock -n 9 || { echo "Another knowledge transport launcher is active." >&2; exit 3; }
[[ -f outputs/logs/SDWPF/residual_drift_audit_latest.txt ]] || { echo "Verified training history missing." >&2; exit 2; }
drift="$(< outputs/logs/SDWPF/residual_drift_audit_latest.txt)"
if [[ -f "${pointer}" ]]; then
    previous="$(< "${pointer}")"
    previous_pid="$(cat "${previous}/launcher.pid" 2>/dev/null || true)"
    if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Existing transport PID=${previous_pid}; no duplicate started." >&2; exit 3
    fi
    [[ "${1:-}" == --resume ]] || { echo "Existing pilot; inspect --status or --resume." >&2; exit 3; }
fi
if [[ "${1:-}" == --resume ]]; then
    [[ -f "${pointer}" ]] || { echo "No pilot to resume." >&2; exit 2; }
    SDWPF_LOG_DIR="$(< "${pointer}")"
else
    sdwpf_log_init knowledge_transport "h12_f1_s2024_expertbound_time50-25-25_turbmod5_cpu2" launch.log "knowledge_transport_v12_$(date +%Y%m%d_%H%M%S)"
    printf '%s\n' "${SDWPF_LOG_DIR}" > "${pointer}"
fi
nohup env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
    nice -n 10 python -u scripts/train_sdwpf_knowledge_transport.py --result-dir "${SDWPF_LOG_DIR}" \
    --drift-dir "${drift}" >> "${SDWPF_LOG_DIR}/launch.log" 2>&1 < /dev/null 9>&- &
printf '%s\n' "$!" > "${SDWPF_LOG_DIR}/launcher.pid"
echo "RESULT_DIR=${SDWPF_LOG_DIR}"
echo "PID=$!"
echo "Check: bash scripts/train/SDWPF_launch_knowledge_transport.sh --status"
