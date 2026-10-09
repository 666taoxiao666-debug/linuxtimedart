#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh
pointer=outputs/logs/SDWPF/physics_norm_latest.txt
if [[ "${1:-}" == --status ]]; then
    [[ -f "${pointer}" ]] || { echo "No normalization pilot launched."; exit 2; }
    directory="$(< "${pointer}")"
    echo "RESULT_DIR=${directory}"
    [[ ! -f "${directory}/status.env" ]] || cat "${directory}/status.env"
    for filename in progress.json result.json; do
        [[ ! -f "${directory}/${filename}" ]] || { echo "${filename}:"; cat "${directory}/${filename}"; }
    done
    exit 0
fi
[[ -z "${1:-}" || "${1:-}" == --resume ]] || { echo "Usage: bash $0 [--status|--resume]" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || { echo "GPU occupied." >&2; exit 3; }
source_v5=outputs/logs/SDWPF/20261009/003_accuracy_tuning_h12_f1_s2024_inner4_mix0.2-0.35-0.5_ep8_trainonly
[[ -f "${source_v5}/result.json" ]] || { echo "Completed v5 source missing." >&2; exit 2; }
if [[ -f "${pointer}" ]]; then
    previous="$(< "${pointer}")"
    previous_pid="$(cat "${previous}/launcher.pid" 2>/dev/null || true)"
    if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Existing pilot PID=${previous_pid}; no duplicate started." >&2; exit 3
    fi
    [[ "${1:-}" == --resume ]] || { echo "Existing pilot; inspect --status or --resume." >&2; exit 3; }
fi
if [[ "${1:-}" == --resume ]]; then
    [[ -f "${pointer}" ]] || { echo "No pilot to resume." >&2; exit 2; }
    SDWPF_LOG_DIR="$(< "${pointer}")"
    [[ ! -f "${SDWPF_LOG_DIR}/result.json" ]] || { echo "Pilot already completed; nothing started."; exit 0; }
else
    sdwpf_log_init physics_norm "h12_f1_s2024_inner_physnorm1_mix0.2_ep8_trainonly" launch.log "physics_norm_$(date +%Y%m%d_%H%M%S)"
    printf '%s\n' "${SDWPF_LOG_DIR}" > "${pointer}"
fi
nohup python -u scripts/train_sdwpf_physics_norm.py --result-dir "${SDWPF_LOG_DIR}" \
    --source-v5-dir "${source_v5}" --protocol configs/sdwpf_physics_norm_protocol_v6.json \
    >> "${SDWPF_LOG_DIR}/launch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "${SDWPF_LOG_DIR}/launcher.pid"
echo "RESULT_DIR=${SDWPF_LOG_DIR}"
echo "PID=$!"
echo "Check: bash scripts/train/SDWPF_launch_physics_norm.sh --status"
