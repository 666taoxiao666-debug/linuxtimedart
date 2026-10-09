#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh
pointer=outputs/logs/SDWPF/accuracy_tuning_latest.txt
protocol=configs/sdwpf_accuracy_tuning_protocol_v5.json
trend=outputs/logs/SDWPF/20260914/001_cv_trend_h12_rolling_holdout_folds0-1-2_seeds2024-2025-202_h36d88ba5e218
if [[ "${1:-}" == --status ]]; then
    [[ -f "${pointer}" ]] || { echo "No parameter search launched."; exit 2; }
    directory="$(< "${pointer}")"
    echo "RESULT_DIR=${directory}"
    [[ ! -f "${directory}/status.env" ]] || cat "${directory}/status.env"
    for filename in progress.json selected.json result.json; do
        [[ ! -f "${directory}/${filename}" ]] || { echo "${filename}:"; cat "${directory}/${filename}"; }
    done
    exit 0
fi
[[ -z "${1:-}" || "${1:-}" == --resume ]] || { echo "Usage: bash $0 [--status|--resume]" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${gpu_processes}" ]] || { echo "GPU occupied: ${gpu_processes}" >&2; exit 3; }
if [[ -f "${pointer}" ]]; then
    previous="$(< "${pointer}")"
    previous_pid="$(cat "${previous}/launcher.pid" 2>/dev/null || true)"
    if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Existing search PID=${previous_pid}; no duplicate started." >&2
        exit 3
    fi
    if [[ "${1:-}" != --resume ]]; then
        echo "A search already exists; inspect --status or resume its unfinished stages with --resume." >&2
        exit 3
    fi
fi
if [[ "${1:-}" == --resume ]]; then
    [[ -f "${pointer}" ]] || { echo "No search to resume." >&2; exit 2; }
    SDWPF_LOG_DIR="$(< "${pointer}")"
else
    sdwpf_log_init accuracy_tuning "h12_f1_s2024_inner4_mix0.2-0.35-0.5_ep8_trainonly" launch.log "accuracy_tuning_$(date +%Y%m%d_%H%M%S)"
    printf '%s\n' "${SDWPF_LOG_DIR}" > "${pointer}"
fi
nohup python -u scripts/tune_sdwpf_accuracy.py --result-dir "${SDWPF_LOG_DIR}" \
    --protocol "${protocol}" --trend-cv-dir "${trend}" \
    >> "${SDWPF_LOG_DIR}/launch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "${SDWPF_LOG_DIR}/launcher.pid"
echo "RESULT_DIR=${SDWPF_LOG_DIR}"
echo "PID=$!"
echo "Check: bash scripts/train/SDWPF_launch_accuracy_tuning.sh --status"
