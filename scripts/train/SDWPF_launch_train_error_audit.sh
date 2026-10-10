#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh
pointer=outputs/logs/SDWPF/train_error_audit_latest.txt
if [[ "${1:-}" == --status ]]; then
    [[ -f "${pointer}" ]] || { echo "No train error audit launched."; exit 2; }
    directory="$(< "${pointer}")"
    echo "RESULT_DIR=${directory}"
    for filename in status.env progress.json data_audit.json; do
        [[ ! -f "${directory}/${filename}" ]] || cat "${directory}/${filename}"
    done
    [[ ! -f "${directory}/report.json" ]] || echo "REPORT=${directory}/report.json"
    exit 0
fi
[[ -z "${1:-}" || "${1:-}" == --resume ]] || { echo "Usage: bash $0 [--status|--resume]" >&2; exit 2; }
mkdir -p outputs/logs/SDWPF
exec 9> outputs/logs/SDWPF/train_error_audit_launcher.lock
flock -n 9 || { echo "Another audit launcher is active." >&2; exit 3; }
source_v5=outputs/logs/SDWPF/20261009/003_accuracy_tuning_h12_f1_s2024_inner4_mix0.2-0.35-0.5_ep8_trainonly
source_v6=outputs/logs/SDWPF/20261009/004_physics_norm_h12_f1_s2024_inner_physnorm1_mix0.2_ep8_trainonly
for source_dir in "${source_v5}" "${source_v6}"; do
    [[ -f "${source_dir}/result.json" ]] || { echo "Completed source missing: ${source_dir}" >&2; exit 2; }
done
if [[ -f "${pointer}" ]]; then
    previous="$(< "${pointer}")"
    previous_pid="$(cat "${previous}/launcher.pid" 2>/dev/null || true)"
    if [[ -n "${previous_pid}" ]] && kill -0 "${previous_pid}" 2>/dev/null; then
        echo "Existing audit PID=${previous_pid}; no duplicate started." >&2; exit 3
    fi
    [[ "${1:-}" == --resume ]] || { echo "Existing audit; inspect --status or --resume." >&2; exit 3; }
fi
if [[ "${1:-}" == --resume ]]; then
    [[ -f "${pointer}" ]] || { echo "No audit to resume." >&2; exit 2; }
    SDWPF_LOG_DIR="$(< "${pointer}")"
    [[ ! -f "${SDWPF_LOG_DIR}/report.json" ]] || { echo "Audit already completed; nothing started."; exit 0; }
else
    sdwpf_log_init train_error_audit "h12_f1_s2024_forward_oof_cpu2_epoch8" launch.log "train_error_audit_$(date +%Y%m%d_%H%M%S)"
    printf '%s\n' "${SDWPF_LOG_DIR}" > "${pointer}"
fi
# This read-only task deliberately avoids an occupied GPU. niceness and two
# CPU threads limit interference with other users' work; no model is trained.
nohup env CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
    nice -n 10 python -u scripts/audit_sdwpf_train_errors.py --result-dir "${SDWPF_LOG_DIR}" \
    --source-v5-dir "${source_v5}" --source-v6-dir "${source_v6}" \
    >> "${SDWPF_LOG_DIR}/launch.log" 2>&1 < /dev/null 9>&- &
printf '%s\n' "$!" > "${SDWPF_LOG_DIR}/launcher.pid"
echo "RESULT_DIR=${SDWPF_LOG_DIR}"
echo "PID=$!"
echo "Check: bash scripts/train/SDWPF_launch_train_error_audit.sh --status"
