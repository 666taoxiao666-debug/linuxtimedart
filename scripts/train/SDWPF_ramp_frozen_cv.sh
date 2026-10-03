#!/usr/bin/env bash
set -euo pipefail

# Frozen replication of the wind-discordant ramp gate. Reuse each fold/seed's
# matched pure-trend pretraining; run one fine-tuning job at a time.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"
POINTER="outputs/logs/SDWPF/ramp_frozen_cv_latest.txt"

if [[ "${1:-}" == "--status" ]]; then
    [[ -f "${POINTER}" ]] || { echo "No frozen ramp CV has been launched."; exit 2; }
    result_dir="$(< "${POINTER}")"
    echo "RESULT_DIR=${result_dir}"
    [[ ! -f "${result_dir}/status.env" ]] || cat "${result_dir}/status.env"
    [[ ! -f "${result_dir}/runs.tsv" ]] || cat "${result_dir}/runs.tsv"
    exit 0
fi
[[ -z "${1:-}" ]] || { echo "Usage: bash $0 [--status]" >&2; exit 2; }

TREND_CV_DIR="${TREND_CV_DIR:-}"
[[ -f "${TREND_CV_DIR}/cv.env" ]] || {
    echo "TREND_CV_DIR must point to the completed pure-trend CV." >&2
    exit 2
}
grep -q '^PROMPT_ROUTER=trend$' "${TREND_CV_DIR}/cv.env" || {
    echo "TREND_CV_DIR is not the pure-trend control." >&2
    exit 2
}
grep -q '^PRED_LEN=12$' "${TREND_CV_DIR}/cv.env" || {
    echo "This replication requires the h12 control." >&2
    exit 2
}

CV_ID="${CV_ID:-ramp_discordant_frozen_s2025-2026_$(date +%Y%m%d_%H%M%S)}"
sdwpf_log_init "ramp_frozen_cv" "h12_f0-1-2_s2025-2026_rlr0.00005_wind_discordant" "matrix.log" "${CV_ID}"
sdwpf_log_install_exit_trap
result_dir="${SDWPF_LOG_DIR}"
printf '%s\n' "${result_dir}" > "${POINTER}"
sdwpf_log_capture
{
    echo "TREND_CV_DIR=${TREND_CV_DIR}"
    echo "FOLDS=0,1,2"
    echo "SEEDS=2025,2026"
    echo "PRED_LEN=12"
    echo "RAMP_GATE_MODE=wind_discordant"
    echo "RAMP_LEARNING_RATE=0.00005"
    echo "ROBUST_PITCH=0"
    echo "TEST_SPLIT=sealed"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
} > "${result_dir}/matrix.env"
printf 'fold\tseed\tresult_dir\n' > "${result_dir}/runs.tsv"

for seed in 2025 2026; do
    for fold in 0 1 2; do
        stage="${TREND_CV_DIR}/runs/f${fold}_s${seed}/pipeline.env"
        [[ -f "${stage}" ]] || { echo "Missing matched stage: ${stage}" >&2; exit 2; }
        if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
            echo "GPU is occupied; stopping before fold=${fold} seed=${seed}." >&2
            exit 3
        fi
        echo "===== frozen ramp fold=${fold} seed=${seed} ====="
        SDWPF_LOG_DIR= SDWPF_LOG_FILE= \
        TREND_CV_DIR="${TREND_CV_DIR}" FOLD="${fold}" SEED="${seed}" \
        RAMP_LEARNING_RATE=0.00005 RAMP_GATE_MODE=wind_discordant \
        TRAIN_EPOCHS=5 PATIENCE=2 LEARNING_RATE=0.000001 \
        NEW_MODULE_LEARNING_RATE=0.000005 \
        RUN_ID="${CV_ID}_f${fold}_s${seed}" \
        bash scripts/train/SDWPF_ramp_pilot.sh
        run_dir="$(< outputs/logs/SDWPF/ramp_pilot_latest.txt)"
        grep -q '^STATUS=COMPLETED$' "${run_dir}/status.env" || {
            echo "Run did not complete: ${run_dir}" >&2
            exit 3
        }
        printf '%s\t%s\t%s\n' "${fold}" "${seed}" "${run_dir}" >> "${result_dir}/runs.tsv"
    done
done
echo "[RAMP_FROZEN_CV] Complete: ${result_dir}/runs.tsv"
