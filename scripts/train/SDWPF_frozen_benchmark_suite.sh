#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

LATEST_POINTER="outputs/logs/SDWPF/frozen_benchmark_latest.txt"
if [[ "${1:-}" == "--status" ]]; then
    if [[ ! -f "${LATEST_POINTER}" ]]; then
        echo "No frozen benchmark run has been launched."
        exit 1
    fi
    ROOT="$(cat "${LATEST_POINTER}")"
    echo "RESULT_DIR=$(cd "${ROOT}" && pwd)"
    [[ ! -f "${ROOT}/status.env" ]] || cat "${ROOT}/status.env"
    [[ ! -f "${ROOT}/benchmark_summary.txt" ]] || cat "${ROOT}/benchmark_summary.txt"
    [[ ! -f "${ROOT}/matrix.tsv" ]] || tail -n 20 "${ROOT}/matrix.tsv"
    exit 0
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/train/SDWPF_frozen_benchmark_suite.sh [--status]" >&2
    exit 2
fi

if [[ -f "${LATEST_POINTER}" ]]; then
    previous="$(cat "${LATEST_POINTER}")"
    if [[ -f "${previous}/launcher.pid" && -f "${previous}/status.env" ]] \
       && grep -q '^STATUS=RUNNING$' "${previous}/status.env"; then
        previous_pid="$(cat "${previous}/launcher.pid")"
        if [[ "${previous_pid}" =~ ^[0-9]+$ ]] && kill -0 "${previous_pid}" 2>/dev/null; then
            echo "Frozen benchmark is already running: pid=${previous_pid} dir=${previous}" >&2
            exit 4
        fi
    fi
fi

source scripts/lib/sdwpf_log.sh
PROTOCOL_CONFIG="configs/sdwpf_frozen_protocol_v1.json"
PROTOCOL_AUDIT="$(python scripts/check_sdwpf_frozen_protocol.py --config "${PROTOCOL_CONFIG}")"
PROTOCOL_ID="$(sed -n 's/^PROTOCOL_ID=//p' <<< "${PROTOCOL_AUDIT}")"
PROTOCOL_SHA256="$(sed -n 's/^PROTOCOL_SHA256=//p' <<< "${PROTOCOL_AUDIT}")"
export PROTOCOL_ID PROTOCOL_SHA256
GPU="${GPU:-0}"

sdwpf_log_init "frozen_benchmark" "h12_f0-1-2_s2024-2025-2026_patchtst-dlinear-classical_${PROTOCOL_SHA256:0:12}" "matrix.log" "${PROTOCOL_ID}"
sdwpf_log_install_exit_trap
ROOT="${SDWPF_LOG_DIR}"
sdwpf_log_capture
mkdir -p "$(dirname "${LATEST_POINTER}")"
printf '%s\n' "${ROOT}" > "${LATEST_POINTER}"
printf 'family\tmethod\tfold\tseed\tstatus\tresult_dir\n' > "${ROOT}/matrix.tsv"
cp "${PROTOCOL_CONFIG}" "${ROOT}/frozen_protocol.json"
printf '%s\n' "${PROTOCOL_AUDIT}" > "${ROOT}/protocol_audit.env"
printf '%s\n' "$$" > "${ROOT}/launcher.pid"

for seed in 2024 2025 2026; do
    for fold in 0 1 2; do
        result="${ROOT}/runs/classical/f${fold}_s${seed}"
        mkdir -p "${result}"
        if [[ -f "${result}/artifacts/metrics_summary.csv" ]]; then
            printf 'classical\tall\t%s\t%s\tREUSED_COMPLETE\t%s\n' "${fold}" "${seed}" "${result}" >> "${ROOT}/matrix.tsv"
            continue
        fi
        printf 'classical\tall\t%s\t%s\tRUNNING\t%s\n' "${fold}" "${seed}" "${result}" >> "${ROOT}/matrix.tsv"
        SEED="${seed}" FOLD="${fold}" N_FOLDS=3 SPLIT=rolling_holdout PRED_LEN=12 \
        EVAL_STRIDE=12 EVAL_SPLIT=val BASELINE_MAX_SAMPLES=500000 \
        BASELINE_RIDGE_ALPHA=1.0 BASELINE_RIDGE_MAX_WINDOWS=200000 GPU="${GPU}" \
        RUN_ID="${PROTOCOL_ID}_classical_f${fold}_s${seed}" \
        SDWPF_LOG_DIR="${result}" SDWPF_LOG_FILE="${result}/baseline.log" \
        bash scripts/eval/SDWPF_baselines.sh
        printf 'classical\tall\t%s\t%s\tCOMPLETED\t%s\n' "${fold}" "${seed}" "${result}" >> "${ROOT}/matrix.tsv"
    done
done

for model in PatchTST DLinear; do
    if [[ "${model}" == "PatchTST" ]]; then
        lr=0.0001
    else
        lr=0.001
    fi
    for seed in 2024 2025 2026; do
        for fold in 0 1 2; do
            result="${ROOT}/runs/deep/${model}/f${fold}_s${seed}"
            mkdir -p "${result}"
            if [[ -f "${result}/artifacts/metrics.json" && -f "${result}/baseline.env" ]] \
               && grep -q '^COMPLETED_AT=' "${result}/baseline.env"; then
                printf 'deep\t%s\t%s\t%s\tREUSED_COMPLETE\t%s\n' "${model}" "${fold}" "${seed}" "${result}" >> "${ROOT}/matrix.tsv"
                continue
            fi
            printf 'deep\t%s\t%s\t%s\tRUNNING\t%s\n' "${model}" "${fold}" "${seed}" "${result}" >> "${ROOT}/matrix.tsv"
            MODEL="${model}" SEED="${seed}" FOLD="${fold}" GPU="${GPU}" \
            TRAIN_EPOCHS=20 PATIENCE=5 LEARNING_RATE="${lr}" LOSS=MSE \
            RUN_ID="${PROTOCOL_ID}_${model}_f${fold}_s${seed}" \
            SDWPF_LOG_DIR="${result}" SDWPF_LOG_FILE="${result}/train.log" \
            bash scripts/train/SDWPF_deep_baseline.sh
            printf 'deep\t%s\t%s\t%s\tCOMPLETED\t%s\n' "${model}" "${fold}" "${seed}" "${result}" >> "${ROOT}/matrix.tsv"
        done
    done
done

python scripts/summarize_frozen_benchmarks.py "${ROOT}"
bash scripts/eval/SDWPF_frozen_evidence.sh "${ROOT}"
echo "[FROZEN BENCHMARK] Completed without opening the sealed test split."
echo "[FROZEN BENCHMARK] Results: ${ROOT}"
