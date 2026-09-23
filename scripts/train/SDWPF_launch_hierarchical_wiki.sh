#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh

# One-command background launch; all files live in the dated run directory.
# Existing fold-matched pretraining and trend forecasts are reused.
export UTILITY_ADAPTER_MODE="${UTILITY_ADAPTER_MODE:-hierarchical_evidence}"
LAUNCH_TASK=hierarchical_wiki
if [[ "$UTILITY_ADAPTER_MODE" == calibrated_evidence ]]; then
    LAUNCH_TASK=calibrated_wiki
fi
if [[ "${UTILITY_FACTORIZED:-0}" == 1 ]]; then
    LAUNCH_TASK=factorized_wiki
fi
POINTER="outputs/logs/SDWPF/${LAUNCH_TASK}_latest.txt"
case "${1:-}" in
    --status)
        [[ -f "$POINTER" ]] || { echo "No hierarchical Wiki run has been launched."; exit 2; }
        directory="$(< "$POINTER")"
        [[ -n "$directory" && -d "$directory" ]] || { echo "Run directory is missing: $directory"; exit 2; }
        echo "RESULT_DIR=$directory"
        [[ ! -f "$directory/status.env" ]] || cat "$directory/status.env"
        if [[ -f "$directory/cv_metrics_summary.txt" ]]; then
            cat "$directory/cv_metrics_summary.txt"
            [[ ! -f "$directory/wiki_vs_trend.txt" ]] || cat "$directory/wiki_vs_trend.txt"
        else
            if [[ -f "$directory/launch.log" ]]; then
                tail -n 35 "$directory/launch.log"
            elif [[ -f "$directory/cv.log" ]]; then
                tail -n 35 "$directory/cv.log"
            else
                echo "Log allocation completed; worker has not written output yet."
            fi
        fi
        exit 0
        ;;
    --worker)
        # The parent allocated this directory; this worker owns finalization.
        _SDWPF_LOG_OWNS_DIR=1
        sdwpf_log_install_exit_trap
        bash scripts/train/SDWPF_utility_wiki_cv.sh
        python scripts/summarize_hierarchical_wiki.py --wiki-dir "$SDWPF_LOG_DIR" --trend-dir "$TREND_CV_DIR"
        exit 0
        ;;
    ""|--foreground) ;;
    *) echo "Usage: bash $0 [--status|--foreground]" >&2; exit 2 ;;
esac

export SOURCE_CV_DIR="${SOURCE_CV_DIR:-outputs/logs/SDWPF/20260913/002_cv_h12_rolling_holdout_folds0-1-2_seeds2024-2025-2026_blr0_hffcfe2bf396b}"
export TREND_CV_DIR="${TREND_CV_DIR:-outputs/logs/SDWPF/20260914/001_cv_trend_h12_rolling_holdout_folds0-1-2_seeds2024-2025-202_h36d88ba5e218}"
export FOLDS="${FOLDS:-0}" SEEDS="${SEEDS:-2024}" PRED_LEN="${PRED_LEN:-12}"
export FREEZE_NON_UTILITY=1
export UTILITY_CALIBRATION_FRACTION="${UTILITY_CALIBRATION_FRACTION:-0.2}"
export UTILITY_EVENT_MAX_SCALE="${UTILITY_EVENT_MAX_SCALE:-0.5}"
export UTILITY_COMPOSITION_MAX_SCALE="${UTILITY_COMPOSITION_MAX_SCALE:-0.25}"
export UTILITY_ADAPTER_WARMUP_EPOCHS="${UTILITY_ADAPTER_WARMUP_EPOCHS:-3}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" PATIENCE="${PATIENCE:-4}"
export UTILITY_LEARNING_RATE="${UTILITY_LEARNING_RATE:-0.0001}"
export UTILITY_LOSS_WEIGHT="${UTILITY_LOSS_WEIGHT:-0.2}"
export UTILITY_DECISION_LOSS_WEIGHT="${UTILITY_DECISION_LOSS_WEIGHT:-0.5}"
export UTILITY_CANDIDATE_LOSS_WEIGHT="${UTILITY_CANDIDATE_LOSS_WEIGHT:-1.0}"
export UTILITY_RANKING_LOSS_WEIGHT="${UTILITY_RANKING_LOSS_WEIGHT:-0.1}"
export UTILITY_RANKING_MARGIN="${UTILITY_RANKING_MARGIN:-0.0}"
export UTILITY_GATE_TEMPERATURE="${UTILITY_GATE_TEMPERATURE:-0.25}"
export UTILITY_TARGET_EPS="${UTILITY_TARGET_EPS:-0.05}"
export UTILITY_MIN_GAIN="${UTILITY_MIN_GAIN:-0.005}" UTILITY_INTERVENTION_FLOOR=1
export CV_ID="${CV_ID:-${LAUNCH_TASK}_h${PRED_LEN}_$(date +%Y%m%d_%H%M%S)_$$}"
python -c 'import torch; print("PyTorch:", torch.__version__); assert torch.cuda.is_available(), "CUDA unavailable in current environment"'
for directory in "$SOURCE_CV_DIR" "$TREND_CV_DIR"; do
    [[ -f "$directory/cv.env" ]] || { echo "Missing cv.env: $directory" >&2; exit 2; }
done
parameters="h${PRED_LEN}_f${FOLDS// /-}_s${SEEDS// /-}_lr${UTILITY_LEARNING_RATE}_warm${UTILITY_ADAPTER_WARMUP_EPOCHS}_ep${TRAIN_EPOCHS}_pat${PATIENCE}_ecap${UTILITY_EVENT_MAX_SCALE}_ccap${UTILITY_COMPOSITION_MAX_SCALE}_ulw${UTILITY_LOSS_WEIGHT}_dlw${UTILITY_DECISION_LOSS_WEIGHT}_clw${UTILITY_CANDIDATE_LOSS_WEIGHT}_rlw${UTILITY_RANKING_LOSS_WEIGHT}_margin${UTILITY_RANKING_MARGIN}_temp${UTILITY_GATE_TEMPERATURE}_eps${UTILITY_TARGET_EPS}_gain${UTILITY_MIN_GAIN}"
parameters="${parameters}_calfrac${UTILITY_CALIBRATION_FRACTION}_factor${UTILITY_FACTORIZED:-0}"
sdwpf_log_init "$LAUNCH_TASK" "$parameters" "cv.log" "$CV_ID"
printf '%s\n' "$SDWPF_LOG_DIR" > "$POINTER"
if [[ "${1:-}" == "--foreground" ]]; then
    _SDWPF_LOG_OWNS_DIR=1
    sdwpf_log_install_exit_trap
    bash scripts/train/SDWPF_utility_wiki_cv.sh
    python scripts/summarize_hierarchical_wiki.py --wiki-dir "$SDWPF_LOG_DIR" --trend-dir "$TREND_CV_DIR"
else
    nohup bash scripts/train/SDWPF_launch_hierarchical_wiki.sh --worker > "$SDWPF_LOG_DIR/launch.log" 2>&1 < /dev/null &
    job_pid=$!
    printf '%s\n' "$job_pid" > "$SDWPF_LOG_DIR/launcher.pid"
    echo "PID=$job_pid"
    echo "tail -f '$SDWPF_LOG_DIR/launch.log'"
    echo "Check results: UTILITY_FACTORIZED=${UTILITY_FACTORIZED:-0} UTILITY_ADAPTER_MODE=$UTILITY_ADAPTER_MODE bash scripts/train/SDWPF_launch_hierarchical_wiki.sh --status"
fi
