#!/usr/bin/env bash
# Run from repository root. No training; validation-only fixed-checkpoint probes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export WIKI_CV_DIR="${WIKI_CV_DIR:-outputs/logs/SDWPF/20260913/002_cv_h12_rolling_holdout_folds0-1-2_seeds2024-2025-2026_blr0_hffcfe2bf396b}"
export FOLDS="${FOLDS:-0 1 2}" SEEDS="${SEEDS:-2024}"
export WIKI_DIAGNOSTIC_SINGLE_EVENTS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WIKI_LLM_PATH="${WIKI_LLM_PATH:-outputs/model_cache/Qwen2.5-0.5B}"
[[ -f "$WIKI_CV_DIR/cv.env" ]] || { echo "Missing $WIKI_CV_DIR/cv.env" >&2; exit 2; }
pred_len="$(sed -n 's/^PRED_LEN=//p' "$WIKI_CV_DIR/cv.env" | tail -n 1)"
diag_dir="$(python -m utils.sdwpf_logging allocate --task wiki_single_event --parameters "h${pred_len}_val_folds${FOLDS// /-}_seeds${SEEDS// /-}_fixed_contribution")"
[[ -n "$diag_dir" && -d "$diag_dir" ]] || exit 2
nohup env SDWPF_LOG_DIR="$diag_dir" SDWPF_LOG_FILE="$diag_dir/diagnostic.log" bash -c '
    finish() {
        result=$?
        trap - EXIT
        python -m utils.sdwpf_logging finish --log-dir "$SDWPF_LOG_DIR" --exit-code "$result"
        exit "$result"
    }
    trap finish EXIT
    set -euo pipefail
    bash scripts/eval/SDWPF_wiki_diagnostic.sh
    python scripts/summarize_wiki_diagnostic.py --diagnostic-dir "$SDWPF_LOG_DIR"
    tar -czf "$SDWPF_LOG_DIR/wiki_single_event_review.tar.gz" -C "$SDWPF_LOG_DIR" diagnostic_report.txt diagnostic_metrics.csv diagnostic.env runs
    echo "[READY] $SDWPF_LOG_DIR/wiki_single_event_review.tar.gz"
' > "$diag_dir/launch.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$diag_dir" > outputs/logs/SDWPF/latest_wiki_single_event.txt
echo "PID=$pid"
echo "Directory: $diag_dir"
echo "Read: tail -f \"$diag_dir/launch.log\""
