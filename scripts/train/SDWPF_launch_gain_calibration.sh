#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/lib/sdwpf_log.sh
POINTER=outputs/logs/SDWPF/wiki_gain_calibration_latest.txt
case "${1:-}" in
    --status)
        [[ -f "$POINTER" ]] || { echo "No gain calibration run launched."; exit 2; }
        directory="$(< "$POINTER")"
        [[ -n "$directory" && -d "$directory" ]] || { echo "Missing result directory"; exit 2; }
        echo "RESULT_DIR=$directory"
        [[ ! -f "$directory/status.env" ]] || cat "$directory/status.env"
        if [[ -f "$directory/summary.txt" ]]; then
            cat "$directory/summary.txt"
            if [[ -f "$directory/validation_metrics.json" && -f "$directory/gain_calibration.json" ]]; then
                python scripts/report_gain_calibration.py "$directory"
            fi
        elif [[ -f "$directory/launch.log" ]]; then
            tail -n 30 "$directory/launch.log"
        fi
        exit 0 ;;
    --worker)
        _SDWPF_LOG_OWNS_DIR=1
        sdwpf_log_install_exit_trap
        python -u scripts/calibrate_factorized_wiki.py --source-dir "$SOURCE_FACTOR_DIR" \
            --output-dir "$SDWPF_LOG_DIR" --fold "$FOLD" --seed "$SEED" \
            --blocks "$GAIN_BLOCKS" --min-windows "$GAIN_MIN_WINDOWS" --penalty "$GAIN_PENALTY"
        exit 0 ;;
    "") ;;
    *) echo "Usage: bash $0 [--status]" >&2; exit 2 ;;
esac
if [[ -z "${SOURCE_FACTOR_DIR:-}" ]]; then
    [[ -f outputs/logs/SDWPF/factorized_wiki_latest.txt ]] || { echo "Missing factorized run pointer"; exit 2; }
    SOURCE_FACTOR_DIR="$(< outputs/logs/SDWPF/factorized_wiki_latest.txt)"
fi
export SOURCE_FACTOR_DIR
export FOLD="${FOLD:-0}" SEED="${SEED:-2024}"
export GAIN_BLOCKS="${GAIN_BLOCKS:-3}" GAIN_MIN_WINDOWS="${GAIN_MIN_WINDOWS:-32}" GAIN_PENALTY="${GAIN_PENALTY:-0.25}"
[[ -f "$SOURCE_FACTOR_DIR/cv.env" && -f "$SOURCE_FACTOR_DIR/runs/f${FOLD}_s${SEED}/finetune.log" ]] || { echo "Missing source run: $SOURCE_FACTOR_DIR"; exit 2; }
pred_len="$(sed -n 's/^PRED_LEN=//p' "$SOURCE_FACTOR_DIR/cv.env" | head -n 1)"
parameters="h${pred_len}_f${FOLD}_s${SEED}_blocks${GAIN_BLOCKS}_min${GAIN_MIN_WINDOWS}_pen${GAIN_PENALTY}_joint_hout"
sdwpf_log_init wiki_gain_calibration "$parameters" launch.log "gain_joint_$(date +%Y%m%d_%H%M%S)_$$"
{
    echo "SOURCE_FACTOR_DIR=$SOURCE_FACTOR_DIR"
    echo "FOLD=$FOLD"
    echo "SEED=$SEED"
    echo "PRED_LEN=$pred_len"
    echo "GAIN_BLOCKS=$GAIN_BLOCKS"
    echo "GAIN_MIN_WINDOWS=$GAIN_MIN_WINDOWS"
    echo "GAIN_PENALTY=$GAIN_PENALTY"
} > "$SDWPF_LOG_DIR/calibration.env"
printf '%s\n' "$SDWPF_LOG_DIR" > "$POINTER"
nohup bash scripts/train/SDWPF_launch_gain_calibration.sh --worker > "$SDWPF_LOG_DIR/launch.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$SDWPF_LOG_DIR/launcher.pid"
echo "PID=$!"
echo "Check: bash scripts/train/SDWPF_launch_gain_calibration.sh --status"
