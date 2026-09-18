#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/sdwpf_log.sh"

: "${WIKI_CV_DIR:?Set WIKI_CV_DIR to the completed static Wiki CV directory}"
read_value() { sed -n "s/^${2}=//p" "$1" | tail -n 1; }
[[ -f "${WIKI_CV_DIR}/cv.env" ]] || { echo "Missing cv.env" >&2; exit 2; }
[[ "$(read_value "${WIKI_CV_DIR}/cv.env" PROMPT_ROUTER)" == compositional_wiki ]] || exit 2
[[ "$(read_value "${WIKI_CV_DIR}/cv.env" SPLIT)" == rolling_holdout ]] || exit 2
FOLDS="${FOLDS:-0 1 2}"
SEEDS="${SEEDS:-2024 2025 2026}"
PRED_LEN="$(read_value "${WIKI_CV_DIR}/cv.env" PRED_LEN)"
export PROMPT_ROUTER=compositional_wiki SPLIT=rolling_holdout EVAL_SPLIT=val WIKI_DIAGNOSTIC=1
export PRED_LEN
for name in SCENE_WIKI_CONFIG SCENE_WIKI_EMBEDDINGS SCENE_WIKI_TOP_K SCENE_WIKI_TEMPERATURE SCENE_WIKI_RULE_WEIGHT SCENE_WIKI_PROMPT_GATE_INIT SCENE_WIKI_ACTIVATION_THRESHOLD SCENE_WIKI_CONFIDENCE_POWER REGIME_LABEL_METHOD REGIME_CALIBRATION_QUANTILE; do
    value="$(read_value "${WIKI_CV_DIR}/cv.env" "$name")"
    [[ -n "$value" ]] || { echo "Missing CV parameter: $name" >&2; exit 2; }
    export "$name=$value"
done
for name in UTILITY_WIKI UTILITY_GATE_TEMPERATURE UTILITY_MIN_GAIN UTILITY_INTERVENTION_FLOOR; do
    value="$(read_value "${WIKI_CV_DIR}/cv.env" "$name")"
    [[ -z "$value" ]] || export "$name=$value"
done
[[ -f "$SCENE_WIKI_CONFIG" && -f "$SCENE_WIKI_EMBEDDINGS" ]] || { echo "Missing frozen Wiki files" >&2; exit 2; }
# Validate all selected checkpoints before doing any inference. Do not execute
# pipeline.env as shell code; read only the expected values.
for f in $FOLDS; do
    for s in $SEEDS; do
        stage="${WIKI_CV_DIR}/runs/f${f}_s${s}/pipeline.env"
        [[ -f "$stage" ]] || { echo "Missing $stage" >&2; exit 2; }
        checkpoint="$(read_value "$stage" FINETUNE_CHECKPOINT)"
        [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
    done
done
RUN_ID="${RUN_ID:-wiki_diag_$(date +%Y%m%d_%H%M%S)}"
sdwpf_log_init wiki_diagnostic "h${PRED_LEN}_val_folds${FOLDS// /-}_seeds${SEEDS// /-}_single${WIKI_DIAGNOSTIC_SINGLE_EVENTS:-0}" diagnostic.log "$RUN_ID"
sdwpf_log_install_exit_trap
DIAG_DIR="$SDWPF_LOG_DIR"
sdwpf_log_capture
printf 'WIKI_CV_DIR=%s\nFOLDS=%s\nSEEDS=%s\n' "$WIKI_CV_DIR" "$FOLDS" "$SEEDS" > "$DIAG_DIR/diagnostic.env"
for f in $FOLDS; do
    for s in $SEEDS; do
        echo "[WIKI-DIAG] fold=$f seed=$s"
        stage="${WIKI_CV_DIR}/runs/f${f}_s${s}/pipeline.env"
        checkpoint="$(read_value "$stage" FINETUNE_CHECKPOINT)"
        n_folds="$(read_value "$stage" N_FOLDS)"
        rated="$(read_value "$stage" RATED_POWER)"
        run_dir="$DIAG_DIR/runs/f${f}_s${s}"
        mkdir -p "$run_dir"
        FOLD="$f" SEED="$s" N_FOLDS="$n_folds" RATED_POWER="$rated" \
        RUN_ID="${RUN_ID}_f${f}_s${s}" FINETUNE_CHECKPOINT="$checkpoint" \
        REPORT_OUTPUT_DIR="$run_dir/artifacts" \
        bash scripts/eval/SDWPF_final_eval.sh 2>&1 | tee "$run_dir/eval.log"
    done
done
echo "[WIKI-DIAG] Completed: $DIAG_DIR"
