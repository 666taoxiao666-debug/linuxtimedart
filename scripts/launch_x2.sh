#!/usr/bin/env bash
# X2 launcher for Cursor (Claude 2026-11-16).  Run INSIDE a clean worktree of the reviewed commit.
# Usage: bash scripts/launch_x2.sh <smoke|a|b|report|a2|report2|x3smoke|x3gpu|x3q4|x3final>
#   a2/report2: second A-line round (CPU) against the round-1 run X2_ROUND1 (default x2_91d4d20/a).
#   smoke : CPU, tiny real slices, A -> B -> report; must pass before a/b.
#   a     : part A, LightGBM on CPU (X2_THREADS, default 8).
#   b     : part B, neural bases on GPU (X2_GPU_FRACTION, default 0.35; 2 CPU threads).
#   report: after a and b completed (CPU).
# Never reuses an output directory: a failed run stays as evidence; fix -> new commit -> new worktree -> rerun.
set -euo pipefail
DATA=${X2_DATA_ROOT:-/home/XT/nuist/pythoncode/TimeDARTFirst}
LOGS=$DATA/outputs/logs
PY=${X2_PYTHON:-/home/XT/miniconda3/envs/timedart/bin/python}
HERE=$(cd "$(dirname "$0")/.." && pwd)
MODE=${1:?usage: launch_x2.sh <smoke|a|b|report|a2|report2|x3smoke|x3gpu|x3q4|x3final>}
if [ -n "$(git -C "$HERE" status --porcelain --untracked-files=no)" ]; then
  echo "worktree not clean: $HERE"; exit 1
fi
COMMIT=$(git -C "$HERE" rev-parse --short=7 HEAD)
case "$MODE" in
  x3smoke|x3gpu|x3q4|x3final)
    # X3 (Claude 2026-11-18): chains of run_x3.py modes; a chain stops at the first failure (failed.json kept).
    #   x3smoke : q1check q1 q2 q3 q4 report on tiny slices (must complete before any formal chain)
    #   x3gpu   : q1check q1 q2 (GPU)          x3q4 : q4 (CPU, after formal q1)
    #   x3final : q3 report (after q2 finished or failed AND q4 completed)
    X3SHA=$(sha256sum "$HERE/configs/x3_protocol.json" | cut -d' ' -f1)
    X3BASE=$LOGS/X3/x3_${COMMIT}
    mkdir -p "$X3BASE"
    case "$MODE" in
      x3smoke) STEPS="q1check q1 q2 q3 q4 report"; FLAG="--smoke" ;;
      x3gpu)   STEPS="q1check q1 q2"; FLAG="" ;;
      x3q4)    STEPS="q4"; FLAG="" ;;
      x3final) STEPS="q3 report"; FLAG="" ;;
    esac
    if ! mkdir "$X3BASE/.$MODE.launch"; then echo "launch already claimed: $MODE (do not retry)"; exit 1; fi
    cd "$HERE"
    R=$LOGS/X1/cursor_env
    nohup bash -c "set -e; for m in $STEPS; do echo \"[X3] start \$m\"; '$PY' scripts/run_x3.py --mode \$m --logs '$LOGS' --base '$X3BASE' --protocol-sha256 $X3SHA $FLAG --threads ${X2_THREADS:-8} --chronos-receipt '$R/receipt.json' --chronos-manifest '$R/chronos2_manifest.json'; done; echo '[X3] chain done'" \
      > "$X3BASE/$MODE.log" 2>&1 &
    echo $! > "$X3BASE/$MODE.pid"
    echo "started X3 chain $MODE pid=$(cat "$X3BASE/$MODE.pid") commit=$COMMIT protocol=$X3SHA"
    echo "base=$X3BASE log=$X3BASE/$MODE.log status: bash scripts/status_x2.sh"
    exit 0 ;;
esac
PSHA=$(sha256sum "$HERE/configs/x2_protocol.json" | cut -d' ' -f1)
BASE=$LOGS/X2/x2_${COMMIT}
OUT=$BASE/$MODE
if [ -e "$OUT" ]; then echo "already exists: $OUT (do not reuse)"; exit 1; fi
mkdir -p "$BASE"
case "$MODE" in
  smoke)  EXTRA=(--threads 2 --device cpu) ;;
  a)      EXTRA=(--threads "${X2_THREADS:-8}" --smoke-result "$BASE/smoke") ;;
  b)      EXTRA=(--threads 2 --device cuda --gpu-fraction "${X2_GPU_FRACTION:-0.35}" --smoke-result "$BASE/smoke") ;;
  report) EXTRA=(--threads 2 --a-results "$BASE/a" --b-results "$BASE/b") ;;
  a2)     EXTRA=(--threads "${X2_THREADS:-8}" --a-results "${X2_ROUND1:-$LOGS/X2/x2_91d4d20/a}") ;;
  report2) EXTRA=(--threads 2 --a-results "${X2_ROUND1:-$LOGS/X2/x2_91d4d20/a}" --a2-results "$BASE/a2") ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
# Persistent atomic claim: duplicate clicks must not replace the log or PID.
if ! mkdir "$BASE/.$MODE.launch"; then
  echo "launch already claimed: $MODE (do not retry)"; exit 1
fi
cd "$HERE"
nohup "$PY" scripts/run_x2.py --mode "$MODE" --logs "$LOGS" --output "$OUT" --protocol-sha256 "$PSHA" "${EXTRA[@]}" \
  > "$BASE/$MODE.log" 2>&1 &
echo $! > "$BASE/$MODE.pid"
echo "started mode=$MODE pid=$(cat "$BASE/$MODE.pid") commit=$COMMIT protocol=$PSHA"
echo "output=$OUT log=$BASE/$MODE.log status: bash scripts/status_x2.sh"
