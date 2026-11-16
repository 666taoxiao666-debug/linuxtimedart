#!/usr/bin/env bash
# X2 launcher for Cursor (Claude 2026-11-16).  Run INSIDE a clean worktree of the reviewed commit.
# Usage: bash scripts/launch_x2.sh <smoke|a|b|report>
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
MODE=${1:?usage: launch_x2.sh <smoke|a|b|report>}
if [ -n "$(git -C "$HERE" status --porcelain --untracked-files=no)" ]; then
  echo "worktree not clean: $HERE"; exit 1
fi
COMMIT=$(git -C "$HERE" rev-parse --short=7 HEAD)
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
