#!/usr/bin/env bash
# X2 launcher for Cursor (Claude 2026-11-16).  Run INSIDE a clean worktree of the reviewed commit.
# Usage: bash scripts/launch_x2.sh <smoke|a|b|report|a2|report2|x3smoke|x3gpu|x3q4|x3final|x4smoke|x4gpu|x4d0|x4final|x5smoke|x5run|x5bsmoke|x5brun|wjbsmoke|wjbcpu|wjbgpu|wjbfinal>
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
MODE=${1:?usage: launch_x2.sh <smoke|a|b|report|a2|report2|x3smoke|x3gpu|x3q4|x3final|x4smoke|x4gpu|x4d0|x4final|x5smoke|x5run|x5bsmoke|x5brun|wjbsmoke|wjbcpu|wjbgpu|wjbfinal>}
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
  x4smoke|x4gpu|x4d0|x4final)
    # X4 (Claude 2026-11-20): chains of run_x4.py modes; a chain stops at the first failure (failed.json kept).
    #   x4smoke : a b d0 report on tiny slices (must complete before any formal chain)
    #   x4gpu   : a b (one GPU queue, in this order)      x4d0 : d0 (CPU, may run in parallel with x4gpu)
    #   x4final : report (after a, b and d0 completed)
    X4SHA=$(sha256sum "$HERE/configs/x4_protocol.json" | cut -d' ' -f1)
    X4BASE=$LOGS/X4/x4_${COMMIT}
    mkdir -p "$X4BASE"
    case "$MODE" in
      x4smoke) STEPS="a b d0 report"; FLAG="--smoke" ;;
      x4gpu)   STEPS="a b"; FLAG="" ;;
      x4d0)    STEPS="d0"; FLAG="" ;;
      x4final) STEPS="report"; FLAG="" ;;
    esac
    if ! mkdir "$X4BASE/.$MODE.launch"; then echo "launch already claimed: $MODE (do not retry)"; exit 1; fi
    cd "$HERE"
    R=$LOGS/X1/cursor_env
    nohup bash -c "set -e; for m in $STEPS; do echo \"[X4] start \$m\"; '$PY' scripts/run_x4.py --mode \$m --logs '$LOGS' --base '$X4BASE' --protocol-sha256 $X4SHA $FLAG --threads ${X2_THREADS:-8} --chronos-receipt '$R/receipt.json' --chronos-manifest '$R/chronos2_manifest.json'; done; echo '[X4] chain done'" \
      > "$X4BASE/$MODE.log" 2>&1 &
    echo $! > "$X4BASE/$MODE.pid"
    echo "started X4 chain $MODE pid=$(cat "$X4BASE/$MODE.pid") commit=$COMMIT protocol=$X4SHA"
    echo "base=$X4BASE log=$X4BASE/$MODE.log status: bash scripts/status_x2.sh"
    exit 0 ;;
  x5smoke|x5run)
    # X5 (Claude 2026-11-21): NWP-latency value of upstream observations, LightGBM on CPU (run_x5.py).
    #   x5smoke : tiny slices, delays 6 and 12 (must pass before x5run)      x5run : formal, all delays
    X5SHA=$(sha256sum "$HERE/configs/x5_protocol.json" | cut -d' ' -f1)
    X5BASE=$LOGS/X5/x5_${COMMIT}
    mkdir -p "$X5BASE"
    case "$MODE" in
      x5smoke) X5ARGS=(--mode smoke --output "$X5BASE/smoke" --threads 2) ;;
      x5run)   X5ARGS=(--mode run --output "$X5BASE/run" --threads "${X2_THREADS:-8}" --smoke-result "$X5BASE/smoke") ;;
    esac
    if ! mkdir "$X5BASE/.$MODE.launch"; then echo "launch already claimed: $MODE (do not retry)"; exit 1; fi
    cd "$HERE"
    nohup "$PY" scripts/run_x5.py --logs "$LOGS" --protocol-sha256 "$X5SHA" "${X5ARGS[@]}" > "$X5BASE/$MODE.log" 2>&1 &
    echo $! > "$X5BASE/$MODE.pid"
    echo "started X5 $MODE pid=$(cat "$X5BASE/$MODE.pid") commit=$COMMIT protocol=$X5SHA"
    echo "base=$X5BASE log=$X5BASE/$MODE.log status: bash scripts/status_x2.sh"
    exit 0 ;;
  x5bsmoke|x5brun)
    # X5b (Claude 2026-11-23): matched coverage, fresh 6 h vs old 15 h GFS cycle (run_x5.py msmoke / mrun, CPU).
    #   x5bsmoke : tiny slices (must pass before x5brun)      x5brun : formal, 8 LightGBM arms per farm
    X5SHA=$(sha256sum "$HERE/configs/x5b_protocol.json" | cut -d' ' -f1)
    X5BASE=$LOGS/X5/x5_${COMMIT}
    mkdir -p "$X5BASE"
    case "$MODE" in
      x5bsmoke) X5ARGS=(--mode msmoke --output "$X5BASE/msmoke" --threads 2) ;;
      x5brun)   X5ARGS=(--mode mrun --output "$X5BASE/mrun" --threads "${X2_THREADS:-8}" --smoke-result "$X5BASE/msmoke") ;;
    esac
    if ! mkdir "$X5BASE/.$MODE.launch"; then echo "launch already claimed: $MODE (do not retry)"; exit 1; fi
    cd "$HERE"
    nohup "$PY" scripts/run_x5.py --logs "$LOGS" --protocol-sha256 "$X5SHA" "${X5ARGS[@]}" > "$X5BASE/$MODE.log" 2>&1 &
    echo $! > "$X5BASE/$MODE.pid"
    echo "started X5b $MODE pid=$(cat "$X5BASE/$MODE.pid") commit=$COMMIT protocol=$X5SHA"
    echo "base=$X5BASE log=$X5BASE/$MODE.log status: bash scripts/status_x2.sh"
    exit 0 ;;
  wjbsmoke|wjbcpu|wjbgpu|wjbfinal)
    # wind-journal-baselines (Claude 2026-11-24): chains of run_wjb.py modes; a chain stops at the first failure.
    #   wjbsmoke : krr lgbh tsg cabin report on tiny slices (must complete before any formal chain)
    #   wjbcpu   : krr lgbh tsg (CPU; TSG last: it may stop as a registered technical limit)
    #   wjbgpu   : cabin (GPU; may run in parallel with wjbcpu)
    #   wjbfinal : report (after wjbcpu and wjbgpu have finished)
    # Env: X2_THREADS (LightGBM / torch threads, default 8), WJB_WORKERS (TSG processes, default 8),
    #      WJB_WORKER_THREADS (threads per TSG process, default 1), WJB_GPU_FRACTION (default 0.7).
    WSHA=$(sha256sum "$HERE/configs/wjb_protocol.json" | cut -d' ' -f1)
    WBASE=$LOGS/WJB/wjb_${COMMIT}
    mkdir -p "$WBASE"
    case "$MODE" in
      wjbsmoke) STEPS="krr lgbh tsg cabin report"; FLAG="--smoke" ;;
      wjbcpu)   STEPS="krr lgbh tsg"; FLAG="" ;;
      wjbgpu)   STEPS="cabin"; FLAG="" ;;
      wjbfinal) STEPS="report"; FLAG="" ;;
    esac
    if ! mkdir "$WBASE/.$MODE.launch"; then echo "launch already claimed: $MODE (do not retry)"; exit 1; fi
    cd "$HERE"
    nohup bash -c "set -e; for m in $STEPS; do echo \"[WJB] start \$m\"; '$PY' scripts/run_wjb.py --mode \$m --logs '$LOGS' --base '$WBASE' --protocol-sha256 $WSHA $FLAG --threads ${X2_THREADS:-8} --workers ${WJB_WORKERS:-8} --worker-threads ${WJB_WORKER_THREADS:-1} --gpu-fraction ${WJB_GPU_FRACTION:-0.7}; done; echo '[WJB] chain done'" \
      > "$WBASE/$MODE.log" 2>&1 &
    echo $! > "$WBASE/$MODE.pid"
    echo "started WJB chain $MODE pid=$(cat "$WBASE/$MODE.pid") commit=$COMMIT protocol=$WSHA"
    echo "base=$WBASE log=$WBASE/$MODE.log status: bash scripts/status_x2.sh"
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
