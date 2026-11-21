#!/usr/bin/env bash
# One-command X2 status for Cursor/the user: process alive?, last heartbeat, completed/failed, log tail.
DATA=${X2_DATA_ROOT:-/home/XT/nuist/pythoncode/TimeDARTFirst}
HERE=$(cd "$(dirname "$0")/.." && pwd)
COMMIT=$(git -C "$HERE" rev-parse --short=7 HEAD)
BASE=$DATA/outputs/logs/X2/x2_${COMMIT}
echo "== X2 runs under $BASE"
for MODE in smoke a b report a2 report2; do
  [ -e "$BASE/$MODE.pid" ] || continue
  PID=$(cat "$BASE/$MODE.pid")
  STATE=exited
  if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    STATE=pid_alive_unverified
    if [ -r "/proc/$PID/cmdline" ]; then
      mapfile -d '' -t ARGV < "/proc/$PID/cmdline"
      for ((i=0; i<${#ARGV[@]}-1; i++)); do
        if [ "${ARGV[i]}" = --output ] && [ "${ARGV[i+1]}" = "$BASE/$MODE" ]; then STATE=running; fi
      done
    fi
  fi
  RESULT=-
  [ -e "$BASE/$MODE/failed.json" ] && RESULT=FAILED
  for DONE in "$BASE/$MODE/completed.json" "$BASE/$MODE/A/completed.json" "$BASE/$MODE/B/completed.json" "$BASE/$MODE/A2/completed.json"; do
    [ -e "$DONE" ] && [ "$MODE" != smoke ] && RESULT=COMPLETED
  done
  [ -e "$BASE/$MODE/smoke_passed.json" ] && RESULT=SMOKE_PASSED
  [ -e "$BASE/$MODE/failed.json" ] && RESULT=FAILED
  echo "-- $MODE pid=$PID $STATE result=$RESULT"
  [ -e "$BASE/$MODE/heartbeat.json" ] && cat "$BASE/$MODE/heartbeat.json"
  [ -e "$BASE/$MODE/failed.json" ] && cat "$BASE/$MODE/failed.json"
  tail -n 3 "$BASE/$MODE.log" 2>/dev/null
done
X3BASE=$DATA/outputs/logs/X3/x3_${COMMIT}
if [ -d "$X3BASE" ]; then
  echo "== X3 runs under $X3BASE"
  for CHAIN in x3smoke x3gpu x3q4 x3final; do
    [ -e "$X3BASE/$CHAIN.pid" ] || continue
    PID=$(cat "$X3BASE/$CHAIN.pid"); STATE=exited
    # R1-S2: a reused PID is not "running" unless its command line is still the X3 chain
    if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null \
       && tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q "run_x3.py"; then STATE=running; fi
    echo "-- chain $CHAIN pid=$PID $STATE"; tail -n 3 "$X3BASE/$CHAIN.log" 2>/dev/null
  done
  for D in "$X3BASE"/*/; do
    [ -d "$D" ] || continue
    R=running_or_not_started
    [ -e "$D/completed.json" ] && R=COMPLETED
    [ -e "$D/failed.json" ] && R=FAILED
    echo "   $(basename "$D"): $R"
    [ -e "$D/failed.json" ] && cat "$D/failed.json"
  done
fi
X4BASE=$DATA/outputs/logs/X4/x4_${COMMIT}
if [ -d "$X4BASE" ]; then
  echo "== X4 runs under $X4BASE"
  for CHAIN in x4smoke x4gpu x4d0 x4final; do
    [ -e "$X4BASE/$CHAIN.pid" ] || continue
    PID=$(cat "$X4BASE/$CHAIN.pid"); STATE=exited
    if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null \
       && tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q "run_x4.py"; then STATE=running; fi
    echo "-- chain $CHAIN pid=$PID $STATE"; tail -n 3 "$X4BASE/$CHAIN.log" 2>/dev/null
  done
  for D in "$X4BASE"/*/; do
    [ -d "$D" ] || continue
    R=running_or_not_started
    [ -e "$D/completed.json" ] && R=COMPLETED
    [ -e "$D/failed.json" ] && R=FAILED
    echo "   $(basename "$D"): $R"
    [ -e "$D/failed.json" ] && cat "$D/failed.json"
  done
fi
X5BASE=$DATA/outputs/logs/X5/x5_${COMMIT}
if [ -d "$X5BASE" ]; then
  echo "== X5 runs under $X5BASE"
  for MODE in x5smoke x5run; do
    [ -e "$X5BASE/$MODE.pid" ] || continue
    PID=$(cat "$X5BASE/$MODE.pid"); STATE=exited
    if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null \
       && tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null | grep -q "run_x5.py"; then STATE=running; fi
    echo "-- $MODE pid=$PID $STATE"; tail -n 3 "$X5BASE/$MODE.log" 2>/dev/null
  done
  for D in "$X5BASE"/*/; do
    [ -d "$D" ] || continue
    R=running_or_not_started
    [ -e "$D/completed.json" ] && R=COMPLETED
    [ -e "$D/smoke_passed.json" ] && R=SMOKE_PASSED
    [ -e "$D/failed.json" ] && R=FAILED
    echo "   $(basename "$D"): $R"
    [ -e "$D/heartbeat.json" ] && cat "$D/heartbeat.json"
    [ -e "$D/failed.json" ] && cat "$D/failed.json"
  done
fi
