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
