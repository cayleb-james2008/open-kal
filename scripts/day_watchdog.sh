#!/bin/bash
# Keep the daytime supervisor alive until 18:00 EST.
#
# The supervisor is the only thing watching a possibly-live engine, and the only
# thing that scores the forward slice, so it is launched in its own session
# (setsid, immune to a group signal) and restarted if it exits before its
# deadline. It is resumable: `data/day-state.json` records the decision, so a
# restart after promotion goes straight back to supervision instead of
# re-promoting, re-canarying or re-scoring the slice.
#
# The deadline is not a stop signal. If the supervisor dies during the last
# restart sleep, the loop would otherwise end at 18:00 with no report. A closing
# run is started when the window closes without a fresh `data/evening-report.md`;
# the supervisor booted after 18:00 sees the window closed, writes the handoff
# and exits. A pgrep guard makes a second watchdog harmless: it waits for the
# supervisor that is already running instead of starting a duplicate.
#
#   bash scripts/day_watchdog.sh --until "2026-11-17T18:00:00" --since 1794909600
set -uo pipefail
cd "$(dirname "$0")/.."
# The supervisor writes its own data/day-supervisor.log; this file holds the
# watchdog's messages and any interpreter traceback the supervisor could not log
# itself. Pointing both at the same path duplicated every line.
log=data/day-watchdog.log
out=data/day-supervisor.out
report=data/evening-report.md
sleep_s=${OPENKAL_WATCHDOG_SLEEP:-30}
# Scoped to the command this watchdog starts: a supervisor belonging to another
# directory must not make this one wait. The drill overrides it.
sup_pattern=${OPENKAL_SUPERVISOR_PATTERN:-'^\.venv/bin/python scripts/day_supervisor\.py'}
if [ -n "${OPENKAL_SUPERVISOR_CMD:-}" ]; then
  read -r -a sup <<< "$OPENKAL_SUPERVISOR_CMD"
else
  sup=(.venv/bin/python scripts/day_supervisor.py)
fi
if [ "$#" -eq 0 ]; then
  echo "usage: day_watchdog.sh --until <iso> [supervisor args]" >&2
  exit 2
fi
until_iso=""
prev=""
for a in "$@"; do
  if [ "$prev" = "--until" ]; then until_iso="$a"; break; fi
  prev="$a"
done
until_ts=$(date -d "$until_iso" +%s 2>/dev/null || echo 0)
if [ "${until_ts:-0}" -eq 0 ]; then
  echo "could not read --until (got '$until_iso')" >&2
  exit 2
fi
echo "watchdog: supervising until $(date -d "@$until_ts")" | tee -a "$log"
waited=0
while true; do
  now=$(date +%s)
  [ "$now" -ge "$until_ts" ] && break
  # A second watchdog (or a hand-started supervisor) must not become a second
  # scorer. Wait for the one that is already running.
  if pgrep -f "$sup_pattern" >/dev/null 2>&1; then
    if [ "$waited" -eq 0 ]; then
      echo "watchdog: a supervisor is already running; waiting for it" | tee -a "$log"
      waited=1
    fi
    sleep "$sleep_s"
    continue
  fi
  waited=0
  setsid nohup "${sup[@]}" "$@" >> "$out" 2>&1
  rc=$?
  if [ "$(date +%s)" -ge "$until_ts" ]; then
    echo "watchdog: window closed (supervisor rc=$rc)" | tee -a "$log"
    break
  fi
  echo "watchdog: supervisor exited rc=$rc; restarting in ${sleep_s}s" | tee -a "$log"
  sleep "$sleep_s"
done
# The supervisor is still writing its handoff, or the loop crossed 18:00 after a
# crash. Never start a duplicate: wait briefly for the live one, and only then
# check whether a closing run is needed.
if pgrep -f "$sup_pattern" >/dev/null 2>&1; then
  echo "watchdog: supervisor still running at the deadline; waiting for it to close out" \
    | tee -a "$log"
  for _ in $(seq 60); do
    pgrep -f "$sup_pattern" >/dev/null 2>&1 || break
    sleep 1
  done
  if pgrep -f "$sup_pattern" >/dev/null 2>&1; then
    echo "watchdog: supervisor still running; not starting a second" | tee -a "$log"
    exit 0
  fi
fi
if [ ! -f "$report" ] || [ "$(stat -c %Y "$report")" -lt "$until_ts" ]; then
  echo "watchdog: window closed without a fresh report; one closing run" | tee -a "$log"
  setsid nohup "${sup[@]}" "$@" >> "$out" 2>&1
  echo "watchdog: closing run rc=$?" | tee -a "$log"
else
  echo "watchdog: 18:00 report is present and fresh" | tee -a "$log"
fi
exit 0
