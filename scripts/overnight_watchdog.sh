#!/bin/bash
# Keep the overnight supervisor alive until its window closes.
#
# The supervisor is the only thing standing between a live engine and an unattended
# night, so it is launched in its own session (setsid, immune to a group signal) and
# restarted if it exits for any reason before its deadline. A crash at 3 a.m. must
# not silently leave the breaker unwatched.
#
#   bash scripts/overnight_watchdog.sh --until "2026-11-17T04:30:00" [supervisor args]
set -uo pipefail
cd "$(dirname "$0")/.."
log=/tmp/opencode/supervisor.log
until [ "$#" -gt 0 ]; do
  echo "usage: overnight_watchdog.sh --until <iso> [supervisor args]" >&2
  exit 2
done
# The shell has already stripped the quotes by the time the argument arrives, so the
# value is the argument that follows --until, not a quoted string.
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
while [ "$(date +%s)" -lt "$until_ts" ]; do
  setsid nohup .venv/bin/python scripts/overnight_supervisor.py "$@" >> "$log" 2>&1
  rc=$?
  if [ "$(date +%s)" -ge "$until_ts" ]; then
    echo "watchdog: window closed (supervisor rc=$rc)" | tee -a "$log"
    exit 0
  fi
  echo "watchdog: supervisor exited rc=$rc; restarting in 30s" | tee -a "$log"
  sleep 30
done
