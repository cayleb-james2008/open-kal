#!/bin/bash
# Rebuild the sub-second cache from the production tape, then re-run the frozen
# discovery on it. Used by the overnight supervisor at the rebuild hour and by
# hand when the tape has grown materially.
#
# The book files are NOT moved aside before the rebuild. That used to be the
# failure mode: the only copy was renamed into prev-<stamp>/ first, so a rebuild
# killed between the move and the new write (a supervisor restart, a full disk)
# left the study with no book_*.npz at all -- which is exactly what happened on
# 2026-11-17 at 14:24. scripts/book_cache.py now writes each lane beside the live
# file and renames it into place, so the last good book survives a failed rebuild
# and there is never a window with no tape.
set -euo pipefail
cd "$(dirname "$0")/.."
# One whole rebuild at a time. `scripts/book_cache.py` takes its own lock around the
# export, but that leaves the other three steps unprotected: on 2026-11-18 at 06:00:36
# the daemon's scheduled rebuild started while a hand-run one was mid-book, and the
# two scanned the 31 GB tape through refresh_markets_ready, evolve cache and
# subsec_discovery at the same time. The lock covers the whole chain, so the second
# invocation leaves the tape alone and exits 0 -- the caller should not read a skip
# as a failed rebuild. flock releases on process death, so a killed rebuild cannot
# leave a stale lock behind.
exec 9>data/.rebuild-cache.lock
if ! flock -n 9; then
  echo "=== rebuild already in progress $(date -Iseconds) -- skipping ==="
  exit 0
fi
# The market metadata first: a window that recorded on the tape but has no settled
# market record has no strike and no outcome, so the study would silently skip it.
echo "=== market metadata $(date -Iseconds) ==="
.venv/bin/python scripts/refresh_markets_ready.py --timeout 180
echo "=== index cache $(date -Iseconds) ==="
.venv/bin/python scripts/evolve.py cache
echo "=== book cache $(date -Iseconds) ==="
# --force: a rebuild that skips every lane because its file exists is not a rebuild.
# Each lane is still written beside the live file and renamed into place, so the
# last good book survives a killed run.
.venv/bin/python scripts/book_cache.py --force KXBTC15M KXETH15M KXSOL15M KXXRP15M
echo "=== discovery $(date -Iseconds) ==="
.venv/bin/python scripts/subsec_discovery.py
echo "=== done $(date -Iseconds) ==="
