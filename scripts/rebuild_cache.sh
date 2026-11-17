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
# The market metadata first: a window that recorded on the tape but has no settled
# market record has no strike and no outcome, so the study would silently skip it.
echo "=== market metadata $(date -Iseconds) ==="
.venv/bin/python scripts/refresh_markets_ready.py --timeout 180
echo "=== index cache $(date -Iseconds) ==="
.venv/bin/python scripts/evolve.py cache
echo "=== book cache $(date -Iseconds) ==="
.venv/bin/python scripts/book_cache.py KXBTC15M KXETH15M KXSOL15M KXXRP15M
echo "=== discovery $(date -Iseconds) ==="
.venv/bin/python scripts/subsec_discovery.py
echo "=== done $(date -Iseconds) ==="
