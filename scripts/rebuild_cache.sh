#!/bin/bash
# Rebuild the sub-second cache from the production tape, then re-run the frozen
# discovery on it. Used by the overnight supervisor at the rebuild hour and by
# hand when the tape has grown materially.
#
# The existing book caches are moved aside rather than deleted, and the trap puts
# them back if the rebuild fails: an unattended rebuild that dies halfway must not
# leave the study with no tape at all.
set -euo pipefail
cd "$(dirname "$0")/.."
stamp=$(date +%Y%m%d-%H%M%S)
backup=""
if compgen -G "data/evolve-cache/book_*.npz" > /dev/null; then
  backup="data/evolve-cache/prev-$stamp"
  mkdir -p "$backup"
  mv data/evolve-cache/book_*.npz "$backup/"
  echo "=== previous book cache moved to $backup ==="
fi
restore() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ -n "$backup" ] && compgen -G "$backup/book_*.npz" > /dev/null; then
    echo "=== rebuild failed ($rc); restoring the previous book cache ==="
    cp "$backup"/book_*.npz data/evolve-cache/
  fi
  exit "$rc"
}
trap restore EXIT
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
trap - EXIT
