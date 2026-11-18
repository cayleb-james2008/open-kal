#!/usr/bin/env python3
"""Age `ws_book` and `ws_trade` out without stopping the engine.

`Store.prune_ws` deletes in committed chunks, so this can run against the live
production database while the recorder writes to it. One pass by default;
`--passes` repeats until the backlog is drained, and `--watch` keeps a bounded
pass running every `--interval` seconds.

    .venv/bin/python scripts/prune_ws.py --keep-days 3 --passes 4 --budget-s 30
    .venv/bin/python scripts/prune_ws.py --watch

No VACUUM: the file will not shrink, but once the 3-day window is full the free
pages are reused by new rows and the file stops growing. Shrinking it is a flat
restart job, not something to run against the live writer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.config import load_settings  # noqa: E402
from openkal.store import Store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="database path (default: settings.db_path)")
    ap.add_argument("--keep-days", type=float, default=3.0)
    ap.add_argument("--chunk", type=int, default=5000,
                    help="rows deleted per committed transaction")
    ap.add_argument("--budget-s", type=float, default=30.0,
                    help="max seconds per pass; 0 = no limit")
    ap.add_argument("--passes", type=int, default=1,
                    help="passes before exiting; 0 = until drained")
    ap.add_argument("--watch", action="store_true",
                    help="run a bounded pass every --interval seconds")
    ap.add_argument("--interval", type=float, default=300.0)
    args = ap.parse_args(argv)
    if args.keep_days <= 0:
        print("--keep-days must be > 0 (0 would delete the live window)", file=sys.stderr)
        return 2
    db = Path(args.db) if args.db else load_settings().db_path
    store = Store(db)
    keep_s = args.keep_days * 86400.0
    total = {"ws_book": 0, "ws_trade": 0}
    passes = 0
    started = time.time()
    try:
        while True:
            got = store.prune_ws(keep_s=keep_s, chunk=args.chunk,
                                 budget_s=args.budget_s)
            total["ws_book"] += got["ws_book"]
            total["ws_trade"] += got["ws_trade"]
            passes += 1
            print(json.dumps({
                "pass": passes,
                "deleted": got,
                "total": total,
                "db_bytes": db.stat().st_size if db.exists() else 0,
            }), flush=True)
            if args.watch:
                time.sleep(args.interval)
                continue
            if not any(got.values()):
                break
            if args.passes and passes >= args.passes:
                break
    except KeyboardInterrupt:
        pass
    print(json.dumps({
        "summary": total,
        "passes": passes,
        "elapsed_s": round(time.time() - started, 1),
        "db_bytes": db.stat().st_size if db.exists() else 0,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
