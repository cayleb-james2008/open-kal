"""Parity harness: force fills through the *same* engine, then replay them.

The deployed strategy refuses almost every trade on this tape (that is the honest
result), so a forward epoch can contain no fills and parity cannot be exercised.
This harness runs the real engine, real feed and real execution core on a scratch
database with a deliberately permissive impulse program. Its only purpose is to
produce a ledger with fills that `tests/test_gate_controls.py` can reconcile 1:1.

It never touches the production database and can never arm a venue order.

    .venv/bin/python scripts/parity_harness.py --minutes 8 --target 6
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.config import load_settings  # noqa: E402
from openkal.engine import Engine  # noqa: E402
from openkal.store import Store  # noqa: E402
from openkal.strategy import Params  # noqa: E402

DB = ROOT / "data" / "openkal-parity.db"


async def run(minutes: float, target: int) -> int:
    settings = load_settings()
    settings.db_path = DB
    settings.allow_live = False
    settings.arm_live = False

    if DB.exists():
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(DB) + suffix)
            if p.exists():
                p.unlink()
    store = Store(DB)
    now = time.time()
    store.set_meta("epoch_start", now)
    store.set_meta("epoch_equity", "262.31")
    engine = Engine(settings, store)
    # Deliberately permissive: the point is to exercise the ledger, not to trade well.
    # The real program refuses nearly every fill on this tape, which is the honest result.
    program = Params(
        impulse_lookback_s=30.0, impulse_min_bps=8.0, impulse_max_bps=40.0,
        impulse_stale_share=1.0, impulse_min_edge=-1.0, max_spread=Decimal("0.20"),
        min_price=Decimal("0.05"), max_price=Decimal("0.95"),
        impulse_max_hold_s=30.0, min_seconds_left=90.0,
    )
    for slot in engine.slots:
        slot.params = program
    store.set_meta("deployed_params",
                   {s.cfg.series: s.params.to_json() for s in engine.slots})

    task = asyncio.create_task(engine.run())
    started = time.time()
    last = 0
    try:
        while time.time() - started < minutes * 60:
            await asyncio.sleep(5)
            n = int(store.stats()["trades"])
            if n != last:
                print(f"{time.time() - started:6.1f}s trades={n} "
                      f"pnl={store.stats()['pnl']:.2f}", flush=True)
                last = n
            if n >= target:
                break
    finally:
        engine.stop()
        try:
            await asyncio.wait_for(task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()
    print(f"harness done: trades={store.stats()['trades']} pnl={store.stats()['pnl']:.2f} "
          f"ticks={store._db.execute('select count(*) from ticks').fetchone()[0]}")  # noqa: SLF001
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--target", type=int, default=6)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args.minutes, args.target)))
