#!/usr/bin/env python3
"""Follow the live engine and the venue tape, and log every change.

Read-only: no orders, no state writes. Every line is either a `/api/state`
snapshot of the four lanes or a print/settlement the venue reported that this
process has not logged before.

    setsid nohup .venv/bin/python scripts/live_monitor.py >> data/live-monitor.log 2>&1 &

Log format, one line per event:

    STATE 14:52:03 armed=True slots=KXBTC15M:abstain/570s/sp0.0100 ...
    FILL  14:52:03 KXBTC15M-... buy yes 42.00 @ 0.7800 fee=0.000000 taker=False
    STL   14:52:03 KXBTC15M-... result=yes yes=84 no=84 fee=0.7904
    ORDER 14:52:03 new/resting KXXRP...
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal.auth import KalshiClient  # noqa: E402
from openkal.config import load_settings  # noqa: E402

STATE_URL = "http://127.0.0.1:8787/api/state"


def log(kind: str, message: str) -> None:
    print(f"{kind:5s} {time.strftime('%H:%M:%S')} {message}", flush=True)


def state_line(snapshot: dict) -> str:
    lanes = []
    for slot in snapshot.get("slots") or []:
        order = slot.get("order") or {}
        pos = slot.get("position") or {}
        lanes.append(
            f"{slot['id']}:{slot.get('state')}"
            f"/T{round(slot.get('secs_left') or 0)}s"
            f"/sp{slot.get('spread')}"
            f"/px{slot.get('yes_bid')}-{slot.get('yes_ask')}"
            + (f"/ORD {order.get('side')} {order.get('price')}x{order.get('contracts')}"
               if order else "")
            + (f"/POS {pos.get('side')} {pos.get('entry')}x{pos.get('contracts')}"
               if pos else "")
        )
    return (f"armed={snapshot.get('live_armed')} mode={snapshot.get('mode')} "
            f"bal={snapshot.get('venue_balance')} " + " ".join(lanes))


async def main() -> int:
    settings = load_settings()
    client = KalshiClient(settings)
    seen_fills: set[str] = set()
    seen_settlements: set[str] = set()
    seen_orders: set[str] = set()
    last_state = ""
    next_tape = 0.0
    log("START", f"monitor pid {__import__('os').getpid()} env={settings.env_name}")
    try:
        while True:
            try:
                async with httpx.AsyncClient(timeout=4.0) as http:
                    response = await http.get(STATE_URL)
                snapshot = response.json()
                line = state_line(snapshot)
                # The lane line carries a countdown, so dedupe on the interesting part.
                key = line.split("/T")[0] + line.split("bal=")[-1].split(" ")[0]
                if line != last_state and (key != last_state or time.time() % 30 < 1):
                    log("STATE", line)
                    last_state = line
            except Exception as exc:  # noqa: BLE001 - the dashboard may restart under us
                log("DOWN ", f"{type(exc).__name__}: {exc}")
            if time.time() >= next_tape:
                next_tape = time.time() + 10
                try:
                    _s, fills = await client.get("/portfolio/fills", {"limit": 50})
                    for f in fills.get("fills") or []:
                        fid = f.get("fill_id") or ""
                        if fid in seen_fills:
                            continue
                        seen_fills.add(fid)
                        price = (f.get("yes_price_dollars") if f.get("side") == "yes"
                                 else f.get("no_price_dollars"))
                        log("FILL ", f"{f.get('ticker')} {f.get('action')} {f.get('side')} "
                                     f"{f.get('count_fp')} @ {price} "
                                     f"fee={f.get('fee_cost')} taker={f.get('is_taker')} "
                                     f"oid={(f.get('order_id') or '')[:8]}")
                    _s, stl = await client.get("/portfolio/settlements", {"limit": 50})
                    for s in stl.get("settlements") or []:
                        key = f"{s.get('ticker')}:{s.get('settled_time')}"
                        if key in seen_settlements:
                            continue
                        seen_settlements.add(key)
                        log("STL  ", f"{s.get('ticker')} result={s.get('market_result')} "
                                     f"yes={s.get('yes_count_fp')} no={s.get('no_count_fp')} "
                                     f"fee={s.get('fee_cost')} value={s.get('value')}")
                    _s, orders = await client.get("/portfolio/orders",
                                                  {"status": "resting", "limit": 50})
                    current = {o.get("order_id") for o in orders.get("orders") or []}
                    for o in orders.get("orders") or []:
                        if o.get("order_id") in seen_orders:
                            continue
                        seen_orders.add(o.get("order_id"))
                        log("ORDER", f"{o.get('ticker')} {o.get('side')} "
                                     f"{o.get('initial_count_fp')} @ "
                                     f"{o.get('yes_price_dollars')} "
                                     f"oid={(o.get('order_id') or '')[:8]}")
                    for gone in seen_orders - current:
                        pass  # a cancel shows up as a fill or a settlement; no line needed
                except Exception as exc:  # noqa: BLE001 - keep following the engine
                    log("TAPE ", f"read failed {type(exc).__name__}: {exc}")
            await asyncio.sleep(5)
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(0)
