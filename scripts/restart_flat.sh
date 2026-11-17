#!/bin/bash
# Restart run.py once, and only if the venue is flat (0 positions, 0 resting orders).
# A boot with anything open flattens it at market, so this refuses instead.
# `--new-epoch` also resets epoch_start/epoch_equity to now / the live balance.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import asyncio, sys
from openkal.auth import KalshiClient
from openkal.config import load_settings

async def main() -> int:
    c = KalshiClient(load_settings())
    try:
        s1, p = await c.get("/portfolio/positions")
        s2, o = await c.get("/portfolio/orders", {"status": "resting"})
    finally:
        await c.close()
    if s1 != 200 or s2 != 200:
        print(f"venue read failed ({s1}/{s2}): not restarting"); return 1
    held = [x["ticker"] for x in p.get("market_positions", [])
            if float(x.get("position_fp") or 0) != 0]
    resting = o.get("orders", [])
    if held or resting:
        print(f"NOT FLAT: positions {held}, {len(resting)} resting order(s): not restarting")
        return 1
    print("venue flat"); return 0

sys.exit(asyncio.run(main()))
PY
old=$(cat data/engine.pid 2>/dev/null || true)
if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
  kill "$old"
  for _ in $(seq 20); do kill -0 "$old" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$old" 2>/dev/null && { echo "pid $old did not exit: not starting a second engine"; exit 1; }
fi
if [ "${1:-}" = "--new-epoch" ]; then
  # Start a clean epoch at the live balance. No trade or fill row is touched; the old
  # epoch's rows stay in the ledger and the old markers are logged here.
  .venv/bin/python - <<'PY' | tee -a data/engine.log
import asyncio, time
from openkal.auth import KalshiClient
from openkal.config import load_settings
from openkal.store import Store

async def balance() -> str:
    c = KalshiClient(load_settings())
    try:
        status, d = await c.get("/portfolio/balance")
    finally:
        await c.close()
    if status != 200 or not d.get("balance_dollars"):
        raise SystemExit(f"balance read failed ({status}): epoch not changed")
    return str(d["balance_dollars"])

bal = asyncio.run(balance())
store = Store(load_settings().db_path)
old = (store.get_meta("epoch_start"), store.get_meta("epoch_equity"))
now = time.time()
store.set_meta("epoch_start", now)
store.set_meta("epoch_equity", bal)
print(f"new epoch: start {now:.3f} equity {bal} (previous start {old[0]} equity {old[1]})")
PY
fi
echo "===== session $(date -Iseconds) detached daemon start (prev pid ${old:-none}) =====" >> data/engine.log
nohup setsid .venv/bin/python run.py --port 8787 >> data/engine.log 2>&1 < /dev/null &
echo $! > data/engine.pid
echo "started pid $(cat data/engine.pid)"
