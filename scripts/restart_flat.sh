#!/bin/bash
# Restart the engine once, and only if the venue is flat (0 positions, 0 resting orders).
# A boot with anything open flattens it at market, so this refuses instead.
# The engine is the `openkal-engine.service` systemd user unit, not a child of this
# shell: Restart=on-failure and linger keep it alive after the terminal that started
# it is gone. This script is the only start path, so the flat gate always runs first.
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
unit=openkal-engine.service
# Stop the service if systemd owns it, then any stray pid from an older nohup
# start, so only one engine can ever bind 8787.
old=$(cat data/engine.pid 2>/dev/null || true)
if systemctl --user is-active --quiet "$unit"; then
  systemctl --user stop "$unit"
fi
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
echo "===== session $(date -Iseconds) systemd user start (prev pid ${old:-none}) =====" >> data/engine.log
# An operator restart must not be blocked by the crash-loop limiter.
systemctl --user reset-failed "$unit" 2>/dev/null || true
systemctl --user restart "$unit"
new=""
for _ in $(seq 25); do
  new=$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null || true)
  [ -n "$new" ] && [ "$new" != "0" ] && break
  sleep 0.2
done
case "$new" in
  ''|0) echo "unit $unit did not report a main pid"; systemctl --user status "$unit" --no-pager | head -20; exit 1 ;;
esac
echo "$new" > data/engine.pid
# "started" has to mean the console answers, not that a pid exists.
for _ in $(seq 60); do
  curl -sf --max-time 2 http://127.0.0.1:8787/api/state > /dev/null && break
  sleep 0.5
done
if ! curl -sf --max-time 2 http://127.0.0.1:8787/api/state > /dev/null; then
  echo "engine pid $new (unit $unit) is not answering /api/state"
  systemctl --user status "$unit" --no-pager | head -20
  exit 1
fi
echo "started pid $new (unit $unit, /api/state answering)"
