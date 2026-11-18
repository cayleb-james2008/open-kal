"""Guards for the paper epoch: retention, arming, parity, requoting and boot state.

Every check is negative in spirit: it fails if a mechanism that is supposed to stop
something stops stopping it. The arming checks are the important ones -- the engine
is one green readiness pass away from live, so each of them feeds it a reason it
must refuse (an environment variable, a promotion, a restart, a stored flag) and
requires `live_armed` to stay false.

Run: .venv/bin/python tests/test_paper_guards.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import arming, parity  # noqa: E402
from openkal.config import (REQUOTE_COOLDOWN_S, Settings,  # noqa: E402
                            entry_quote_allowed)
from openkal.engine import Engine  # noqa: E402
from openkal.store import KEEP_S, SPOT_KEEP_S, Store  # noqa: E402
from openkal.strategy import Params  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append((name, bool(ok), str(detail)))


# --------------------------------------------------------------- 1. retention
def _write_tick(store: Store, ts: float, series: str = "KXBTC15M", **over) -> None:
    row = dict(ts=ts, series=series, ticker=f"{series}-T", yes_bid="0.40",
               yes_ask="0.42", last="0.41", spot=100.0, p_model=0.5, edge=0.0,
               secs_left=600.0, spread=0.02, signal="", can_open=False,
               stake="100", spot_age=0.0, hurdle="0.015", impulse_bps=0.0,
               gate="", strike=99.0, decision="", quote_age=0.0)
    row.update(over)
    store.tick(**row)


def test_prune_boundary() -> None:
    check("prune: the spot window is 14 days", SPOT_KEEP_S == 86400 * 14, SPOT_KEEP_S)
    check("prune: the decision tape keeps its 3-day window", KEEP_S == 86400 * 3, KEEP_S)
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "prune.db")
        now = time.time()
        _write_tick(store, now - 60)
        _write_tick(store, now - KEEP_S - 60)
        _write_tick(store, now - KEEP_S + 60)
        store.spot_batch([
            (now - 60, "BTC-USD", "coinbase", 1.0, None, None),
            (now - SPOT_KEEP_S + 60, "BTC-USD", "coinbase", 1.0, None, None),
            (now - SPOT_KEEP_S - 60, "BTC-USD", "coinbase", 1.0, None, None),
        ])
        store.prune()
        ticks = [r["ts"] for r in store.recent_ticks("KXBTC15M", 0.0, limit=50)]
        spots = [r["ts"] for r in store.spot_ticks("BTC-USD", 0.0, limit=50)]
        check("prune: a tick inside 3 days survives", any(abs(t - (now - 60)) < 1 for t in ticks))
        check("prune: a tick just outside 3 days is deleted",
              not any(abs(t - (now - KEEP_S - 60)) < 1 for t in ticks))
        check("prune: a tick just inside 3 days survives",
              any(abs(t - (now - KEEP_S + 60)) < 1 for t in ticks))
        check("prune: a spot print just inside 14 days survives",
              any(abs(t - (now - SPOT_KEEP_S + 60)) < 1 for t in spots))
        check("prune: a spot print just outside 14 days is deleted",
              not any(abs(t - (now - SPOT_KEEP_S - 60)) < 1 for t in spots))
        check("prune: a fresh spot print survives the decision tape's 3-day window",
              any(abs(t - (now - 60)) < 1 for t in spots))
        store.prune(keep_s=0.0, spot_keep_s=0.0)
        check("prune: an explicit zero window clears both tapes",
              store.recent_ticks("KXBTC15M", 0.0, limit=50) == []
              and store.spot_ticks("BTC-USD", 0.0, limit=50) == [])
        store._db.close()


# ----------------------------------------------------------------- 2. arming
class _MetaStore:
    def __init__(self) -> None:
        self.meta: dict[str, object] = {}

    def set_meta(self, key: str, value: object) -> None:
        self.meta[key] = value

    def get_meta(self, key: str, default: object = None) -> object:
        return self.meta.get(key, default)


def _arm_engine(tmp: str, *, started: float | None = None, live: bool = True,
                arm_live: bool = True, kill_reason: str | None = None) -> Engine:
    engine = Engine.__new__(Engine)
    engine.settings = SimpleNamespace(live=live, arm_live=arm_live)
    engine.live_armed = False
    engine.arm_ts = None
    engine.kill_reason = kill_reason
    engine.started = time.time() if started is None else started
    engine._last_arm_check = 0.0
    engine._arm_lock_noted = False
    engine.arm_path = Path(tmp) / "live-arm.json"
    engine.slots = []
    engine.store = _MetaStore()
    engine.broker = object()
    engine.live = None
    engine.source = "paper"
    engine.supervisor = SimpleNamespace(note=lambda k, v: None)
    return engine


def _try_arm(engine: Engine, *, ready: bool = True, now: float | None = None) -> None:
    """Run the arm path with the functional gate forced green or red."""
    with mock.patch("openkal.readiness.evaluate",
                    return_value={"ready": ready, "blockers": []}), \
         mock.patch("openkal.live_exec.LiveExecutor", lambda ctx, broker: object()):
        asyncio.run(Engine._maybe_arm(engine, time.time() if now is None else now))


def test_arming_lock() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # The environment alone: this is the .env on the box today.
        env_only = _arm_engine(tmp)
        _try_arm(env_only)
        check("arming: OPENKAL_ARM_LIVE plus a green gate does not arm",
              env_only.live_armed is False)

        # A promotion is not an arm.
        promoted = _arm_engine(tmp)
        promoted.supervisor.promoted = True
        promoted.store.set_meta("promoted", True)
        promoted.store.set_meta("promotion", {"verdict": "promoted"})
        _try_arm(promoted)
        check("arming: a promotion or verdict artifact does not arm",
              promoted.live_armed is False)

        # A row persisted by an earlier process is not permission for this one.
        restored = _arm_engine(tmp)
        restored.store.set_meta("live_armed", True)
        restored.store.set_meta("armed_at", time.time())
        _try_arm(restored)
        check("arming: a persisted live_armed row does not arm a new process",
              restored.live_armed is False)

        # A record written before this boot belonged to another process.
        started = time.time()
        arming.operator_arm("cayleb", path=Path(tmp) / "live-arm.json",
                            now=started - 300)
        restarted = _arm_engine(tmp, started=started)
        check("arming: a pre-boot record is refused even while the file exists",
              not arming.armed(arming.read(restarted.arm_path),
                               started=restarted.started, now=time.time()))
        _try_arm(restarted)
        check("arming: a restart with a stale record stays paper",
              restarted.live_armed is False)
        cleared = arming.clear(restarted.arm_path)
        check("arming: boot clears the leftover record",
              cleared is not None and not restarted.arm_path.exists())

        # The operator act, with a green gate.
        armed = _arm_engine(tmp)
        arming.operator_arm("cayleb", path=armed.arm_path)
        _try_arm(armed)
        check("arming: the operator record plus a green gate arms", armed.live_armed is True)
        check("arming: the arm is written to the ledger meta",
              armed.store.meta.get("armed_at") is not None)
        check("arming: the record is consumed so it arms exactly one process",
              not armed.arm_path.exists())

        # One act, one process: the next boot has nothing to consume.
        second = _arm_engine(tmp)
        _try_arm(second)
        check("arming: the next boot with no new act stays paper",
              second.live_armed is False)

        # A killed process never re-arms.
        killed = _arm_engine(tmp, kill_reason="rolling decay: PF 0.80")
        arming.operator_arm("cayleb", path=killed.arm_path)
        _try_arm(killed)
        check("arming: a killed process cannot re-arm", killed.live_armed is False)

        # A paper-configured build cannot be armed by a record.
        paper = _arm_engine(tmp, live=False)
        arming.operator_arm("cayleb", path=paper.arm_path)
        _try_arm(paper)
        check("arming: a record cannot arm a paper-configured build", paper.live_armed is False)

        # An operator record is necessary, not sufficient: readiness still decides.
        gated = _arm_engine(tmp)
        arming.operator_arm("cayleb", path=gated.arm_path)
        _try_arm(gated, ready=False)
        check("arming: a red readiness gate keeps paper even with a record",
              gated.live_armed is False)
        check("arming: a blocked gate does not consume the record", gated.arm_path.exists())

        # An anonymous record is not a person.
        try:
            arming.operator_arm("   ", path=Path(tmp) / "anon.json")
            refused = False
        except ValueError:
            refused = True
        check("arming: an anonymous record is refused", refused)

    # No file may flip the flag around the operator record. Every assignment of
    # `live_armed = True` must sit beside an `arming.consume()`, which is the only
    # thing that proves a person wrote the record for this run.
    assigners: dict[str, bool] = {}
    for path in list((ROOT / "openkal").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        text = path.read_text()
        if "live_armed = True" in text:
            assigners[path.name] = "arming.consume(" in text
    check("arming: every file that can set live_armed also requires the operator record",
          bool(assigners) and all(assigners.values()), assigners)


def test_fresh_engine_is_paper() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fresh.db"
        store = Store(path)
        store.set_meta("live_armed", True)
        store.set_meta("armed_at", time.time())
        engine = Engine(Settings(db_path=path), store)
        check("arming: construction never restores an armed flag", engine.live_armed is False)
        check("boot: a fresh engine is not ready", engine.ready is False)
        check("boot: a fresh engine serves the booting payload",
              Engine.dashboard_payload(engine)["mode"] == "booting")
        store._db.close()


# ------------------------------------------------------------- 3. boot state
def test_boot_payload() -> None:
    class _Booting:
        ready = False
        started = time.time()

        def snapshot(self) -> dict:
            return {"mode": "paper",
                    "account": {"equity": "-42.54", "realized": "-42.54"}}

    class _Ready(_Booting):
        ready = True

    payload = Engine.dashboard_payload(_Booting())
    check("boot: /api/state says booting before the loop is ready",
          payload["mode"] == "booting")
    check("boot: the booting payload carries no stale equity",
          "account" not in payload and "equity" not in json.dumps(payload))
    check("boot: once the loop is ready the real snapshot is served",
          Engine.dashboard_payload(_Ready())["mode"] == "paper")

    cli_src = (ROOT / "openkal" / "cli.py").read_text()
    check("boot: the dashboard publisher asks for the boot-aware payload",
          "dashboard_payload()" in cli_src and "engine.snapshot()" not in cli_src)


# ------------------------------------------------------------ 4. signal parity
def _row(ts: float, **over) -> dict:
    row = dict(ts=ts, series="KXBTC15M", ticker="KXBTC15M-T", yes_bid="0.40",
               yes_ask="0.42", last="0.41", spot=100.0, p_model=0.62, edge=0.0,
               secs_left=500.0, spread=0.02, signal="F2_taker_1s", can_open=True,
               stake="100", spot_age=0.2, hurdle="0.015", impulse_bps=0.0,
               gate="", strike=99.0, quote_age=0.1)
    row.update(over)
    return row


def test_signal_parity() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "parity.db")
        params = Params()
        now = time.time()

        good = _row(now - 60)
        decided = parity.canonical_decision(good, params)
        check("parity: the fixture's canonical decision is an entry",
              decided is not None and decided.action == "entry", decided)
        good["decision"] = parity.decision_tag(decided) if decided else ""
        store.tick(**good)
        first = parity.lane_parity("KXBTC15M", store.recent_ticks("KXBTC15M", 0.0), params)
        check("parity: a recorded decision re-derives identically",
              first["ok"] and first["checked"] == 1 and first["mismatches"] == 0, first)

        bad = _row(now - 30)
        bad["decision"] = "hold:-"
        store.tick(**bad)
        result = parity.lane_parity("KXBTC15M", store.recent_ticks("KXBTC15M", 0.0), params)
        check("parity: a tampered decision fails loudly",
              (not result["ok"]) and result["mismatches"] == 1, result)
        check("parity: the mismatch names the tick and both decisions",
              bool(result["examples"])
              and abs(result["examples"][0]["ts"] - (now - 30)) < 1
              and result["examples"][0]["live"] == "hold:-"
              and result["examples"][0]["canonical"].startswith("entry:"),
              result["examples"])

        skipped_row = _row(now - 10, decision="")
        store.tick(**skipped_row)
        counted = parity.lane_parity("KXBTC15M", store.recent_ticks("KXBTC15M", 0.0), params)
        check("parity: a tick with no recorded decision is counted as skipped, not matched",
              counted["skipped"] == 1 and counted["checked"] == 2, counted)
        store._db.close()


# -------------------------------------------------------------- 5. re-quoting
def _lane(series: str, asset: str, **over) -> SimpleNamespace:
    lane = SimpleNamespace(
        cfg=SimpleNamespace(series=series, asset=asset, feed_symbol="BTC-USD"),
        halted=False, position=None, window_entries=0, window_halted=False,
        last_entry_quote_ts=0.0,
        params=SimpleNamespace(f2_max_window_trades=3, max_window_trades=2),
    )
    for key, value in over.items():
        setattr(lane, key, value)
    return lane


def test_requote_cooldown() -> None:
    now = 10_000.0
    check("requote: the cooldown is 90s", REQUOTE_COOLDOWN_S == 90.0, REQUOTE_COOLDOWN_S)
    check("requote: a lane that has never quoted may quote",
          entry_quote_allowed(0.0, now))
    check("requote: a quote 89.9s old blocks the next one",
          not entry_quote_allowed(now - 89.9, now))
    check("requote: a quote exactly 90s old allows the next one",
          entry_quote_allowed(now - 90.0, now))
    check("requote: a clock step backwards refuses rather than allows",
          not entry_quote_allowed(now + 10.0, now))

    engine = Engine.__new__(Engine)
    engine.settings = SimpleNamespace(max_concurrent_positions=4,
                                      max_concurrent_crypto=2,
                                      max_same_direction_crypto=2)
    btc = _lane("KXBTC15M", "BTC")
    eth = _lane("KXETH15M", "ETH")
    engine.slots = [btc, eth]
    signal = SimpleNamespace(name="F2_taker_1s")

    check("requote: the first entry quote of a window is allowed",
          Engine._can_open(engine, btc, signal, False, "YES", False, now))
    btc.last_entry_quote_ts = now
    check("requote: a second new quote 5s later is refused",
          not Engine._can_open(engine, btc, signal, False, "YES", False, now + 5))
    check("requote: another lane may still quote",
          Engine._can_open(engine, eth, signal, False, "YES", False, now + 5))
    check("requote: the second quote is still refused at 89.9s",
          not Engine._can_open(engine, btc, signal, False, "YES", False, now + 89.9))
    check("requote: the same lane may quote again at 90s",
          Engine._can_open(engine, btc, signal, False, "YES", False, now + 90.0))
    btc.halted = True
    check("requote: a halted lane never quotes, cooldown or not",
          not Engine._can_open(engine, btc, signal, False, "YES", False, now + 200))


def main() -> int:
    test_prune_boundary()
    test_arming_lock()
    test_fresh_engine_is_paper()
    test_boot_payload()
    test_signal_parity()
    test_requote_cooldown()
    width = max(len(name) for name, _, _ in RESULTS)
    failed = 0
    for name, ok, detail in RESULTS:
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} paper-guard checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
