"""The hot execution plane: discovery, slots, paper fills, ledger, SHR hookup.

One asyncio loop owns all mutable state. The verifier runs beside it and can never
reach an order path; the dashboard only ever reads a snapshot.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import features
from .auth import KalshiClient, money
from .config import ROOT, SeriesConfig, Settings
from .decision_gate import LayaGate, verdict_payload
from .execution import Executor, RestingExit, RestingOrder
from .feeds import SpotFeed
from .kalshi_ws import INDEX_BY_SERIES, KalshiWS
from .shr import Supervisor
from .store import Store
from .strategy import (Decision, Params, Position, Quote, contracts_for, decide,
                       with_persistence)

TICK = Decimal("0.01")


def parse_ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _last_point(points: list[tuple[float, float]], now: float,
                max_age_s: float) -> float | None:
    """The last point on a tape at or before `now`, if it is fresh enough.

    None rather than a stale value: a settlement price read off a tape that has
    stopped moving is not a price, it is a guess.
    """
    for ts, value in reversed(points):
        if ts <= now:
            return float(value) if (now - ts) <= max_age_s else None
    return None


@dataclass
class SlotRuntime:
    cfg: SeriesConfig
    params: Params
    bankroll: Decimal
    peak_bankroll: Decimal

    market: dict[str, Any] | None = None
    quote: Quote | None = None
    spot: float = 0.0
    p_up: float = 0.5
    sigma: float = 0.0
    decision: Decision | None = None
    position: Position | None = None
    order: RestingOrder | None = None
    exit: RestingExit | None = None

    state: str = "boot"
    reason: str = "waiting for a market"
    consecutive_losses: int = 0
    halted: bool = False
    flatten_requested: bool = False
    wins: int = 0
    losses: int = 0
    pnl_today: Decimal = Decimal("0")
    ece: float = 0.0
    pairs: list[tuple[float, int]] = field(default_factory=list)
    tape: list[dict[str, float]] = field(default_factory=list)
    # Last quote seen per contract. The roll needs the *old* window's last price, and
    # by the time the new market is discovered `slot.quote` may already point at it.
    quotes: dict[str, Quote] = field(default_factory=dict)
    last_trade: dict[str, Any] | None = None
    window_open_ts: float = 0.0
    last_tick_store: float = 0.0
    gate: dict[str, Any] | None = None
    signal: str = ""
    signal_detail: str = ""
    gate_ts: float = 0.0
    weak_ticks: int = 0
    edge_streak: int = 0
    gate_ticker: str = ""
    # Last fully printed settlement average for the current contract. A lock-in
    # position is settled at 1/0 from this, not marked at the next window's bid.
    lock_in_settlement: tuple[str, float, float] | None = None
    # Per-window budget for the promoted flow program: at most `max_window_trades`
    # entries in one 15-minute contract, and no re-entry once it has taken a loss.
    window_entries: int = 0
    window_halted: bool = False
    # The bankroll this window started with. The drawdown cap is measured from it, so
    # a bad window stops the lane for *that window* and cannot strand it forever.
    window_start_bankroll: Decimal = Decimal("0")


class Engine:
    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store
        self.client = KalshiClient(settings)
        self.feed = SpotFeed(
            symbols=sorted({s.feed_symbol for s in settings.slots if s.feed_symbol}),
            on_price=self._on_price,
        )
        spots = {s.feed_symbol for s in settings.slots if s.feed_symbol}
        self.tape = features.SpotTape()
        self._spot_symbols = spots

        slot_cfg = [s for s in settings.slots if s.has_spot]
        share = (settings.paper_equity / Decimal(len(slot_cfg))).quantize(Decimal("0.01"))
        self.slots: list[SlotRuntime] = [
            SlotRuntime(cfg=c, params=Params(), bankroll=share, peak_bankroll=share) for c in slot_cfg
        ]
        # The ledger survives a restart; a bot that forgets its own P&L is lying. But a
        # *clean epoch* starts from its own base: carrying the previous strategy's
        # paper losses into the new epoch double-counts a bankroll the venue never lost.
        epoch_since = float(store.get_meta("epoch_start") or 0.0)
        restored = Decimal(str(
            store.epoch_stats(epoch_since)["pnl"] if epoch_since else store.stats()["pnl"]
        )).quantize(Decimal("0.01"))
        self._epoch_since = epoch_since
        self.cash = settings.paper_equity + restored
        self.start_equity = settings.paper_equity
        self.day_start_equity = self.cash
        self.realized = restored
        self.equity = self.cash
        self.peak_equity = self.cash
        for slot in self.slots:
            lane_pnl = (store.epoch_stats(epoch_since, slot.cfg.series)["pnl"] if epoch_since
                        else store.stats(slot.cfg.series)["pnl"])
            slot.bankroll += Decimal(str(lane_pnl)).quantize(Decimal("0.01"))
            slot.peak_bankroll = max(slot.peak_bankroll, slot.bankroll)
        self.supervisor = Supervisor(store=store)
        self.gate = LayaGate(os.environ.get("OPENKAL_GATE_URL", "http://127.0.0.1:8000"))
        self.executor = Executor(self)
        # The ledger must say what actually happened. This is paper until `_maybe_arm`
        # flips it, and every trade written before that is a paper trade.
        self.source = "paper"
        self.started = time.time()
        self.last_snapshot = time.time()
        self.equity_curve: list[dict[str, float]] = []
        self.fills: list[dict[str, Any]] = []
        self.market_errors = 0
        self.error_times: list[float] = []
        self.fault_times: list[float] = []
        self.exchange_active = True
        # Venue truth is read on every window roll, not once at boot: a bot that
        # quotes a balance it read an hour ago is sizing and reporting a different
        # account than the one the orders actually settle against.
        self._last_window_sync = 0.0
        self._venue_flat = False
        # Live state. Paper until readiness passes; the snapshot never claims otherwise.
        self.venue_balance = Decimal("0")
        self.live_armed = False
        self.arm_ts: float | None = None
        self.kill_reason: str | None = None
        self.epoch_start = time.time()
        self.broker = None
        self.live = None
        self._repro_cache: dict[str, Any] | None = None
        self._last_arm_check = 0.0
        self._stop = asyncio.Event()
        self._last_equity_store = 0.0
        # Deterministic per-lane calibration, loaded from the validated artifact. The
        # file currently holds identity: every fitted map lost to identity on the
        # frozen holdout, and a fitted map that only wins in-sample is a regression.
        self.calibrators = features.load_calibrators(ROOT / "data" / "calibration.json")
        # F2's probability is a different quantity on a different feature space: the
        # chance the trade's own net is positive. It has its own frozen calibrator,
        # fitted in-sample by scripts/f2_structural.py, and it is never passed
        # through the settlement calibrator above.
        self.payoff_calibrator = features.load_payoff_calibrator(
            ROOT / "data" / "f2-structural.json")
        self.ws: KalshiWS | None = None
        self.index_tape: dict[str, list[tuple[float, float]]] = {}
        self.index_prints: dict[str, Any] = {}
        self.book_hist: dict[str, list[tuple[float, Decimal]]] = {}
        # Venue print tape and touch history: the promoted flow program's two
        # microstructure inputs. Both are receive-time ordered and bounded.
        self.trade_hist: dict[str, list[tuple[float, float, int]]] = {}
        self.touch_hist: dict[str, list[tuple[float, Decimal, Decimal]]] = {}
        self.lag_ms: float = 0.0

    # ---------- feed callbacks ----------
    def _on_price(self, symbol: str, price: float, source: str) -> None:
        now = time.time()
        self.tape.add(symbol, now, price)
        for slot in self.slots:
            if slot.cfg.feed_symbol == symbol:
                slot.spot = price

    def _on_ws_event(self, msg: dict[str, Any]) -> None:
        """Venue print tape, per contract, in receive order (the flow term reads it)."""
        if msg.get("type") != "trade":
            return
        body = msg.get("msg") or {}
        ticker = body.get("market_ticker") or ""
        if not ticker:
            return
        try:
            size = float(body.get("count_fp") or 0.0)
        except (TypeError, ValueError):
            return
        side = 1 if str(body.get("taker_side") or "").lower() == "yes" else 0
        hist = self.trade_hist.setdefault(ticker, [])
        hist.append((time.time(), size, side))
        if len(hist) > 600:
            del hist[:200]

    def _on_book(self, book) -> None:
        if book.yes_bid is None or book.yes_ask is None:
            return
        now = time.time()
        touch = self.touch_hist.setdefault(book.ticker, [])
        touch.append((now, book.yes_bid, book.yes_ask))
        if len(touch) > 800:
            del touch[:300]
        hist = self.book_hist.setdefault(book.ticker, [])
        # One sample per 250ms, not per message. A hot 15-minute market prints ~1000
        # book messages a second, so a 600-row history was less than a second long and
        # a 5s impulse could never find its anchor. 2400 points is 10 minutes of tape.
        if not hist or now - hist[-1][0] >= 0.25:
            hist.append((now, book.yes_bid))
            if len(hist) > 2400:
                del hist[:600]
        self.lag_ms = self.ws.lag_ms if self.ws else 0.0

    def _on_index(self, print_) -> None:
        tape = self.index_tape.setdefault(print_.index_id, [])
        tape.append((print_.received_ts, float(print_.value)))
        if len(tape) > 3000:
            del tape[:1000]
        self.index_prints[print_.index_id] = print_

    def _book_move_c(self, ticker: str, lookback_s: float) -> float:
        hist = self.book_hist.get(ticker) or []
        if len(hist) < 2:
            return 0.0
        ts_now, bid_now = hist[-1]
        for ts, bid in reversed(hist):
            if ts_now - ts >= lookback_s:
                return float(bid_now - bid) * 100.0
        return 0.0

    def _micro_signal(self, slot: SlotRuntime, quote: Quote, seconds_left: float,
                      sigma: float):
        """Pick the best available microstructure thesis for this tick, or None."""
        if self.ws is None or self.ws.degraded:
            return None
        index_id = INDEX_BY_SERIES.get(slot.cfg.series)
        prints = self.index_tape.get(index_id) or []
        params = slot.params
        # The settlement program is a different thesis on a different tape, not a
        # variation on the momentum one: it is selected by configuration and it
        # reads the settlement index, the resting size and the aggressor tape.
        if params.strategy_mode == "student_t_l2_settle":
            return self._settlement_signal(slot, quote, seconds_left)
        # 0. The promoted program: the strike z and the venue's own prints agree, and
        #    the touch has been re-pricing the same way. This is the only thesis with
        #    a passing frozen-OOS verdict, so it is checked first.
        now = time.time()
        pts = self.tape.points(slot.cfg.feed_symbol or "")
        flow = features.print_flow(self.trade_hist.get(quote.ticker) or [], now,
                                   params.zobi_flow_window_s)
        revision = features.touch_revisions(self.touch_hist.get(quote.ticker) or [],
                                            now, params.zobi_revision_window_s)
        # F2 is checked first. Its gates are a strict superset of the flow program's
        # (z >= 0.3 against 0.6, flow >= 0.1 against 0.15, revisions >= 0.0 against
        # 0.10), so anything the old detector fires on, this one fires on too --
        # and this is the program whose probability is calibrated to its own trade
        # payoff rather than to the market's settlement. Leaving the old detector
        # in front of it would mean the deployed signal is never the frozen one.
        f2_sig = features.f2_taker_1s(
            spot=slot.spot, strike=quote.strike, seconds_left=seconds_left, sigma=sigma,
            flow=flow, revision=revision,
            calibrator=self.payoff_calibrator,
            min_z=params.f2_min_z, min_flow=params.f2_min_flow,
            min_revision=params.f2_min_revision,
            min_delta_bps=params.f2_min_delta_bps)
        if f2_sig is not None:
            return f2_sig
        flow_sig = features.z_obi_flow(
            spot=slot.spot, strike=quote.strike, seconds_left=seconds_left, sigma=sigma,
            flow=flow, revision=revision,
            min_z=params.zobi_min_z, min_flow=params.zobi_min_flow,
            min_revision=params.zobi_min_revision,
            min_delta=float(params.zobi_min_delta),
            trend_fast=(features.momentum(pts, params.zobi_trend_fast_s)
                        if params.zobi_trend_gate else 0.0),
            trend_slow=(features.momentum(pts, params.zobi_trend_slow_s)
                        if params.zobi_trend_gate else 0.0))
        if flow_sig is not None:
            return flow_sig
        # 1. Asian lock-in: inside the final printed minute the settlement average is
        #    mostly arithmetic, and the venue's book often still sits near 50c.
        if seconds_left <= params.final_minute_s and prints:
            latest = self.index_prints.get(index_id)
            if latest is not None and latest.final_minute is not None:
                # `last_60s_windowed_average_15min` keeps its k=60 value between
                # windows. Reading it at T-59 as a fully locked average was the bug
                # that made the model claim 1.000 against a 0.10 book and lose.
                elapsed = max(0.0, params.final_minute_s - seconds_left)
                if latest.final_minute_ticks <= elapsed + 3:
                    if (latest.final_minute_ticks >= params.final_minute_min_ticks
                            and sigma > 0):
                        sig = features.final_minute(
                            spot=float(latest.value), strike=quote.strike,
                            locked_avg=float(latest.final_minute),
                            locked_ticks=latest.final_minute_ticks,
                            sigma_per_sqrt_s=sigma, seconds_left=seconds_left)
                        if sig is not None:
                            return sig
        # 2. Spot-impulse snipe: the settlement index moved, the ladder has not.
        return features.impulse_snipe(
            prints=prints, spot=slot.spot, strike=quote.strike,
            seconds_left=seconds_left, sigma=sigma,
            book_hist=self.book_hist.get(quote.ticker) or [],
            min_seconds_left=params.min_seconds_left,
            lookback_s=params.impulse_lookback_s,
            min_bps=params.impulse_min_bps, max_bps=params.impulse_max_bps,
            stale_share=params.impulse_stale_share)

    def _settlement_signal(self, slot: SlotRuntime, quote: Quote, seconds_left: float):
        """The settlement program: Student-t fair value, true-L2 imbalance, signed flow.

        The fair value is priced off the *settlement index* -- the tape the contract
        actually settles on -- and not off the Coinbase spot, because the frozen
        Student-t fit is of the index's own standardised returns. The imbalance is
        the resting size on the two sides of the touch, and the flow is the signed
        aggressor print tape on a fast and a slow window.
        """
        fit = features.STUDENT_T_FIT.get(slot.cfg.series)
        if fit is None:
            return None
        index_id = INDEX_BY_SERIES.get(slot.cfg.series)
        pts = self.index_tape.get(index_id) or []
        now = time.time()
        spot = _last_point(pts, now, slot.params.max_spot_age_s)
        if spot is None:
            return None
        index_sigma = features.realized_sigma(pts)
        if index_sigma <= 0:
            return None
        obi = features.l2_imbalance(quote.yes_bid_sz, quote.yes_ask_sz)
        if obi is None:
            return None
        prints = self.trade_hist.get(quote.ticker) or []
        return features.student_t_l2_settle(
            spot=spot, strike=quote.strike, seconds_left=seconds_left, sigma=index_sigma,
            nu=fit[0], s=fit[1], l2_obi=obi,
            flow_fast=features.print_flow(prints, now, slot.params.sl2_flow_fast_s),
            flow_slow=features.print_flow(prints, now, slot.params.sl2_flow_slow_s),
            min_seconds_left=slot.params.sl2_min_seconds_left,
            max_seconds_left=slot.params.sl2_max_seconds_left)

    # ---------- lifecycle ----------
    async def run(self) -> None:
        await self.feed.start()
        # The authenticated socket is the primary market data plane: index prints the
        # contracts settle on, and the real ladder instead of a 3-second poll.
        self.ws = KalshiWS(self.settings, self.client, store=self.store,
                           on_book=self._on_book, on_index=self._on_index,
                           on_event=self._on_ws_event)
        await self.ws.start()
        try:
            await self._sync_account()
        except Exception as exc:  # noqa: BLE001 - a balance read must not stop the soak
            self.market_errors += 1
            self.supervisor.note("SYNC_FAILED", f"{type(exc).__name__}: {exc}")
        # Venue truth before the first decision: adopt or flatten whatever the
        # exchange says we hold, so no lane starts blind to an orphaned position.
        try:
            await self._reconcile_venue("boot")
        except Exception as exc:  # noqa: BLE001 - a reconcile failure is reported, not fatal
            self.supervisor.note("RECONCILE_FAILED", f"boot: {type(exc).__name__}: {exc}")
        # Size the lanes off the account the venue actually holds, not the epoch's
        # opening number. Without this a restart divides the old $201.10 that no longer
        # exists, and a lane the drawdown cap stopped stays stranded on its old slice
        # until the next 15-minute roll.
        try:
            await self._rebalance_window(time.time(), force=True)
        except Exception as exc:  # noqa: BLE001 - sizing off a stale read is worse, but not fatal
            self.supervisor.note("SYNC_FAILED", f"boot rebalance: {type(exc).__name__}: {exc}")
        self._seen_reconnects = self.ws.reconnects if self.ws else 0
        self.store.set_meta("started", self.started)
        # Record the program this process is actually running: the replay controls
        # re-run *that* program against the ledger, not whatever today's defaults are.
        self._record_program()
        try:
            while not self._stop.is_set():
                started = time.perf_counter()
                try:
                    await self.step()
                except Exception as exc:  # noqa: BLE001 - the loop must survive a bad payload
                    self.market_errors += 1
                    self.fault_times = (self.fault_times + [time.time()])[-100:]
                    self.supervisor.note("HALT_FLATTEN", f"engine error {type(exc).__name__}: {exc}")
                    await asyncio.sleep(1.0)
                spent = time.perf_counter() - started
                # Sub-second cadence: the ladder moves several ticks between 1 Hz
                # samples, and a fill that happened on one of them must exist as a
                # recorded tick. Five passes a second is well inside the REST budget.
                await asyncio.sleep(max(0.05, 0.20 - spent))
        finally:
            await self.feed.stop()
            await self.client.close()

    def stop(self) -> None:
        self._stop.set()

    async def _ensure_ws(self, now: float) -> None:
        """Restart the market-data socket if its receive loop ever exits.

        The plane is supposed to outlive every other component: paper or live, the
        settlement index and the ladder both come from this socket. A receive task
        that dies (or a handshake that raises before the first connect) previously
        left the engine polling REST for the rest of the run with no signal beyond a
        silent `0 messages` counter.
        """
        if self.ws is None or now - getattr(self, "_last_ws_check", 0.0) < 5.0:
            return
        self._last_ws_check = now
        task = getattr(self.ws, "task", None)
        if task is not None and not task.done():
            return
        detail = self.ws.last_error or "socket task is not running"
        if task is not None:
            try:
                exc = task.exception()
            except Exception:  # noqa: BLE001 - cancelled or never started
                exc = None
            if exc is not None:
                detail = f"{type(exc).__name__}: {exc}"
        self.supervisor.note("WS_RESTART", detail)
        await self.ws.start()

    def _record_program(self) -> None:
        """Persist the running program *and when it started running*.

        One flat `deployed_params` key made a restart rewrite the program for every
        tick already in the epoch, so the parity control replayed yesterday's trades
        under today's params and called the ledger a liar. The history is a timeline:
        `replay_series` switches params at each boundary, which is exactly what the
        live loop did (it reads `slot.params` at decision time).
        """
        params = {s.cfg.series: s.params.to_json() for s in self.slots}
        history = self.store.get_meta("deployed_params_history") or []
        if not isinstance(history, list):
            history = []
        if not history:
            previous = self.store.get_meta("deployed_params")
            if isinstance(previous, dict) and previous:
                # Seed the timeline with the program the epoch actually opened under.
                history = [{"ts": float(self.store.get_meta("epoch_start") or self.started),
                            "params": previous}]
        if not history or history[-1].get("params") != params:
            history.append({"ts": time.time(), "params": params})
            self.store.set_meta("deployed_params_history", history[-20:])
            self.supervisor.note("PROGRAM", "recorded the params this process trades")
        self.store.set_meta("deployed_params", params)

    async def _sync_account(self) -> None:
        """Read the venue balance. Paper mirrors it so sizing is the live sizing."""
        from .broker import Broker

        broker = Broker(self.client)
        self.broker = broker
        balance = await broker.balance()
        if balance > 0:
            self.venue_balance = balance
        if self.settings.paper_equity <= 0 and balance > 0:
            # The epoch's starting equity is part of the ledger. Persist it so a restart
            # continues the same epoch instead of silently resetting to zero.
            epoch_equity = self.store.get_meta("epoch_equity")
            if epoch_equity is None:
                epoch_equity = str(balance)
                self.store.set_meta("epoch_equity", epoch_equity)
            base = Decimal(str(epoch_equity))
            epoch_start = self.store.get_meta("epoch_start")
            since = float(epoch_start) if epoch_start else 0.0
            restored = Decimal(str(
                self.store.epoch_stats(since).get("pnl") if since
                else self.store.stats().get("pnl") or "0"))
            self.start_equity = base
            self.cash = base + restored
            self.equity = self.cash
            self.day_start_equity = self.cash
            self.peak_equity = max(self.peak_equity, self.cash)
            share = (base / Decimal(len(self.slots))).quantize(Decimal("0.0001"))
            for slot in self.slots:
                lane_pnl = (self.store.epoch_stats(since, slot.cfg.series)["pnl"] if since
                            else self.store.stats(slot.cfg.series)["pnl"])
                slot.bankroll = share + Decimal(str(lane_pnl))
                slot.peak_bankroll = max(slot.peak_bankroll, slot.bankroll)
            if epoch_start:
                self.epoch_start = float(epoch_start)
            else:
                self.store.set_meta("epoch_start", self.epoch_start)
            self.supervisor.note("SYNC", f"epoch {base} USD, live balance {balance} USD")

    # ---------- venue reconciliation ----------
    def _slot_for_ticker(self, ticker: str) -> SlotRuntime | None:
        for slot in self.slots:
            if ticker.startswith(slot.cfg.series):
                return slot
        return None

    async def _reconcile_venue(self, reason: str) -> None:
        """Adopt or flatten whatever the venue says we hold, before trading anything.

        The engine's memory is not the venue's truth: a crash, a restart or a lost
        socket can leave contracts the process does not know about. Holding one of
        those blind into expiry is exactly the failure this closes. On boot and on
        every socket reconnect the venue's positions and resting orders are read and
        reconciled: adopted while already armed, otherwise flattened (when this run
        is allowed to send orders) or halted loudly under a paper lock.
        """
        if self.broker is None:
            return
        try:
            rows = await self.broker.positions()
            resting = await self.broker.open_orders()
        except Exception as exc:  # noqa: BLE001 - a read failure must not stop boot
            self.supervisor.note("RECONCILE_FAILED", f"{reason}: {type(exc).__name__}: {exc}")
            return
        held: list[tuple[str, int, dict[str, Any]]] = []
        for pos in rows:
            qty = int(float(pos.get("position_fp") or pos.get("position") or 0))
            ticker = pos.get("ticker") or ""
            if ticker and qty:
                held.append((ticker, qty, pos))
        self._venue_flat = not held and not resting
        if not held and not resting:
            if reason == "boot":
                self.supervisor.note("RECONCILE", "venue flat: 0 positions, 0 resting orders")
            return
        self.supervisor.note(
            "RECONCILE", f"{reason}: {len(held)} venue position(s), {len(resting)} resting order(s)")
        if not self.settings.live:
            # A paper-locked run must not send a single venue order, so it cannot
            # flatten. It also must never paper-trade a contract the venue holds:
            # halt the affected lanes and say so instead of pretending to be flat.
            for ticker, qty, _row in held:
                slot = self._slot_for_ticker(ticker)
                if slot is not None:
                    slot.halted = True
                    slot.reason = f"venue holds {qty} {ticker}; paper lock cannot flatten"
                self.supervisor.note("ORPHAN_POSITION", f"{ticker}: {qty} contracts, paper locked")
            return
        if self.live_armed:
            for ticker, qty, row in held:
                slot = self._slot_for_ticker(ticker)
                if slot is not None:
                    self._adopt(slot, ticker, qty, row)
            # Orders for adopted contracts are pulled: the exit machine will place
            # its own from the adopted price rather than trust a stale one.
            for oid, order in resting.items():
                await self.broker.cancel(oid, order.get("ticker"))
            return
        # Not armed yet: there is no exit machine to adopt into, and a position this
        # process did not open has no entry fee, target or signal. Flatten it.
        self.supervisor.note(
            "RECONCILE_FLAT", "flattening orphan: "
            + ", ".join(f"{ticker} {qty}" for ticker, qty, _ in held))
        await self.broker.flatten_all()

    def _adopt(self, slot: SlotRuntime, ticker: str, qty: int, row: dict[str, Any]) -> None:
        """Re-attach a venue position this process lost track of."""
        side = "YES" if qty > 0 else "NO"
        contracts = abs(qty)
        exposure = abs(Decimal(str(row.get("market_exposure_dollars") or "0")))
        fallback = slot.quote.mid if slot.quote is not None else Decimal("0.50")
        price = (exposure / contracts) if exposure > 0 and contracts else fallback
        price = max(Decimal("0.01"), min(Decimal("0.99"), price))
        slot.position = Position(
            ticker=ticker, side=side, entry=price, contracts=contracts,
            opened_ts=time.time(), p_entry=slot.p_up, stake=price * contracts,
            tp=min(Decimal("0.99"), price + slot.params.take_profit),
            sl=max(Decimal("0.01"), price - slot.params.stop_loss),
            signal="reconciled",
        )
        self.supervisor.note("ADOPT", f"{ticker}: {contracts} {side} at {price}")

    async def _maybe_arm(self, now: float) -> None:
        """The only place `live_armed` can become true."""
        if self.live_armed or not self.settings.arm_live or not self.settings.live:
            return
        # 5s, not 30s, until armed. The first check runs before any quote exists (the
        # `feeds` row needs one), and a 30s retry left a whole window for a paper fill
        # to land first, which then defers arming until it settles. `evaluate` is cached
        # by its own key, so the extra calls are cheap.
        if now - self._last_arm_check < 5:
            return
        self._last_arm_check = now
        from .readiness import evaluate

        verdict = evaluate(self)
        if verdict["ready"]:
            # A paper fill placed before the gate opened has no venue counterpart.
            # Arming on top of one makes the engine hold, mark and try to exit a position
            # the exchange has never heard of (observed live: 79 phantom SOL contracts,
            # a live badge and an unmoved balance). Resting paper orders are pulled; an
            # open paper *position* defers arming until its window settles. Closing it
            # at the mark wrote a trade no tape replay can reproduce, and the parity
            # control caught exactly that (`paper book cleared to arm`, replay +$12.86
            # vs ledger +$8.17).
            self._clear_paper_orders()
            open_paper = [s.cfg.series for s in self.slots if s.position is not None]
            if open_paper:
                self.supervisor.note(
                    "ARM_WAIT", "paper position open on " + ", ".join(open_paper)
                    + "; arming once the window settles")
                return
            self.live_armed = True
            self.arm_ts = now
            self.source = "live"
            # Persisted evidence that this epoch ran against the venue: the replay
            # controls use it to know the exchange, not a queue model, decided fills.
            self.store.set_meta("armed_at", now)
            from .live_exec import LiveExecutor

            self.live = LiveExecutor(self, self.broker)
            self.supervisor.note("ARM_LIVE", "readiness passed: venue orders enabled")
        elif verdict["blockers"]:
            self.supervisor.note("READY_WAIT", "; ".join(verdict["blockers"][:3]))

    def _clear_paper_orders(self) -> None:
        """Pull resting paper bids before the venue takes over. No fills, no trades."""
        for slot in self.slots:
            if slot.order is None:
                continue
            self.cash += slot.order.reserved
            if slot.quote is not None:
                self.executor.record(slot, "cancel", slot.order.side, slot.order.price,
                                     slot.order.contracts, Decimal("0"),
                                     "paper book cleared to arm")
            slot.order = None

    # ---------- one pass ----------
    async def step(self) -> None:
        now = time.time()
        # A reconnect is a fresh view of the venue. Re-read it before sizing or
        # exiting anything, because a dropped socket is exactly when a fill can go
        # unseen and a position can be orphaned.
        if self.ws is not None and self.ws.reconnects != getattr(self, "_seen_reconnects", 0):
            self._seen_reconnects = self.ws.reconnects
            await self._reconcile_venue("reconnect")
        stale_market = now - getattr(self, "_last_market_refresh", 0.0) > self.settings.market_poll_s
        await self._ensure_ws(now)
        # Arm before this pass places anything. Checking at the end of the pass let a
        # paper entry slip in first, and arming then has to wait a whole window for it
        # to settle (or write a trade off the venue's tape).
        await self._maybe_arm(now)
        venue = await self._venue_snapshot() if self.live_armed else None
        for slot in self.slots:
            try:
                if stale_market:
                    await self._refresh_market(slot)
                await self._step_slot(slot, now, venue)
            except Exception as exc:  # noqa: BLE001 - one bad lane must not stall the rest
                self.fault_times = (self.fault_times + [now])[-100:]
                self.supervisor.note("LANE_FAULT", f"{slot.cfg.series}: {type(exc).__name__}: {exc}")
                # `_step_slot` already fail-safes its own body; this covers the
                # market refresh beside it, which runs in the same try but outside
                # that wrapper. Either way, a live position is never left blind.
                if (self.live_armed and self.live is not None
                        and (slot.position is not None or slot.order is not None)):
                    try:
                        await self._fail_safe_slot(slot, exc, now)
                    except Exception as nested:  # noqa: BLE001
                        slot.halted = True
                        slot.reason = "fail-safe could not flatten: position open"
                        self.supervisor.note(
                            "FAIL_SAFE_FAILED", f"{slot.cfg.series}: {type(nested).__name__}: {nested}")
            if slot.halted:
                self.supervisor.run(slot, slot.bankroll, now)
        if stale_market:
            self._last_market_refresh = now
        self._mark()
        if now - self._last_equity_store > 10:
            self.store.equity(now, str(self.equity), str(self.cash), str(self.realized))
            self.equity_curve.append({"ts": now, "equity": float(self.equity)})
            self.equity_curve = self.equity_curve[-720:]
            self._last_equity_store = now
        self.last_snapshot = now

    async def _refresh_market(self, slot: SlotRuntime) -> None:
        status, payload = await self.client.get(
            "/markets",
            {"series_ticker": slot.cfg.series, "status": "open", "limit": 10},
        )
        if status != 200:
            self.market_errors += 1
            self.error_times = (self.error_times + [time.time()])[-300:]
            slot.reason = "venue unreachable"
            return
        now = time.time()
        # Only true 15-minute windows: a longer-dated contract on the same series is a
        # different instrument and would poison the whole model.
        markets = [
            m for m in payload.get("markets", [])
            # Keep the contract alive to the last seconds: the lock-in model trades the
            # final printed minute, so a 20-second buffer blinded the lane exactly when
            # its edge existed. Two seconds is enough to not quote through settlement.
            if now + 2 < parse_ts(m.get("close_time")) <= now + 1200
            and 480 <= parse_ts(m.get("close_time")) - parse_ts(m.get("open_time")) <= 1100
        ]
        if not markets:
            # `status=open` lags the quarter-hour rollover by up to half a minute; the
            # next window is already listed as `unopened` and that is where quotes will go.
            status, payload = await self.client.get(
                "/markets",
                {"series_ticker": slot.cfg.series, "status": "unopened", "limit": 10},
            )
            if status == 200:
                markets = [
                    m for m in payload.get("markets", [])
                    # Keep the contract alive to the last seconds: the lock-in model trades the
            # final printed minute, so a 20-second buffer blinded the lane exactly when
            # its edge existed. Two seconds is enough to not quote through settlement.
            if now + 2 < parse_ts(m.get("close_time")) <= now + 1200
                    and 480 <= parse_ts(m.get("close_time")) - parse_ts(m.get("open_time")) <= 1100
                ]
        if not markets:
            slot.market = None
            # Do not drop the quote out from under an open trade: nulling it erased the
            # ticker transition the roll keyed on, and the position was then stranded
            # across the boundary with no way to settle or mark it.
            if slot.position is None and slot.order is None:
                slot.quote = None
            slot.state = "no market"
            slot.reason = "no open 15-minute contract"
            return
        market = min(markets, key=lambda m: parse_ts(m.get("close_time")))
        if self.ws is not None and market.get("ticker"):
            self.ws.watch([market["ticker"]])
        previous = slot.quote
        if previous and previous.ticker != market.get("ticker"):
            await self._roll_slot(slot, slot.quotes.get(previous.ticker, previous), now)
        slot.market = market
        slot.quote = self._parse_quote(slot, market)
        slot.quotes[slot.quote.ticker] = slot.quote
        slot.quotes = dict(list(slot.quotes.items())[-6:])
        slot.window_open_ts = parse_ts(market.get("open_time"))

    @staticmethod
    def _parse_quote(slot: SlotRuntime, market: dict[str, Any]) -> Quote:
        now = time.time()
        return Quote(
            ticker=market.get("ticker", ""),
            series=slot.cfg.series,
            strike=float(market.get("floor_strike") or 0.0),
            close_ts=parse_ts(market.get("close_time")),
            yes_bid=money(market.get("yes_bid_dollars")),
            yes_ask=money(market.get("yes_ask_dollars")),
            last=money(market.get("last_price_dollars")),
            # Freshness is our fetch time. The venue's `updated_time` often stays at the
            # window open for minutes while the book moves under it, so using it as a
            # staleness clock abstains on every tradeable window.
            updated_ts=now,
            received_ts=now,
        )

    async def _roll_slot(self, slot: SlotRuntime, old: Quote, now: float) -> None:
        """Window rolled: pull any resting order and settle or mark the survivor."""
        if slot.order:
            self.cash += slot.order.reserved
            slot.quote = slot.quote or old
            self.executor.record(slot, "cancel", slot.order.side, slot.order.price,
                                 slot.order.contracts, Decimal("0"), "window rolled")
            slot.order = None
        if slot.position:
            pos = slot.position
            settlement = slot.lock_in_settlement
            settle_ok = settlement is not None and settlement[0] == pos.ticker
            # A binary carried through expiry settles at 1/0 on the venue, so the
            # ledger must book 1/0 too. Booking the last bid instead (0.99 at best:
            # the venue caps prices at 99c) leaves a real, cent-level gap against
            # `/portfolio/fills`. `z_obi_flow` carries by design when its scratch
            # misses, so it settles like the lock-in thesis.
            if settle_ok and (pos.hold_to_settle or pos.signal == "z_obi_flow"
                              or pos.signal == "student_t_l2_settle"
                              or (pos.signal == "final_minute"
                                  and slot.params.settle_lock_in)):
                # The contract settles on the 60-second BRTI average. We have its last
                # fully printed value, so the outcome is arithmetic, not a guess. The
                # settlement is written as a tick so the replay closes at the same 1/0.
                value, strike = settlement[1], settlement[2]
                won = (value >= strike) if pos.side == "YES" else (value < strike)
                price = Decimal("1") if won else Decimal("0")
                # Timestamp it at the roll, after the last live quote, so the replay
                # processes it last and rolls on the settled 1/0 rather than the stale bid.
                self.store.tick(
                    ts=now, series=slot.cfg.series, ticker=old.ticker,
                    yes_bid=str(price), yes_ask=str(price), last=str(price),
                    spot=slot.spot, p_model=slot.p_up, edge=0.0, secs_left=0.0,
                    spread=0.0, signal="settle", can_open=False, stake=str(slot.bankroll),
                    spot_age=0.0, hurdle=str(self._edge_hurdle(slot)), impulse_bps=0.0,
                )
                slot.quote = replace(old, yes_bid=price, yes_ask=price, last=price,
                                     received_ts=now)
                self.executor.close(slot, price, "settled", maker=True)
            else:
                self.executor.close(slot, pos.side_bid(old), "window rolled", maker=False)
        slot.lock_in_settlement = None
        slot.exit = None
        slot.consecutive_losses = 0
        slot.window_entries = 0
        slot.window_halted = False
        # `pairs` is the calibration sample, not per-window state. Wiping it every roll
        # made an 8-pair lane statistically impossible and the gate row permanently red.
        await self._rebalance_window(now)

    async def _rebalance_window(self, now: float, force: bool = False) -> None:
        """A new 15-minute window is a new budget, sized off the live account.

        Three defects met here. `venue_balance` was read once at boot, so `/api/state`
        quoted $201.10 while the venue held $151.22. A lane the 12% slot-drawdown cap
        stopped (KXBTC after the -$49.88 settlement) never traded again, stranding its
        $0.40 slice permanently. And every slot kept the bankroll it was sliced at, so
        all four divided the *old* account. The window roll is the one moment a lane
        is flat by construction: re-read the venue, re-slice `equity / 4`, and re-arm
        what the drawdown cap stopped, so the cap is per window instead of forever.
        """
        if force or now - self._last_window_sync >= 20.0:
            self._last_window_sync = now
            if self.broker is not None:
                try:
                    # Venue truth before sizing: this also refreshes `_venue_flat`, and
                    # adopts anything the exchange holds that this process forgot.
                    await self._reconcile_venue("window roll")
                    balance = await self.broker.balance()
                    if balance > 0:
                        self.venue_balance = balance
                except Exception as exc:  # noqa: BLE001 - a read failure must not stop the roll
                    self.supervisor.note("SYNC_FAILED",
                                         f"window roll: {type(exc).__name__}: {exc}")
        if self.venue_balance <= 0:
            return
        flat = all(s.position is None and s.order is None for s in self.slots)
        if flat and self._venue_flat:
            # While nothing is held the venue's balance IS cash. Adopting it here is
            # what keeps `/api/state` and `/portfolio/balance` on the same number
            # instead of accumulating cents of model drift forever.
            self.cash = self.venue_balance
            self.equity = self.venue_balance
            self.peak_equity = max(self.peak_equity, self.equity)
        share = (self.venue_balance / Decimal(len(self.slots))).quantize(Decimal("0.0001"))
        reset: list[str] = []
        for slot in self.slots:
            if slot.position is not None or slot.order is not None:
                continue          # a live position keeps its own bankroll for this window
            if slot.halted:
                reset.append(slot.cfg.series)
            slot.bankroll = share
            # Peak equals the window's opening bankroll, so the supervisor's 12% check
            # is a *per-window* loss cap. The old peak was the epoch's, which is how one
            # -$49.88 settlement halted a $37.81 lane for the rest of the epoch.
            slot.peak_bankroll = share
            slot.window_start_bankroll = share
            slot.halted = False
            slot.flatten_requested = False
            slot.consecutive_losses = 0
        if reset:
            self.supervisor.note("WINDOW_RESET", "re-armed " + ", ".join(reset)
                                 + f" at {share} USD per lane")
            if not any(s.halted for s in self.slots):
                self.supervisor.fault = None

    # ---------- per-slot decision ----------
    def _gate_recorder(self, slot: SlotRuntime, ticker: str) -> Any:
        """The callback that files Laya's opinion when the background review lands.

        The verdict is persisted on the trigger it judged: an offline ablation needs
        the opinion the live lane actually saw, not a re-run with hindsight. It is
        pinned to the contract the review was about, so a review still in flight
        when the window rolls cannot be filed against the next contract.
        """
        def record(verdict: Any) -> None:
            slot.gate = verdict_payload(verdict)
            slot.gate_ts = time.time()
            try:
                self.store.set_tick_gate(series=slot.cfg.series, gate=json.dumps(slot.gate),
                                          ticker=ticker)
            except Exception:  # noqa: BLE001 - a logging failure is not a trading event
                pass

        return record

    def _gate_refresh(self, slot: SlotRuntime, quote: Quote, seconds_left: float,
                      decision: Decision) -> None:
        """Ask Laya for a fresh opinion without waiting for it.

        The review runs in a background task and its verdict is cached per ticker.
        The entry path below reads that cache, so a 610-650ms forward pass never
        stalls the tick loop. The cache is refreshed on a timer (15s) or as soon as
        the book's mid has moved a tick: the question Laya was asked describes a
        book, and a book that has moved is a different question.
        """
        if not self.gate.stale(quote.ticker, mid=float(quote.mid)):
            return
        self.gate.schedule(slot, quote, seconds_left, decision,
                           on_done=self._gate_recorder(slot, quote.ticker))

    def _gate_veto(self, slot: SlotRuntime, quote: Quote, decision: Decision) -> Decision:
        """Apply the cached Laya verdict to an entry. A cache read: 0ms.

        The gate can only veto: the kernel's numeric abstention still stands when Laya
        votes enter. It never creates a trade the model did not ask for, and a gate
        that is offline never blocks -- the kernel's numeric edge decides.

        `Params.require_layagate_allow` turns the cached opinion into a hard filter
        for a promoted strategy, with no code edit. The fine-tuned ONNX build has
        always had veto power; the switch exists so a strategy that was promoted
        against the gate's own forward evidence can enforce it too.
        """
        params = slot.params
        if not (params.require_layagate_allow or self.gate.mode == "onnx"):
            return decision
        verdict = self.gate.cached(quote.ticker)
        if verdict is None or not verdict.available:
            return decision
        if verdict.allow and verdict.confidence >= params.layagate_min_confidence:
            return decision
        return Decision("abstain", f"laya gate: {verdict.note or 'skip'}",
                        side=decision.side, price=decision.price, p_model=decision.p_model,
                        confidence=decision.confidence, edge=decision.edge, regime="gate")

    async def _venue_snapshot(self) -> dict[str, Any]:
        """One read of the venue's resting book and fills tape, shared by every lane."""
        cache = getattr(self, "_venue_cache", None)
        now = time.time()
        if cache and now - cache["ts"] < 1.2:
            return cache["value"]
        value = {"orders": {}, "fills": []}
        if self.broker is not None:
            try:
                value = {"orders": await self.broker.open_orders(),
                         "fills": await self.broker.fills(limit=200)}
            except Exception as exc:  # noqa: BLE001 - a read failure must not stop the loop
                self.market_errors += 1
                self.supervisor.note("VENUE_READ", f"{type(exc).__name__}: {exc}")
        self._venue_cache = {"ts": now, "value": value}
        return value

    async def kill(self, reason: str) -> None:
        """Kill switch: stop opening, cancel resting orders, flatten every position."""
        self.kill_reason = reason
        self.live_armed = False
        self.supervisor.note("KILL", reason)
        if self.live is not None:
            await self.live.flatten_everything(self.slots)

    async def _step_slot(self, slot: SlotRuntime, now: float, venue: dict[str, Any] | None = None) -> None:
        """Fail-safe wrapper: a lane error must never leave a live binary unguarded.

        The incident this exists for: an unhandled `NameError` inside the live exit
        path fired every tick, so take-profit and scratch placement never happened
        and 85 contracts went to settlement unhedged. Any exception while this slot
        holds a live position now pulls its orders and crosses out reduce-only.
        """
        try:
            await self._step_slot_inner(slot, now, venue)
        except Exception as exc:  # noqa: BLE001 - the fail-safe is the whole point
            self.fault_times = (self.fault_times + [now])[-100:]
            self.supervisor.note("LANE_FAULT", f"{slot.cfg.series}: {type(exc).__name__}: {exc}")
            if (self.live_armed and self.live is not None
                    and (slot.position is not None or slot.order is not None)):
                try:
                    await self._fail_safe_slot(slot, exc, now)
                except Exception as nested:  # noqa: BLE001 - report, never re-raise
                    slot.halted = True
                    slot.reason = "fail-safe could not flatten: position open"
                    self.supervisor.note(
                        "FAIL_SAFE_FAILED", f"{slot.cfg.series}: {type(nested).__name__}: {nested}")

    async def _fail_safe_slot(self, slot: SlotRuntime, exc: BaseException, now: float) -> None:
        """Cancel this ticker's resting orders, IOC-flatten the position, halt the lane."""
        slot.halted = True
        slot.window_halted = True
        flat = await self.live.emergency_flatten(  # type: ignore[union-attr]
            slot, f"fail-safe {type(exc).__name__}")
        slot.reason = ("fail-safe flatten" if flat
                       else "fail-safe IOC unconfirmed: position may be open")
        self.supervisor.note("FAIL_SAFE", f"{slot.cfg.series}: {slot.reason}")

    async def _step_slot_inner(self, slot: SlotRuntime, now: float,
                               venue: dict[str, Any] | None = None) -> None:
        quote = slot.quote
        if quote is None:
            return
        # A position belongs to one contract. If the lane has already moved on to the
        # next window, settle the old one against its own last quote before doing
        # anything else — never price an exit off a different market's ladder.
        if slot.position is not None and slot.position.ticker != quote.ticker:
            await self._roll_slot(slot, slot.quotes.get(slot.position.ticker, quote), now)
            return
        # The socket's ladder is the only quote with sub-second truth. A missing side
        # means that side is empty: falling back to a REST snapshot that can be 40s old
        # is how the paper tape booked a 31c fill on a book that was already 0.999 with
        # no asks. `received_ts` is the evaluation clock so a tick and the fill it
        # triggers share one timestamp.
        book = self.ws.books.get(quote.ticker) if self.ws else None
        if book is not None and book.synced and now - book.updated_ts < 5.0:
            quote = replace(
                quote,
                yes_bid=book.yes_bid if book.yes_bid is not None else Decimal("0"),
                yes_ask=book.yes_ask if book.yes_ask is not None else Decimal("1"),
                # Carry the ladder's touch sizes with the price: the live executor
                # clips its order to half the resting ask it would pay, and a quote
                # without depth cannot be sized against the book it is trading.
                yes_bid_sz=book.yes_bid_sz,
                yes_ask_sz=book.yes_ask_sz,
                updated_ts=book.updated_ts,
                received_ts=now,
            )
        else:
            quote = replace(quote, received_ts=now)
        slot.quote = quote
        slot.quotes[quote.ticker] = quote
        seconds_left = max(0.0, quote.close_ts - now)
        symbol = slot.cfg.feed_symbol or ""
        fresh = self.feed.freshest(symbol) if symbol else None
        spot, spot_age, spot_source = (fresh if fresh else (slot.spot, self.feed.age(symbol), "tape"))
        slot.spot_source = spot_source
        pts = self.tape.points(symbol)
        sigma = features.realized_sigma(pts)
        mom = features.momentum(pts)
        # Priced off the freshest constituent print, which is the only print that can
        # be ahead of the venue's own book.
        fv = features.fair_value(spot, quote.strike, seconds_left, sigma, mom) if sigma > 0 else None
        slot.sigma = sigma
        slot.p_up = fv.p_up if fv else 0.5
        signal = self._micro_signal(slot, quote, seconds_left, sigma)
        impulse_bps = signal.impulse_bps if signal is not None else 0.0
        # Settlement truth: once the final 60s average is fully printed it is
        # arithmetic, and any position carried into the roll settles on it. The
        # staleness guard stops the previous window's finished average being read as
        # this window's at T-59.
        latest = self.index_prints.get(INDEX_BY_SERIES.get(slot.cfg.series) or "")
        if latest is not None and latest.final_minute is not None:
            if (latest.final_minute_ticks >= 58
                    and latest.final_minute_ticks <= max(0.0, 60.0 - seconds_left) + 3):
                slot.lock_in_settlement = (quote.ticker, float(latest.final_minute),
                                           quote.strike)
        if signal is not None:
            slot.p_up = signal.p_up
            slot.signal = signal.name
            slot.signal_detail = signal.detail
        else:
            slot.signal = "gbm" if fv else ""
            slot.signal_detail = fv and f"z {fv.z:+.2f}" or ""
        # A probability that will price money goes through the validated calibrator
        # before any edge is computed. Identity today; the path is what matters.
        slot.p_up = features.calibrate(slot.p_up, self.calibrators.get(slot.cfg.series))

        # A crossed book (bid >= ask) is a stale side, not a gift. The paper executor
        # would otherwise buy the stale ask and instantly sell it at the real bid — the
        # exact artifact that inflated this soak's SOL fills. No entry on a crossed book.
        crossed = quote.yes_bid >= quote.yes_ask
        # F2's probability is the trade's payoff probability, not `P(YES)`, so it
        # cannot pick the direction for the correlation guard: the move does.
        if signal is not None and signal.name == "F2_taker_1s":
            prospective = "YES" if slot.spot >= quote.strike else "NO"
        else:
            prospective = "YES" if slot.p_up >= quote.mid else "NO"
        # Live is configured but readiness has not flipped live_armed yet. Opening a
        # simulated paper position now would pollute the live epoch ledger, defer ARM_LIVE
        # to the next window roll (ARM_WAIT), and break the live-epoch parity control —
        # the exact failure the 4:06-4:40 PM session hit. Stream quotes and tape only.
        awaiting_arm = (self.settings.live and self.settings.arm_live and not self.live_armed)
        # The zero-drift GBM is a reference price, not a thesis: measured against the
        # book it is worse (Brier 0.2271 vs 0.2187) and it lost money on the tape. It may
        # not open a position. Only a microstructure signal can.
        can_open = ((signal is not None) and not crossed and not slot.halted
                    and not awaiting_arm
                    and self._window_open(slot, signal)
                    and self._capacity_ok(slot, prospective))
        hurdle = self._edge_hurdle(slot)
        decision = decide(
            quote=quote, p_up=slot.p_up, seconds_left=seconds_left, spot_age=spot_age,
            params=slot.params, position=slot.position, edge_hurdle=hurdle,
            can_open=can_open, signal=slot.signal, now=now, spot=slot.spot,
            impulse_bps=impulse_bps,
            l2_obi=signal.l2_obi if signal is not None else 0.0,
            flow_fast=signal.flow_fast if signal is not None else 0.0,
            flow_slow=signal.flow_slow if signal is not None else 0.0,
        )
        slot.edge_streak, decision = with_persistence(slot.edge_streak, decision, slot.params)
        # Laya reviews every lane it is asked about, but never on the caller's
        # clock: the review is scheduled in the background and its verdict is
        # cached per ticker. The entry path below reads that cache, so the tick
        # loop never waits on a forward pass.
        self._gate_refresh(slot, quote, seconds_left, decision)
        slot.gate_ticker = quote.ticker
        if decision.action == "entry":
            decision = self._gate_veto(slot, quote, decision)
        slot.decision = decision
        slot.state = "halted" if slot.halted else decision.regime
        slot.reason = decision.reason

        if self.live_armed and self.live is not None:
            # Real orders: the venue is the only source of fills.
            await self.live.step(slot, decision, quote, seconds_left, now, venue or {"orders": {}, "fills": []})
        elif awaiting_arm:
            # Live is configured but the readiness gate has not flipped this run. Do NOT
            # place simulated paper orders or open a paper position while waiting: stream
            # the tape, show "arming", and arm on the first green readiness check.
            slot.state = "arming"
            slot.reason = "awaiting readiness: paper orders suppressed until live arming"
        else:
            self.executor.review_order(slot, quote, now)
            self.executor.manage_order(slot, quote, seconds_left)
            self.executor.poll_exit(slot, quote, now)
            self.executor.execute(slot, decision, quote, seconds_left, now)

        # Every evaluation is persisted, at whatever cadence the loop runs. The replay
        # has to see the exact quote state that triggered each fill, including the
        # sub-second ladder move a 1 Hz decimation silently dropped. `spot_age` and the
        # effective hurdle are recorded too, so the replay re-derives the same decision
        # instead of guessing with defaults.
        slot.last_tick_store = now
        self.store.tick(
            ts=now, series=slot.cfg.series, ticker=quote.ticker,
            yes_bid=str(quote.yes_bid), yes_ask=str(quote.yes_ask), last=str(quote.last),
            spot=slot.spot, p_model=slot.p_up, edge=float(decision.edge),
            secs_left=seconds_left, spread=float(quote.spread),
            signal=slot.signal, can_open=can_open, stake=str(slot.bankroll),
            spot_age=spot_age, hurdle=str(hurdle), impulse_bps=impulse_bps,
            gate=json.dumps(slot.gate) if slot.gate else "",
        )
        slot.tape.append({"ts": now, "p": slot.p_up, "mid": float(quote.mid),
                          "edge": float(decision.edge), "spot": slot.spot})
        slot.tape = slot.tape[-180:]

    def _edge_hurdle(self, slot: SlotRuntime) -> Decimal:
        # Calibration pressure: a miscalibrated slot must clear a higher bar.
        return slot.params.edge_hurdle + Decimal(str(round(max(0.0, slot.ece - 0.03) * 0.5, 4)))

    @staticmethod
    def _window_open(slot: SlotRuntime, signal: Any) -> bool:
        """Per-window entry cap and halt for the promoted flow programs.

        A 15-minute contract is one slot, not a stream: the frozen study only ever
        counted the first two entries in a window and stopped the window entirely
        once one of them lost. F2's study counted the first three and had no halt,
        so its cap is three and the halt below is not applied to it.
        """
        name = getattr(signal, "name", "") if signal is not None else ""
        if name == "F2_taker_1s":
            return slot.window_entries < slot.params.f2_max_window_trades
        if name == "student_t_l2_settle":
            return slot.window_entries < slot.params.sl2_max_window_trades
        if signal is None or name != "z_obi_flow":
            return True
        return (slot.window_entries < slot.params.max_window_trades
                and not slot.window_halted)

    def _capacity_ok(self, slot: SlotRuntime, prospective: str | None = None) -> bool:
        """Portfolio correlation guard.

        Crypto 15-minute lanes are one trade wearing four tickers: a single BTC shock
        re-prices all of them together. So risk is capped at two lanes at once and at
        one lane per direction, which stops four correlated posts from falling as one.
        """
        open_positions = [s for s in self.slots if s.position is not None]
        if len(open_positions) >= self.settings.max_concurrent_positions:
            return False
        crypto = [s for s in open_positions if s.cfg.asset != "GOLD"]
        if len(crypto) >= self.settings.max_concurrent_crypto:
            return False
        if prospective:
            same = [s for s in crypto if s.position is not None and s.position.side == prospective]
            if len(same) >= self.settings.max_same_direction_crypto:
                return False
        return True

    # ---------- execution ----------
    # Fills, exits and the ledger live in execution.Executor so the live loop and the
    # tape replay cannot drift apart. Nothing below writes a trade by hand.

    # ---------- accounting and snapshot ----------
    def _mark(self) -> None:
        marked = Decimal("0")
        for slot in self.slots:
            if slot.position and slot.quote:
                marked += slot.position.side_bid(slot.quote) * slot.position.contracts
            elif slot.order:
                marked += slot.order.reserved
        self.equity = self.cash + marked
        self.peak_equity = max(self.peak_equity, self.equity)

    def _reconcile_report(self) -> dict[str, Any] | None:
        """The last `scripts/reconcile_live.py --json` run for this epoch, if any."""
        path = Path(self.settings.db_path).parent / "reconcile-live.json"
        try:
            mtime = path.stat().st_mtime
            cached = getattr(self, "_reconcile_cache", None)
            if not cached or cached[0] != mtime:
                raw = json.loads(path.read_text())
                cached = (mtime, {k: raw.get(k) for k in (
                    "generated", "since", "venue_net", "ledger_net", "delta",
                    "open_positions", "resting_orders")} | {"ts": mtime})
                self._reconcile_cache = cached
            report = cached[1]
            return report if float(report.get("since") or 0) >= self.epoch_start - 1 else None
        except (OSError, ValueError, TypeError):
            return None

    @property
    def running_program(self) -> str:
        """The program the lanes are actually running right now."""
        return self.slots[0].params.strategy_mode if self.slots else "F2_taker_1s"

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        open_positions = sum(1 for s in self.slots if s.position is not None)
        return {
            "generated_at": now,
            # Execution is paper until a venue broker is mounted in this plane. The
            # dashboard must never imply real orders are going out when they are not.
            "mode": "live" if self.live_armed else "paper",
            # The program this process actually ran, not a hardcoded name: the board
            # must not claim F2 while the lane is running the settlement program.
            "strategy": self.running_program,
            "env": self.settings.env_name,
            "live_armed": self.live_armed,
            "armed_at": self.arm_ts,
            "kill_reason": self.kill_reason,
            "venue_balance": str(self.venue_balance.quantize(Decimal("0.01"))),
            "readiness": __import__("openkal.readiness", fromlist=["evaluate"]).evaluate(self),
            "gate": {
                "url": self.gate.url, "calls": self.gate.calls, "failures": self.gate.failures,
                "mode": self.gate.mode,
                "last": verdict_payload(self.gate.last),
            },
            "ws": {
                "messages": self.ws.messages if self.ws else 0,
                "lag_ms": round(self.ws.lag_ms if self.ws else 0.0, 1),
                "degraded": bool(self.ws.degraded) if self.ws else True,
                "books": len(self.ws.books) if self.ws else 0,
                "indices": len(self.ws.indices) if self.ws else 0,
                "reconnects": self.ws.reconnects if self.ws else 0,
                "sync_problems": self.ws.sync_problems if self.ws else 0,
                "last_error": self.ws.last_error if self.ws else None,
                "final_minute_index": sum(1 for p in (self.ws.indices.values() if self.ws else [])
                                          if p.final_minute is not None),
            },
            "uptime_s": now - self.started,
            "health": {
                "coinbase": self.feed.connected.get("coinbase", False),
                "kraken": self.feed.connected.get("kraken", False),
                "spot_messages": self.feed.messages,
                "market_errors": self.market_errors,
                "exchange_active": self.exchange_active,
                "latency_ms": round(self.client.last_latency_ms, 1),
                "venue_calls": self.client.calls,
                "recent_errors": len([t for t in self.error_times if time.time() - t < 600]),
                "last_error": self.client.last_error,
                "feed_errors": self.feed.errors[-3:],
            },
            "account": {
                "equity": str(self.equity.quantize(Decimal("0.01"))),
                "start_equity": str(self.start_equity),
                "cash": str(self.cash.quantize(Decimal("0.01"))),
                "realized": str(self.realized.quantize(Decimal("0.01"))),
                "day_pnl": str((self.equity - self.day_start_equity).quantize(Decimal("0.01"))),
                "open_positions": open_positions,
                "peak_equity": str(self.peak_equity.quantize(Decimal("0.01"))),
            },
            "capacity": {"open": open_positions,
                         "max": self.settings.max_concurrent_positions},
            "epoch_start": self.epoch_start,
            "reconcile": self._reconcile_report(),
            "curve": self.equity_curve[-240:],
            "slots": [self._slot_payload(s, now) for s in self.slots],
            "fills": list(reversed(self.fills[-24:])),
            "trades": self.store.recent_trades(18),
            "shr": {
                "state": self.supervisor.state,
                "generation": self.supervisor.generation,
                "promoted": self.supervisor.promoted,
                "rejected": self.supervisor.rejected,
                "fault": None if not self.supervisor.fault else {
                    "reason": self.supervisor.fault.reason,
                    "ts": self.supervisor.fault.ts,
                    "ticks": self.supervisor.fault.ticks,
                },
                "candidates": [
                    {"name": c.name, "status": c.status, "score": c.score,
                     "profit_factor": c.profit_factor, "ece": c.ece, "note": c.note}
                    for c in self.supervisor.candidates
                ],
                "timeline": self.supervisor.timeline[-8:],
                "programs": self.store.programs(12),
            },
            "window": {
                "open_ts": min((s.window_open_ts for s in self.slots if s.window_open_ts), default=0),
                "close_ts": min((s.quote.close_ts for s in self.slots if s.quote), default=0),
            },
        }

    def _slot_payload(self, slot: SlotRuntime, now: float) -> dict[str, Any]:
        quote = slot.quote
        pos = slot.position
        secs = max(0.0, quote.close_ts - now) if quote else 0.0
        return {
            "id": slot.cfg.series,
            "series": slot.cfg.series,
            "asset": slot.cfg.asset,
            "signal": slot.signal,
            "signal_detail": slot.signal_detail,
            "display": slot.cfg.display,
            "state": slot.state,
            "reason": slot.reason,
            "halted": slot.halted,
            "ticker": quote.ticker if quote else None,
            "strike": quote.strike if quote else None,
            "spot": slot.spot,
            "spot_age_s": round(self.feed.age(slot.cfg.feed_symbol or ""), 2),
            "p_up": round(slot.p_up, 4),
            "sigma": slot.sigma,
            "yes_bid": str(quote.yes_bid) if quote else None,
            "yes_ask": str(quote.yes_ask) if quote else None,
            "last": str(quote.last) if quote else None,
            "spread": str(quote.spread) if quote else None,
            "secs_left": round(secs, 1),
            "window_open_ts": slot.window_open_ts,
            "window_close_ts": quote.close_ts if quote else 0,
            "edge": str(slot.decision.edge) if slot.decision else "0",
            "confidence": round(slot.decision.confidence, 4) if slot.decision else 0,
            "bankroll": str(slot.bankroll.quantize(Decimal("0.01"))),
            "pnl_today": str(slot.pnl_today.quantize(Decimal("0.01"))),
            "wins": slot.wins,
            "losses": slot.losses,
            "streak": slot.consecutive_losses,
            "ece": round(slot.ece, 4),
            "gate": slot.gate,
            "params": slot.params.to_json(),
            "position": None if not pos else {
                "side": pos.side, "entry": str(pos.entry), "contracts": pos.contracts,
                "tp": str(pos.tp), "sl": str(pos.sl), "opened_ts": pos.opened_ts,
                "mark": str(pos.side_bid(quote)) if quote else None,
            },
            "order": None if not slot.order else {
                "side": slot.order.side, "price": str(slot.order.price),
                "contracts": slot.order.contracts, "age_s": round(now - slot.order.placed_ts, 1),
                "filled": str(slot.order.seen_fill),
            },
            "exit": None if not slot.exit else {
                "price": str(slot.exit.price), "contracts": slot.exit.contracts,
                "filled": str(slot.exit.seen_fill), "reason": slot.exit.reason,
            },
            "window_entries": slot.window_entries,
            "max_window_trades": slot.params.max_window_trades,
            "last_trade": slot.last_trade,
            "tape": slot.tape[-90:],
        }
