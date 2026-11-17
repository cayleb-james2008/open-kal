"""Tape replay: the engine's own execution core, driven by recorded venue ticks.

This is not a second backtester. It builds the same slot objects the live loop uses,
calls the same `strategy.decide`, and drives the same `execution.Executor`. If the
replay and the live ledger disagree, one of them is a bug.

Every tick comes from `store.ticks`, which is written from real Kalshi orderbook
snapshots and the real spot tape. Nothing here invents a price.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .execution import Executor, RestingExit, RestingOrder
from .strategy import Decision, Params, Position, Quote, decide, with_persistence


@dataclass
class ReplaySlot:
    cfg: Any
    params: Params
    bankroll: Decimal
    peak_bankroll: Decimal
    quote: Quote | None = None
    spot: float = 0.0
    p_up: float = 0.5
    position: Position | None = None
    order: RestingOrder | None = None
    exit: RestingExit | None = None
    halted: bool = False
    flatten_requested: bool = False
    consecutive_losses: int = 0
    wins: int = 0
    losses: int = 0
    pnl_today: Decimal = Decimal("0")
    ece: float = 0.0
    pairs: list[tuple[float, int]] = field(default_factory=list)
    last_trade: dict[str, Any] | None = None
    state: str = ""
    reason: str = ""
    weak_ticks: int = 0
    edge_streak: int = 0


@dataclass
class ReplayContext:
    cash: Decimal
    realized: Decimal = Decimal("0")
    fills: list[dict[str, Any]] = field(default_factory=list)
    store: Any = None
    supervisor: Any = None
    source: str = "tape"
    trade_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReplayReport:
    series: str
    trades: int
    pnl: Decimal
    wins: int
    losses: int
    win_rate: float
    avg_entry_edge: Decimal
    avg_win: Decimal
    avg_loss: Decimal
    profit_factor: float
    max_drawdown: Decimal
    fees: Decimal
    start_cash: Decimal
    end_cash: Decimal


def quote_from_tick(tick: dict[str, Any], series: str, received_ts: float) -> Quote:
    return Quote(
        ticker=tick["ticker"], series=series, strike=0.0,
        close_ts=received_ts + float(tick["secs_left"]),
        yes_bid=Decimal(str(tick["yes_bid"])), yes_ask=Decimal(str(tick["yes_ask"])),
        last=Decimal(str(tick["last"])), updated_ts=received_ts, received_ts=received_ts,
    )


def _param_timeline(params: Any) -> list[tuple[float, Params]] | None:
    """Normalise `Params` or `[(ts, Params), ...]` into a sorted timeline.

    A live epoch can span a program change (this one does: it opened under a hold-to-
    settlement variant and now runs the intra-window one). Replaying the whole epoch
    with the latest params rewrites trades that happened under the old ones, so the
    engine records when each program started and the replay switches there.
    """
    if isinstance(params, Params):
        return None
    try:
        segments = sorted(((float(ts), p) for ts, p in params), key=lambda s: s[0])
    except (TypeError, ValueError):
        return None
    return segments or None


def _params_at(segments: list[tuple[float, Params]] | None, params: Params,
               ts: float) -> Params:
    if not segments:
        return params
    current = segments[0][1]
    for start, candidate in segments:
        if ts < start:
            break
        current = candidate
    return current


def replay_series(cfg: Any, ticks: list[dict[str, Any]], params: Any, stake: Decimal,
                  store: Any = None, source: str = "tape", record: bool = False,
                  venue_fills: list[dict[str, Any]] | None = None,
                  venue_anchored: bool = False,
                  venue_exits: list[dict[str, Any]] | None = None) -> tuple[ReplayReport, ReplayContext]:
    """Walk one series' recorded ticks through the shared decision and execution paths.

    `venue_fills` are the entries the *exchange* reported for a live epoch. Queue
    position is not in the tape, so a print-only rule would book a resting bid that
    100 lots printed through but that was never matched. Pass the venue's own entry
    fills for a live epoch; omit them for a paper/fixture epoch, where the print tape
    is the only evidence that exists.
    """
    segments = _param_timeline(params)
    if segments:
        params = segments[0][1]
    # The tape carries the bankroll the live loop actually sized with. Starting the
    # replay with a different number silently drops every order that needs more cash
    # than the guess, which looks like "no trades" rather than an error.
    tape_stake = next((Decimal(str(t["stake"])) for t in ticks if t.get("stake")), stake)
    # The live executor sizes per slot but checks affordability against the *account*
    # cash. Starting the replay with one slot's slice made it reject every full-slot
    # order the live engine actually filled. Use the epoch's starting equity.
    cash = tape_stake
    if store is not None:
        try:
            base = Decimal(str(store.get_meta("epoch_equity") or 0))
            if base > 0:
                cash = base
        except Exception:  # noqa: BLE001 - a missing meta must not stop the replay
            pass
    ctx = ReplayContext(cash=cash)
    slot = ReplaySlot(cfg=cfg, params=params, bankroll=tape_stake, peak_bankroll=tape_stake)
    # `tape=store` and not `ctx.store=store`: the replay needs the venue print tape to
    # prove a maker fill, but it must never write fills into the ledger it is checking.
    ex = Executor(ctx, tape=store, venue_fills=list(venue_fills or []),
                  venue_anchored=venue_anchored, venue_exits=list(venue_exits or []))
    ex.flatten_maker_s = 8.0
    edges: list[Decimal] = []
    equity = Decimal("0")
    peak = Decimal("0")
    max_dd = Decimal("0")
    fees = Decimal("0")
    entry_edges: list[Decimal] = []

    for tick in ticks:
        ts = float(tick["ts"])
        q = quote_from_tick(tick, cfg.series, ts)
        # Window roll: the live loop closes the old contract at the last seen quote
        # when the next window is discovered. Without this the replay would exit at
        # the *new* market's price, which is not a price the live engine ever saw.
        if slot.quote is not None and q.ticker != slot.quote.ticker:
            if slot.order is not None:
                ctx.cash += slot.order.reserved
                ex.record(slot, "cancel", slot.order.side, slot.order.price,
                          slot.order.contracts, Decimal("0"), "window rolled")
                slot.order = None
            if slot.position is not None:
                ex.close(slot, slot.position.side_bid(slot.quote), "window rolled", maker=False)
            slot.exit = None
            slot.edge_streak = 0
        slot.quote = q
        slot.spot = float(tick.get("spot") or 0.0)
        slot.p_up = float(tick["p_model"])
        slot.params = _params_at(segments, params, ts)
        slot.state = ""
        before = slot.position
        # The venue's own exit print wins over any simulated exit for the same epoch.
        ex.maybe_venue_exit(slot, q, ts)

        ex.review_order(slot, q, ts)
        ex.manage_order(slot, q, float(tick["secs_left"]))
        ex.poll_exit(slot, q, ts)
        # Parity with the live loop: a tick the kernel did not price with one of its
        # microstructure signals cannot open a position, exactly as in production.
        # The portfolio gate (capacity, correlation, halt) is state the replay cannot
        # reconstruct per lane, so the live loop records its own verdict with the tick.
        # Without a recorded verdict, fall back to the signal rule.
        if tick.get("can_open") is not None:
            can_open = bool(tick["can_open"])
        else:
            can_open = (tick.get("signal") or "") not in ("", "gbm")
        tick_stake = tick.get("stake")
        if tick_stake:
            try:
                slot.bankroll = Decimal(str(tick_stake))
            except Exception:  # noqa: BLE001 - a malformed stake must not stop the replay
                pass
        # Freshness and the calibration-adjusted hurdle were inputs to the live
        # decision; the tick carries them so the replay does not guess.
        spot_age = float(tick.get("spot_age") or 0.0)
        hurdle = params.edge_hurdle
        if tick.get("hurdle"):
            try:
                hurdle = Decimal(str(tick["hurdle"]))
            except Exception:  # noqa: BLE001
                hurdle = params.edge_hurdle
        impulse = float(tick.get("impulse_bps") or 0.0)
        d = decide(quote=q, p_up=slot.p_up, seconds_left=float(tick["secs_left"]),
                   spot_age=spot_age, params=slot.params, position=slot.position,
                   edge_hurdle=hurdle, can_open=can_open and not slot.halted,
                   signal=(tick.get("signal") or ""), now=ts, spot=slot.spot,
                   impulse_bps=impulse)
        slot.edge_streak, d = with_persistence(slot.edge_streak, d, slot.params)
        slot.state = d.regime
        ex.execute(slot, d, q, float(tick["secs_left"]), ts)
        if before is None and slot.position is not None:
            entry_edges.append(abs(Decimal(str(slot.p_up)) - q.mid))
        if before is not None and slot.position is None and slot.last_trade is not None:
            # The probability the trade claimed for the *side it held*: a NO trade's
            # claim is `1 - p_up`, and scoring it with the raw YES probability
            # grades a different claim than the one the strategy made.
            p_yes = float(slot.last_trade["p_entry"])
            f2 = slot.last_trade.get("signal") == "F2_taker_1s"
            p_side = (p_yes if (slot.last_trade.get("side") == "YES" or f2)
                      else 1.0 - p_yes)
            ctx.trade_log.append({"ts": slot.last_trade["ts"], "p": p_yes,
                                  "p_side": p_side,
                                  "pnl": Decimal(slot.last_trade["pnl"])})
        edges.append(d.edge)

        equity = ctx.realized
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    # Anything still open at the end of the tape is marked out at the last known bid.
    if slot.position is not None and slot.quote is not None:
        ex.close(slot, slot.position.side_bid(slot.quote), "tape ended", maker=False)

    # Both legs: a taker entry's fee is part of the round trip, not an externality.
    all_fees = sum((f.get("fee") or Decimal("0")) for f in ctx.fills if f["kind"] in {"entry", "exit"})

    wins = [t for t in _trades(ctx) if t["pnl"] > 0]
    losses = [t for t in _trades(ctx) if t["pnl"] <= 0]
    gross_win = sum((t["pnl"] for t in wins), Decimal("0"))
    gross_loss = -sum((t["pnl"] for t in losses), Decimal("0"))
    pf = float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    trades = _trades(ctx)
    report = ReplayReport(
        series=cfg.series, trades=len(trades), pnl=ctx.realized,
        wins=len(wins), losses=len(losses),
        win_rate=(len(wins) / len(trades)) if trades else 0.0,
        avg_entry_edge=(sum(entry_edges, Decimal("0")) / len(entry_edges)) if entry_edges else Decimal("0"),
        avg_win=(gross_win / len(wins)) if wins else Decimal("0"),
        avg_loss=(gross_loss / len(losses)) if losses else Decimal("0"),
        profit_factor=pf, max_drawdown=max_dd, fees=Decimal(str(all_fees)),
        start_cash=stake, end_cash=stake + ctx.realized,
    )
    return report, ctx


def _trades(ctx: ReplayContext) -> list[dict[str, Any]]:
    out = []
    for f in ctx.fills:
        if f["kind"] != "exit":
            continue
        out.append({"pnl": Decimal(str(f["pnl"])), "p": f.get("p_entry", 0.5)})
    return out


def replay_all(store: Any, slots: list[Any], params_for, since_ts: float, source: str = "tape",
               record: bool = False) -> dict[str, Any]:
    """Replay every recorded lane and add the reports up."""
    reports: list[ReplayReport] = []
    for cfg in slots:
        ticks = store.recent_ticks(cfg.series, since_ts, limit=200000)
        if len(ticks) < 30:
            continue
        stake = store.slot_stake(cfg.series) if hasattr(store, "slot_stake") else Decimal("50")
        report, _ = replay_series(cfg, ticks, params_for(cfg), stake, store=store, source=source)
        reports.append(report)
    total = sum((r.pnl for r in reports), Decimal("0"))
    trades = sum(r.trades for r in reports)
    wins = sum(r.wins for r in reports)
    return {
        "trades": trades,
        "wins": wins,
        "losses": sum(r.losses for r in reports),
        "win_rate": (wins / trades) if trades else 0.0,
        "pnl": str(total),
        "net": float(total),
        "profit_factor": (
            float(sum((r.avg_win * r.wins for r in reports), Decimal("0")) /
                  max(Decimal("0.01"), sum((r.avg_loss * r.losses for r in reports), Decimal("0"))))
        ),
        "avg_entry_edge": str(
            sum((r.avg_entry_edge * r.trades for r in reports), Decimal("0")) / max(1, trades)
        ),
        "reports": [r.__dict__ | {"pnl": str(r.pnl), "avg_entry_edge": str(r.avg_entry_edge),
                                  "avg_win": str(r.avg_win), "avg_loss": str(r.avg_loss),
                                  "max_drawdown": str(r.max_drawdown),
                                  "start_cash": str(r.start_cash), "end_cash": str(r.end_cash)}
                    for r in reports],
    }
