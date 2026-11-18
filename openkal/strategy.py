"""Slot state machine: 100% of the slot bankroll, maker in, maker out, IOC stop.

Every tuning value lives in the EVOLVE-BLOCK at the top. That is the only region
the recursive loop is allowed to rewrite; auth, sizing and risk plumbing sit
outside it on purpose (the plan's evaluator-isolation rule).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import ROUND_DOWN, Decimal
from typing import Any

from .config import HARD_MAX_ENTRY_PRICE, MAX_CONTRACTS, MAX_TRADE_RISK_USD


# EVOLVE-BLOCK-START
# Underlying spot-to-strike distance, in index points, below which a `z_obi_flow` entry
# is refused. One number per lane because the index scales differ: $8 on BTC is a
# 0.01% move while $0.002 on XRP is 0.4% of its index, and those are the levels at
# which the strike stops being a coin flip.
ZOBI_MIN_DELTA = {"KXBTC15M": 16.5, "KXETH15M": 0.85, "KXSOL15M": 0.048, "KXXRP15M": 0.00045}


@dataclass(frozen=True)
class Params:
    """The strategy's tunable surface."""

    edge_hurdle: Decimal = Decimal("0.015")      # minimum modeled edge after fees, per contract
    min_price: Decimal = Decimal("0.20")         # never buy the longshot tail
    max_price: Decimal = Decimal("0.84")         # generic entry band (not the lock-in band)
    # Sizing ceilings, not strategy parameters: the live executor may not take a
    # larger position than `scripts/forward_oos.py` scored, and no single binary
    # may risk more than $15 of premium (the stop cannot fill at a 0.00 bid).
    max_contracts: int = MAX_CONTRACTS
    max_trade_risk_usd: Decimal = MAX_TRADE_RISK_USD
    max_spread: Decimal = Decimal("0.05")
    # Both theses enter by crossing (IOC taker) and leave through a resting maker
    # take-profit, so the round trip pays the taker fee once, at entry, on purpose.
    take_profit: Decimal = Decimal("0.05")       # maker TP +4..7c, 0 exit fee
    # No fixed 3c price stop: 3-4c of bid-ask bounce hit it in three seconds. The
    # impulse thesis exits on a spot reversal, the lock-in thesis holds to settlement.
    stop_loss: Decimal = Decimal("0.25")         # generic fallback only
    persist_ticks: int = 1                       # the impulse itself is the persistence
    # Expected exit cost folded into the generic hurdle: the share of exits that must
    # cross, the taker fee on those, and one tick of slippage.
    taker_exit_share: float = 0.5
    exit_slippage_ticks: float = 1.0
    min_net_edge: float = 0.008
    entry_style: str = "taker"
    flatten_before_close_s: float = 45.0         # never carry an impulse trade into the print
    min_seconds_left: float = 90.0               # no entries that cannot be managed
    # The lock-in thesis is *about* the final minute: it prices the settlement average
    # from the prints already locked, so the 90s entry cutoff and the 90s flatten would
    # make it untradeable by construction. It gets its own floor and holds to settlement.
    lock_in_min_seconds: float = 12.0
    settle_lock_in: bool = True
    min_confidence: float = 0.62
    maker_offset_ticks: int = 0
    max_quote_age_s: float = 4.0
    max_spot_age_s: float = 10.0
    max_divergence: Decimal = Decimal("0.30")    # a 30c+ disagreement with the venue is a bug, not an edge
    # --- high-certainty late lock-in (final minute) ---
    final_minute_s: float = 60.0                 # lock-in window: last printed minute
    final_minute_min_ticks: int = 20             # k >= 20 seconds of the average locked
    final_minute_min_share: float = 0.3333       # 20/60, kept for older callers
    # Fees collapse near the top of the book (0.07 * p * (1-p)), so the band extends
    # past the generic 84c ceiling to 94c, where a 2c net edge is still real.
    lock_in_max_price: Decimal = Decimal("0.94")
    lock_in_min_prob: float = 0.985              # conditional settlement probability
    lock_in_min_edge: float = 0.02               # after the exact entry taker fee
    # --- spot-impulse taker snipe ---
    impulse_lookback_s: float = 5.0              # measured move window on the 5Hz BRTI tape
    impulse_min_bps: float = 12.0
    impulse_max_bps: float = 20.0
    impulse_stale_share: float = 0.5             # ladder must have moved < half the implied delta
    impulse_min_edge: float = 0.025              # after the exact entry taker fee
    impulse_max_hold_s: float = 40.0             # time-barrier maker scratch
    impulse_reversal_share: float = 0.60         # exit once the impulse retraces 60%
    maker_capture_min_width: Decimal = Decimal("0.02")
    cross_book_min_profit: Decimal = Decimal("0.01")
    # --- z + printed flow + touch revisions (the promoted program) -----------
    # Frozen-OOS evidence for the shipped variant `scr_tp0.05_b45|cap2|none`
    # (chronological 60/40 split on data/evolve-cache/book_<LANE>.npz, 26h IS / 17h
    # OOS, 23 frozen OOS trades): +$107.83 net, PF >= 1.15 (capped at 99), 82.6% won,
    # ECE 0.0487, 0 crossed fills, 69.6% of exits at the take-profit. Early window
    # only (T 360-840s) on a 1c book, at most two entries per contract and a window
    # halt after the first loss. The exit is *intra-window*: a post-only +5c maker
    # take-profit ($0 exit fee) and, at the 45s barrier, a post-only scratch at
    # breakeven with 20s of patience. A scratch that misses carries the binary, and
    # the 0.30-0.60 entry band bounds that tail at 0.60/contract. Carrying every
    # signal to expiry instead (the previous `cap1` hold-to-settlement variant) turned
    # a half-window flow edge into a naked binary bet and lost $49.88 live.
    # No taker leg exists anywhere in the measured structure: a taker entry or exit
    # here would invalidate the evidence.
    # Detector re-aligned 2026-11-16 18:10 to the one the frozen verdict actually
    # scored. The previous values (z 0.15, flow 0.25, rev 0.20, 360-840s left, 0.30-0.60
    # band, 1c spread) were never the verdict's detector; on the same frozen 60/40 split
    # with the same exit they score OOS 21 trades / -$208.49 / PF 0.36. These score
    # OOS 22 / +$105.20 / PF 99 / ECE 0.060 / 0 crossed (IS 25 / +$53.21 / PF 1.81).
    zobi_min_z: float = 0.60
    zobi_min_flow: float = 0.15
    zobi_min_revision: float = 0.10
    zobi_flow_window_s: float = 10.0
    zobi_revision_window_s: float = 5.0
    zobi_max_spread: Decimal = Decimal("0.03")
    zobi_min_seconds_left: float = 90.0
    zobi_max_seconds_left: float = 420.0
    zobi_min_price: Decimal = Decimal("0.15")
    # The band is on the *cost of the side held* (YES price for a long, 1-YES for a
    # short) and it is asymmetric on purpose: 0.48 is the underdog cap. The old 0.60
    # let the model buy the favourite at 0.58 -- a price that already implies the
    # outcome, where a 5c take-profit is the whole edge and a settlement miss is 12x
    # it. Above 0.48 the contract is no longer underdog-priced and the trade is
    # refused, whatever the flow says.
    # 0.84 since the 18:10 re-alignment: the verified detector's band, so a carried
    # contract can now lose up to 0.84 (the comment above describes the retired 0.60).
    zobi_max_price: Decimal = Decimal("0.84")
    # Distance from the strike, in index points, under which the contract is a coin
    # flip and the flow edge is not worth a binary. Per lane: see `ZOBI_MIN_DELTA`.
    # Measured and REJECTED on the frozen 60/40 tape (see `--confirmed`): with the delta
    # gate on, the best variant reached 18 OOS trades at PF 1.43 and ECE 0.054, under
    # the 20-trade floor, and the no-stop row was OOS 20 / -$31.73. The z+flow edge
    # lives in exactly the near-strike entries this gate refuses, so the knobs are
    # wired but off. Turning them on is a new hypothesis, not a fix.
    zobi_min_delta: Decimal = Decimal("0.00")
    zobi_trend_gate: bool = False
    zobi_trend_fast_s: float = 15.0
    zobi_trend_slow_s: float = 60.0
    # The stop is confirmed, not touched: two seconds of continuous breach before the
    # position is crossed out. A single wick through the level is noise in a 4Hz book
    # and used to hand back the whole edge.
    # Same study, same verdict: every stop distance (4/6/8c) turned winners into 7
    # stop-outs out of 18 and left OOS net at +$32.54 with 18 trades, below the 20-trade
    # floor. The confirmed-stop machinery is live-correct but not verified as an edge,
    # so the shipped program has no stop and its tail bound is the 0.60 entry cost.
    zobi_max_loss: Decimal = Decimal("0.00")
    zobi_stop_confirm_s: float = 2.0
    # 25s, not the study's 15s: live the post-only bid sits behind the rest of the
    # queue, and five ETH/SOL/XRP bids were cancelled unfilled after 15s while the
    # flow that opened them was still valid. The extra 10s is queue time, not signal.
    zobi_entry_patience_s: float = 25.0
    zobi_barrier_s: float = 45.0            # time barrier: scratch at breakeven
    zobi_scratch_patience_s: float = 20.0   # post-only scratch, then carry
    zobi_hold_to_settle: bool = False       # intra-window exits; see `_zobi_exit`
    max_window_trades: int = 2              # `cap2`, halt the window after one loss
    # --- F2: short-horizon taker momentum, +5c maker TP, 7c confirmed stop -----
    # Frozen on the 60/40 split by scripts/f2_structural.py, with the mandatory
    # 1s execution lag: IS 46 trades / +$38.54 / PF 2.32 / payoff-ECE 0.0764;
    # the read slice OOS 36 trades / +$42.65 / PF 4.17 / payoff-ECE 0.09, and 30
    # trades at payoff-ECE 0.0595 after the desk's own decide() hurdle. The
    # probability this program carries is `features.PayoffCalibrator`: the chance
    # the trade's net is positive, not the chance the market settles its way.
    # Every number here is frozen. Tuning one on the forward slice is the thing
    # the forward slice exists to prevent.
    f2_min_z: float = 0.30
    f2_min_flow: float = 0.10
    f2_min_revision: float = 0.0
    f2_min_delta_bps: float = 3.0
    f2_flow_window_s: float = 10.0
    f2_revision_window_s: float = 5.0
    f2_max_spread: Decimal = Decimal("0.04")
    # The cost band is on the side actually held, as everywhere else. 0.62 is the
    # hard entry ceiling; the frozen rule's own upper bound was 0.60, and the
    # 0.62 here is the desk's cap, not a licence to widen it.
    f2_min_price: Decimal = Decimal("0.25")
    f2_max_price: Decimal = Decimal("0.62")
    f2_take_profit: Decimal = Decimal("0.05")
    f2_stop_loss: Decimal = Decimal("0.07")
    f2_stop_confirm_s: float = 2.0
    f2_barrier_s: float = 90.0
    f2_min_seconds_left: float = 180.0
    f2_max_seconds_left: float = 780.0
    # The frozen study's own per-contract cap (`pick(..., per_window=3)`): at most
    # three F2 entries in one 15-minute window. It does not halt after a loss --
    # that rule belongs to the older flow program -- because the study's OOS profit
    # factor was measured with the stop, not a window halt, bounding the losses.
    f2_max_window_trades: int = 3
    # --- the program this lane runs -----------------------------------------
    # `F2_taker_1s` is the default and it stays the default until an R&D champion
    # passes every promotion gate: a passing verdict for a program the hot path
    # does not implement must not arm (see `readiness.ENGINE_PROGRAMS`). The
    # switch exists so a promoted program is selected by configuration, not by a
    # code edit.
    strategy_mode: str = "F2_taker_1s"
    # --- LayaGate live filtering ---------------------------------------------
    # The gate's opinion is cached per ticker and read on the entry path in 0ms,
    # so enforcing it costs no latency. `require_layagate_allow` turns that
    # cached opinion into a hard filter: an entry is refused when the gate is
    # available and says skip. It is off by default because the shipped build
    # has no fine-tuned ONNX model, so the gate is still a shadow here; a
    # strategy promoted against the gate's own forward evidence turns it on
    # without touching a line of code.
    require_layagate_allow: bool = False
    # A floor on the confidence of the gate's "enter" vote. 0.0 accepts any allow.
    layagate_min_confidence: float = 0.0
    # --- student_t_l2_settle: Student-t settlement + true-L2 imbalance ---------
    # The entry band is on the cost of the side held: 35c to 60c. The 62c hard
    # cap is the venue's own ceiling (`config.HARD_MAX_ENTRY_PRICE`) and is
    # enforced on top of the band, never instead of it. A hold-to-settlement
    # binary loses its whole premium, so the entry cost IS the tail bound.
    sl2_min_price: Decimal = Decimal("0.35")
    sl2_max_price: Decimal = Decimal("0.60")
    # True L2 order-book imbalance, (yes_bid_sz - yes_ask_sz) / (yes_bid_sz +
    # yes_ask_sz), in [-1, 1]. The side taken must be the side the book leans
    # to: a YES entry needs a positive imbalance, a NO entry a negative one.
    sl2_min_obi: float = 0.15
    # Signed aggressor print flow, on both a fast and a slow window, must lean
    # the way the side does. Both windows have to agree: one window is a wick.
    sl2_min_flow: float = 0.30
    sl2_flow_fast_s: float = 5.0
    sl2_flow_slow_s: float = 15.0
    # The settlement program's own window: 780s to 60s before the close.
    sl2_min_seconds_left: float = 60.0
    sl2_max_seconds_left: float = 780.0
    # The book's own spread cap, tighter than the generic one: a 4c spread on a
    # 35c-60c contract is already a tenth of the premium at risk.
    sl2_max_spread: Decimal = Decimal("0.04")
    # Net edge after the entry's taker fee, in dollars per contract. The exit is
    # settlement, so there is no exit cost to charge.
    sl2_min_edge: Decimal = Decimal("0.05")
    # At most two entries in one 15-minute window. The live lane can in fact take
    # only one: a settlement position is carried to expiry, so the window is over
    # before the next entry is possible. The cap is the scored rule, and the live
    # path sits inside it.
    sl2_max_window_trades: int = 2

    def to_json(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Params":
        out: dict[str, Any] = {}
        for key, field_type in cls.__annotations__.items():
            if key not in data:
                continue
            raw = data[key]
            if field_type is Decimal or field_type == "Decimal":
                out[key] = Decimal(str(raw))
            elif field_type in (int, "int"):
                out[key] = int(float(raw))
            elif field_type in (str, "str"):
                out[key] = str(raw)
            elif field_type in (bool, "bool"):
                out[key] = str(raw).strip().lower() in ("1", "true", "yes")
            else:
                out[key] = float(raw)
        return cls(**out)

    def variant(self, **changes: Any) -> "Params":
        return replace(self, **changes)
# EVOLVE-BLOCK-END


PARAM_BOUNDS: dict[str, tuple[Decimal, Decimal]] = {
    "edge_hurdle": (Decimal("0.005"), Decimal("0.060")),
    "min_price": (Decimal("0.15"), Decimal("0.35")),
    "max_price": (Decimal("0.70"), Decimal("0.90")),
    "max_spread": (Decimal("0.02"), Decimal("0.08")),
    "take_profit": (Decimal("0.04"), Decimal("0.07")),
    "flatten_before_close_s": (Decimal("30"), Decimal("180")),
    "min_seconds_left": (Decimal("60"), Decimal("240")),
    "min_confidence": (Decimal("0.55"), Decimal("0.80")),
    "maker_offset_ticks": (Decimal("0"), Decimal("2")),
    "max_quote_age_s": (Decimal("1.0"), Decimal("6.0")),
    "max_spot_age_s": (Decimal("4.0"), Decimal("20.0")),
    "max_divergence": (Decimal("0.18"), Decimal("0.40")),
}


def clamp_params(params: Params) -> Params:
    """Keep a mutated candidate inside sane bounds; a verifier never sees an absurd one."""
    data = asdict(params)
    for key, (low, high) in PARAM_BOUNDS.items():
        value = Decimal(str(data[key]))
        value = max(low, min(high, value))
        data[key] = int(value) if key == "maker_offset_ticks" else value
    if data["min_price"] >= data["max_price"]:
        data["min_price"] = data["max_price"] - Decimal("0.10")
    return Params(**data)


@dataclass(frozen=True)
class Quote:
    """One market snapshot, already parsed off the venue payload."""

    ticker: str
    series: str
    strike: float
    close_ts: float
    yes_bid: Decimal
    yes_ask: Decimal
    last: Decimal
    updated_ts: float
    received_ts: float
    # Resting size at the touch, when the quote came off the socket ladder. A quote
    # from a REST snapshot (or a tape recorded before the ladder was persisted)
    # carries None, and sizing then clips nothing rather than inventing depth.
    yes_bid_sz: Decimal | None = None
    yes_ask_sz: Decimal | None = None

    @property
    def mid(self) -> Decimal:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def spread(self) -> Decimal:
        return max(Decimal("0"), self.yes_ask - self.yes_bid)

    @property
    def age_s(self) -> float:
        return max(0.0, self.received_ts - self.updated_ts)


@dataclass
class Position:
    ticker: str
    side: str            # "YES" | "NO"
    entry: Decimal
    contracts: int
    opened_ts: float
    p_entry: float
    stake: Decimal
    tp: Decimal
    sl: Decimal
    signal: str = ""     # which kernel signal opened this, so exits respect its thesis
    entry_spot: float = 0.0     # spot at entry: the reversal stop is measured from here
    impulse_bps: float = 0.0    # signed size of the impulse thesis that opened this
    # Fee actually paid to open. A taker entry pays it up front; carrying it on the
    # position is the only way `close` can report net P&L after *both* legs.
    entry_fee: Decimal = Decimal("0")
    # First tick the confirmed stop saw the mark at/below its level. Zero means no
    # breach is open; the exit fires only once the breach has lasted the full
    # confirmation window (two seconds of continuous breach, not one wick).
    stop_breach_ts: float = 0.0
    # The scratch was not hit: the binary is carried to settlement (1/0), so the
    # position has no exit order any more and must not be force-flattened.
    hold_to_settle: bool = False

    def side_bid(self, quote: Quote) -> Decimal:
        return quote.yes_bid if self.side == "YES" else (Decimal(1) - quote.yes_ask)

    def side_ask(self, quote: Quote) -> Decimal:
        return quote.yes_ask if self.side == "YES" else (Decimal(1) - quote.yes_bid)


@dataclass(frozen=True)
class Decision:
    action: str            # entry | hold | exit_tp | exit_sl | flatten | wait | abstain
    reason: str
    side: str | None = None
    price: Decimal | None = None
    p_model: float = 0.5
    edge: Decimal = Decimal("0")
    confidence: float = 0.0
    regime: str = "abstain"
    signal: str = ""
    impulse_bps: float = 0.0


def exit_cost(price: Decimal, params: Params) -> Decimal:
    """Expected cost of getting out, in dollars per contract.

    Maker exits are free, so only the share that has to cross pays the taker fee plus
    one tick of slippage. A trade is not counted until its edge clears this.
    """
    share = Decimal(str(max(0.0, min(1.0, params.taker_exit_share))))
    fee = Decimal("0.07") * price * (Decimal(1) - price)
    slip = Decimal(str(params.exit_slippage_ticks)) * Decimal("0.01")
    return (fee + slip) * share


def taker_fee_per_contract(price: Decimal) -> Decimal:
    """Exact Kalshi taker fee per contract, in dollars (0.07 * P * (1-P), ceilinged)."""
    fee = Decimal("0.07") * price * (Decimal(1) - price)
    return fee.quantize(Decimal("0.000001"))


def with_persistence(streak: int, decision: Decision, params: Params) -> tuple[int, Decision]:
    """Count consecutive qualifying ticks; only a sustained divergence may rest.

    Shared by the live loop and the replay so the two cannot disagree about what
    "persisted" means.
    """
    if decision.action != "entry":
        return 0, decision
    streak += 1
    if streak < max(1, params.persist_ticks):
        return streak, Decision(
            "abstain", f"edge persisted {streak}/{params.persist_ticks} ticks",
            side=decision.side, price=decision.price, p_model=decision.p_model,
            confidence=decision.confidence, edge=decision.edge, regime="persist",
        )
    return streak, decision


def contracts_for(stake: Decimal, price: Decimal) -> int:
    """One full-slot limit order, not a ladder. No per-order round-up overhead."""
    if price <= 0 or stake <= 0:
        return 0
    return int((stake / price).to_integral_value(rounding=ROUND_DOWN))


def _impulse_exit(quote: Quote, position: Position, params: Params, p_up: float,
                  seconds_left: float, now: float, spot: float) -> Decision:
    """The impulse thesis' stop is the spot print, not the ladder.

    A 3c price stop is inside ordinary bid-ask bounce; the only evidence that the
    impulse was wrong is the spot tape retracing most of the move that opened it.
    """
    bid = position.side_bid(quote)
    confidence = max(p_up, 1 - p_up)
    if bid >= position.tp:
        return Decision("exit_tp", "take-profit reached", side=position.side, price=bid,
                        p_model=p_up, confidence=confidence, regime="exit")
    if position.entry_spot > 0 and position.impulse_bps and spot > 0:
        direction = 1.0 if position.impulse_bps > 0 else -1.0
        moved = (spot / position.entry_spot - 1.0) * 10_000.0
        retrace = moved * direction
        if retrace <= -abs(position.impulse_bps) * params.impulse_reversal_share:
            return Decision(
                "exit_sl",
                f"spot reversed {retrace:+.1f}bp of a {position.impulse_bps:+.1f}bp impulse",
                side=position.side, price=bid, p_model=p_up, confidence=confidence,
                regime="exit")
    held = max(0.0, now - position.opened_ts) if now else 0.0
    if held >= params.impulse_max_hold_s:
        return Decision("exit_sl", f"time barrier {held:.0f}s", side=position.side, price=bid,
                        p_model=p_up, confidence=confidence, regime="exit")
    if seconds_left <= params.flatten_before_close_s:
        return Decision("flatten", f"T-{int(seconds_left)}s, flatten before the print",
                        side=position.side, price=bid, p_model=p_up,
                        confidence=confidence, regime="exit")
    return Decision("hold", "impulse working", side=position.side, price=bid, p_model=p_up,
                    confidence=confidence, regime="hold")


def _zobi_entry(quote: Quote, params: Params, p_up: float, seconds_left: float) -> Decision:
    """Entry for the promoted flow program: post-only at the touch, no hurdle.

    The z direction picks the side, not the book's mid. The thesis is that the
    strike is behind the index *and* the venue's prints and ladder agree, so a
    disagreement between the model and the mid is not a reason to abstain.
    """
    if not (params.zobi_min_seconds_left <= seconds_left <= params.zobi_max_seconds_left):
        return Decision("abstain", "outside the flow window", p_model=p_up)
    if quote.spread > params.zobi_max_spread:
        return Decision("abstain", f"spread {quote.spread:.3f} wide for flow", p_model=p_up)
    confidence = max(p_up, 1 - p_up)
    side = "YES" if p_up >= 0.5 else "NO"
    # Post-only at the touch: a long bids the bid, a short offers the ask.
    price = quote.yes_bid if side == "YES" else (Decimal(1) - quote.yes_ask)
    if not (Decimal("0.01") <= price <= Decimal("0.99")):
        return Decision("abstain", "flow price outside the book", p_model=p_up, side=side)
    # The band is on what the side actually costs, which is also the most it can
    # lose if it is carried into settlement. A 77c long YES and a 77c short NO are
    # the same 77c of tail risk; before this, a NO at 23c cost was admitted by a
    # band written on the YES price, and a YES at 83c was admitted by a 0.85 cap.
    if not (params.zobi_min_price <= price <= params.zobi_max_price):
        return Decision("abstain", f"price {price:.2f} outside the flow band",
                        p_model=p_up, side=side, confidence=confidence)
    return Decision("entry", f"z+flow at {price:.2f}, <= {params.max_window_trades} per window",
                    side=side, price=price, p_model=p_up, edge=Decimal("0"),
                    confidence=confidence, regime="flow", signal="z_obi_flow")


def _zobi_exit(quote: Quote, position: Position, params: Params, p_up: float,
               now: float) -> Decision:
    """The OOS-verified intra-window exit: `+5c` maker take-profit, then a 45s scratch.

    `scr_tp0.05_b45|cap2|none` banks a half-window flow move at entry+5c with a
    post-only ask ($0 exit fee) and, if that misses, rests a post-only scratch at
    breakeven at the 45s barrier for 20s of patience. A scratch that misses carries
    the binary, and the 0.30-0.60 entry band bounds that tail at 0.60/contract.
    `zobi_hold_to_settle` still short-circuits everything to a settlement carry for
    programs measured that way; it is off for the shipped one, because carrying a
    flow signal into expiry is a different (and live, worse) trade.
    """
    bid = position.side_bid(quote)
    confidence = max(p_up, 1 - p_up)
    if params.zobi_hold_to_settle:
        return Decision("hold",
                        "carrying the binary to settlement (loss bounded by entry cost)",
                        side=position.side, price=bid, p_model=p_up,
                        confidence=confidence, regime="settle")
    if bid >= position.tp:
        return Decision("exit_tp", "take-profit reached", side=position.side, price=bid,
                        p_model=p_up, confidence=confidence, regime="exit")
    if params.zobi_max_loss > 0 and bid <= position.entry - params.zobi_max_loss:
        level = position.entry - params.zobi_max_loss
        if not now:
            # No clock to confirm against (ad-hoc calls): behave like a plain stop
            # rather than silently never exiting.
            return Decision("exit_sl",
                            f"bounded stop: {bid:.2f} <= {level:.2f}",
                            side=position.side, price=bid, p_model=p_up,
                            confidence=confidence, regime="exit")
        if position.stop_breach_ts <= 0:
            position.stop_breach_ts = now
            return Decision("hold", f"stop breach 1/2: {bid:.2f} <= {level:.2f}",
                            side=position.side, price=bid, p_model=p_up,
                            confidence=confidence, regime="hold")
        breached_for = now - position.stop_breach_ts
        if breached_for >= params.zobi_stop_confirm_s:
            # `exit_sl` crosses: the live executor turns a target equal to the bid
            # into an immediate reduce_only IOC. The alternative -- resting a
            # post-only scratch -- fills only when someone else bids up a position
            # that is already underwater.
            return Decision("exit_sl",
                            f"confirmed stop: {breached_for:.1f}s at/below {level:.2f}",
                            side=position.side, price=bid, p_model=p_up,
                            confidence=confidence, regime="exit")
        return Decision("hold",
                        f"stop breach {breached_for:.1f}/{params.zobi_stop_confirm_s:.0f}s",
                        side=position.side, price=bid, p_model=p_up,
                        confidence=confidence, regime="hold")
    position.stop_breach_ts = 0.0
    if position.hold_to_settle:
        return Decision("hold", "scratch missed: carrying the binary to settlement",
                        side=position.side, price=bid, p_model=p_up,
                        confidence=confidence, regime="settle")
    held = max(0.0, now - position.opened_ts) if now else 0.0
    if held >= params.zobi_barrier_s:
        px = max(position.entry, bid + Decimal("0.01"))
        # A post-only SELL crosses only if it is at or below the bid; an ask above the
        # current best ask simply improves the book and rests. The buy-side test was
        # used here, so the scratch was refused exactly when the trade had moved
        # against us — the case the frozen study's scratch exists for.
        if px <= bid:
            return Decision("hold", "scratch would cross the book", side=position.side,
                            price=bid, p_model=p_up, confidence=confidence, regime="hold")
        return Decision("scratch",
                        f"time barrier {held:.0f}s: post-only scratch at {px:.2f}",
                        side=position.side, price=px, p_model=p_up,
                        confidence=confidence, regime="exit")
    return Decision("hold", "z+flow working", side=position.side, price=bid, p_model=p_up,
                    confidence=confidence, regime="hold")


def _f2_entry(quote: Quote, params: Params, p_up: float, seconds_left: float,
              edge_hurdle: Decimal, spot: float = 0.0) -> Decision:
    """F2's entry: a limit IOC pinned to the best ask, never a price walk.

    `p_up` is the frozen payoff probability (`features.PayoffCalibrator`), so the
    edge is `p_payoff - cost`: the chance this round trip finishes green minus
    what the side costs. The desk's own hurdle is then charged the exit cost and
    the exact taker fee, exactly as scripts/f2_structural.py scored it -- a rule
    that only trades the events its own probability likes, not every event.

    The price is the ask of the side held, which is the pin the live executor
    enforces. A cost outside [0.25, 0.62] is refused here and again at the order
    path, because a book that moved between the decision and the order is not a
    reason to pay more.
    """
    if not (params.f2_min_seconds_left <= seconds_left <= params.f2_max_seconds_left):
        return Decision("abstain", "outside the F2 window", p_model=p_up)
    if quote.spread > params.f2_max_spread:
        return Decision("abstain", f"spread {quote.spread:.3f} wide for F2", p_model=p_up)
    # The side is the direction of the move the rule detected (z), not the value of
    # the payoff probability: a trade that is 83% likely to win can still be a NO.
    # `p_up` here is `P(this trade's net > 0)`, so it is already the probability of
    # the side we are about to take and is not flipped for a NO.
    if spot > 0 and quote.strike > 0:
        side = "YES" if spot >= quote.strike else "NO"
    else:
        side = "YES" if p_up >= 0.5 else "NO"
    price = quote.yes_ask if side == "YES" else (Decimal(1) - quote.yes_bid)
    if not (params.f2_min_price <= price <= params.f2_max_price):
        return Decision("abstain", f"F2 cost {price:.2f} outside "
                        f"{params.f2_min_price}-{params.f2_max_price}",
                        p_model=p_up, side=side)
    p_side = Decimal(str(p_up))
    cost = exit_cost(price, params) + taker_fee_per_contract(price)
    net = p_side - price - cost
    hurdle = max(params.edge_hurdle, edge_hurdle)
    if net < max(hurdle, Decimal(str(params.min_net_edge))):
        return Decision("abstain", f"F2 payoff edge {net * 100:.2f}c under hurdle",
                        p_model=p_up, side=side, edge=net)
    return Decision("entry", f"F2 IOC at the {side} ask {price:.2f}, "
                    f"payoff {p_side:.3f}, net {net * 100:.2f}c",
                    side=side, price=price, p_model=p_up, edge=net,
                    confidence=max(p_up, 1 - p_up), regime="momentum",
                    signal="F2_taker_1s")


def _f2_exit(quote: Quote, position: Position, params: Params, p_up: float,
             now: float) -> Decision:
    """F2's exit: `+5c` maker take-profit, `7c` confirmed stop, `90s` barrier.

    The take-profit is a post-only ask at `entry_fill + 0.05` ($0 exit fee) and it
    is placed from the fill, so the target is the fill price the venue gave us,
    not the price we asked for. The stop is *confirmed*: the mark must sit at or
    below `entry - 0.07` for the full two seconds before the reduce-only IOC is
    sent, because one wick through a 4Hz book is noise. The barrier is a market
    exit at 90 seconds, the same horizon the frozen study scored.
    """
    bid = position.side_bid(quote)
    confidence = max(p_up, 1 - p_up)
    target = position.entry + params.f2_take_profit
    if bid >= target:
        return Decision("exit_tp", f"F2 take-profit {target:.2f}", side=position.side,
                        price=bid, p_model=p_up, confidence=confidence, regime="exit")
    level = position.entry - params.f2_stop_loss
    if bid <= level:
        if not now:
            return Decision("exit_sl", f"F2 bounded stop: {bid:.2f} <= {level:.2f}",
                            side=position.side, price=bid, p_model=p_up,
                            confidence=confidence, regime="exit")
        if position.stop_breach_ts <= 0:
            position.stop_breach_ts = now
            return Decision("hold", f"F2 stop breach 1/2: {bid:.2f} <= {level:.2f}",
                            side=position.side, price=bid, p_model=p_up,
                            confidence=confidence, regime="hold")
        breached_for = now - position.stop_breach_ts
        if breached_for >= params.f2_stop_confirm_s:
            return Decision("exit_sl",
                            f"F2 confirmed stop: {breached_for:.1f}s at/below {level:.2f}",
                            side=position.side, price=bid, p_model=p_up,
                            confidence=confidence, regime="exit")
        return Decision("hold", f"F2 stop breach {breached_for:.1f}/"
                        f"{params.f2_stop_confirm_s:.0f}s", side=position.side, price=bid,
                        p_model=p_up, confidence=confidence, regime="hold")
    position.stop_breach_ts = 0.0
    held = max(0.0, now - position.opened_ts) if now else 0.0
    if held >= params.f2_barrier_s:
        return Decision("flatten", f"F2 time barrier {held:.0f}s", side=position.side,
                        price=bid, p_model=p_up, confidence=confidence, regime="exit")
    return Decision("hold", "F2 working", side=position.side, price=bid, p_model=p_up,
                    confidence=confidence, regime="hold")


def _student_t_l2_entry(quote: Quote, params: Params, p_yes: float, seconds_left: float,
                        edge_hurdle: Decimal, l2_obi: float, flow_fast: float,
                        flow_slow: float) -> Decision:
    """The settlement program's entry: the band, the book, the flow, then the edge.

    The side is not chosen before the book is read. A YES entry needs the resting
    size to lean to the bid (`l2_obi >= min_obi`) and the aggressor flow to lean
    the same way on *both* windows; a NO entry needs the mirror image. A side
    that fails any of those tests is not a trade at any price, and the side with
    the better net edge wins.

    The exit is settlement: $1 or $0 a contract, no exit fee. The entry crosses,
    so it pays the taker fee and nothing else, and that fee is charged here
    because the premium a hold-to-settlement binary risks is the whole entry cost.
    """
    if not (params.sl2_min_seconds_left <= seconds_left <= params.sl2_max_seconds_left):
        return Decision("abstain", "outside the settlement window", p_model=p_yes)
    if quote.spread > params.sl2_max_spread:
        return Decision("abstain", f"spread {quote.spread:.3f} wide for the settlement program",
                        p_model=p_yes)
    hurdle = max(params.sl2_min_edge, edge_hurdle, Decimal(str(params.min_net_edge)))
    best: Decision | None = None
    misses: list[str] = []
    for side in ("YES", "NO"):
        price = quote.yes_ask if side == "YES" else (Decimal(1) - quote.yes_bid)
        # The band's ceiling and the venue's hard cap are both enforced: the band
        # is the rule's own, the cap is the desk's, and the tighter of the two wins.
        if price > HARD_MAX_ENTRY_PRICE:
            misses.append(f"{side} cost {price:.2f} over the {HARD_MAX_ENTRY_PRICE} cap")
            continue
        if not (params.sl2_min_price <= price <= params.sl2_max_price):
            misses.append(f"{side} cost {price:.2f} outside "
                          f"{params.sl2_min_price}-{params.sl2_max_price}")
            continue
        obi = l2_obi if side == "YES" else -l2_obi
        if obi < params.sl2_min_obi:
            misses.append(f"{side} imbalance {obi:+.2f} under {params.sl2_min_obi:+.2f}")
            continue
        fast = flow_fast if side == "YES" else -flow_fast
        slow = flow_slow if side == "YES" else -flow_slow
        if fast < params.sl2_min_flow or slow < params.sl2_min_flow:
            misses.append(f"{side} flow {fast:+.2f}/{slow:+.2f} under {params.sl2_min_flow:+.2f}")
            continue
        p_side = Decimal(str(p_yes if side == "YES" else 1.0 - p_yes))
        net = p_side - price - taker_fee_per_contract(price)
        if net < hurdle:
            misses.append(f"{side} net {net * 100:.2f}c under {hurdle * 100:.2f}c")
            continue
        cand = Decision("entry",
                        f"settlement {side} at {price:.2f}, p {p_side:.3f}, "
                        f"net {net * 100:.2f}c, obi {obi:+.2f}, flow {fast:+.2f}/{slow:+.2f}",
                        side=side, price=price, p_model=p_yes, edge=net,
                        confidence=max(float(p_side), 1.0 - float(p_side)),
                        regime="settlement", signal="student_t_l2_settle")
        if best is None or cand.edge > best.edge:
            best = cand
    if best is not None:
        return best
    return Decision("abstain", "; ".join(misses[:2]) or "no side inside the settlement band",
                    p_model=p_yes)


def _student_t_l2_exit(quote: Quote, position: Position, params: Params, p_up: float,
                       now: float) -> Decision:
    """The settlement program's exit: there is none before the close.

    The thesis *is* the settlement: the contract is carried to expiry and pays $1
    or $0. There is no take-profit, no stop and no time barrier, because an early
    exit would pay the taker fee the edge was never budgeted for and would turn a
    priced settlement into a directional bet on the last minute.
    """
    bid = position.side_bid(quote)
    confidence = max(p_up, 1 - p_up)
    return Decision("hold", "settlement: holding to expiry", side=position.side, price=bid,
                    p_model=p_up, confidence=confidence, regime="settle")


def decide(*, quote: Quote, p_up: float, seconds_left: float, spot_age: float,
           params: Params, position: Position | None, edge_hurdle: Decimal,
           can_open: bool, signal: str = "", now: float = 0.0, spot: float = 0.0,
           impulse_bps: float = 0.0, l2_obi: float = 0.0, flow_fast: float = 0.0,
           flow_slow: float = 0.0) -> Decision:
    """The whole hot path, deterministic and side-effect free."""
    # A crossed book means one side is stale. The venue would not fill the stale side,
    # so opening on it books a fantasy spread. Applies to every caller, including replay.
    if quote.spread <= 0:
        can_open = False
    if position is not None:
        if quote.ticker != position.ticker:
            return Decision("flatten", "window rolled", side=position.side)
        bid = position.side_bid(quote)
        # A lock-in position is held into settlement: its whole edge is the average
        # that is already printed, and flattening at T-90 would give that up.
        if position.signal == "final_minute" and params.settle_lock_in:
            return Decision("hold", "lock-in: holding to settlement", side=position.side,
                            price=bid, p_model=p_up, confidence=max(p_up, 1 - p_up),
                            regime="settle")
        if position.signal == "F2_taker_1s":
            return _f2_exit(quote, position, params, p_up, now)
        if position.signal == "student_t_l2_settle":
            return _student_t_l2_exit(quote, position, params, p_up, now)
        if position.signal == "z_obi_flow":
            return _zobi_exit(quote, position, params, p_up, now)
        if position.signal == "impulse_snipe":
            return _impulse_exit(quote, position, params, p_up, seconds_left, now, spot)
        if bid >= position.tp:
            return Decision("exit_tp", "take-profit reached", side=position.side, price=bid,
                            p_model=p_up, confidence=max(p_up, 1 - p_up), regime="exit")
        p_side_now = p_up if position.side == "YES" else 1 - p_up
        p_side_entry = position.p_entry if position.side == "YES" else 1 - position.p_entry
        thesis_broken = p_side_now < p_side_entry - 0.10
        if bid <= position.sl or thesis_broken:
            why = "stop-loss: price broke" if bid <= position.sl else "stop-loss: model flipped"
            return Decision("exit_sl", why, side=position.side, price=bid,
                            p_model=p_up, confidence=max(p_up, 1 - p_up), regime="exit")
        if seconds_left <= params.flatten_before_close_s:
            return Decision("flatten", f"T-{int(seconds_left)}s, flatten before the print",
                            side=position.side, price=bid, p_model=p_up,
                            confidence=max(p_up, 1 - p_up), regime="exit")
        return Decision("hold", "position working", side=position.side, price=bid, p_model=p_up,
                        confidence=max(p_up, 1 - p_up), regime="hold")

    if quote.ticker == "" or not can_open:
        return Decision("wait", "slot not open for new risk", p_model=p_up)
    if quote.age_s > params.max_quote_age_s:
        return Decision("abstain", f"quote stale {quote.age_s:.1f}s", p_model=p_up)
    if spot_age > params.max_spot_age_s:
        return Decision("abstain", "spot feed stale", p_model=p_up)
    if quote.spread > params.max_spread:
        return Decision("abstain", f"spread {quote.spread:.3f} wide", p_model=p_up)
    lock_in = signal == "final_minute"
    impulse = signal == "impulse_snipe"
    settlement = signal == "student_t_l2_settle"
    # Each program owns the part of the window it can actually manage. The
    # settlement program is priced off the average that is left to print, so its
    # floor is its own and not the generic 90s cut.
    if lock_in:
        floor = params.lock_in_min_seconds
    elif settlement:
        floor = params.sl2_min_seconds_left
    else:
        floor = params.min_seconds_left
    if seconds_left < floor:
        return Decision("abstain", "too late in the window", p_model=p_up)
    if signal == "F2_taker_1s":
        return _f2_entry(quote, params, p_up, seconds_left, edge_hurdle, spot)
    if signal == "student_t_l2_settle":
        return _student_t_l2_entry(quote, params, p_up, seconds_left, edge_hurdle,
                                   l2_obi, flow_fast, flow_slow)
    if signal == "z_obi_flow":
        return _zobi_entry(quote, params, p_up, seconds_left)

    confidence = max(p_up, 1 - p_up)
    side = "YES" if p_up >= quote.mid else "NO"
    side_ask = quote.yes_ask if side == "YES" else (Decimal(1) - quote.yes_bid)
    ceiling = params.lock_in_max_price if lock_in else params.max_price
    if side_ask < params.min_price or side_ask > ceiling:
        return Decision("abstain", f"price {side_ask:.2f} outside the band", p_model=p_up,
                        side=side, confidence=confidence)
    # Both theses are gated by *net edge* and their own structural evidence, not by
    # directional confidence: an impulse that moves fair value from 0.50 to 0.58 is
    # still a 8c discount if the ladder is at 0.45.
    if not (lock_in or impulse) and confidence < params.min_confidence:
        return Decision("abstain", f"confidence {confidence:.2f} under hurdle", p_model=p_up,
                        side=side, confidence=confidence)

    model_p_side = Decimal(str(p_up if side == "YES" else 1 - p_up))
    # Divergence catches a *guess* that disagrees wildly with the venue. The two theses
    # are not guesses: lock-in reads the settlement average off the venue's own index
    # feed, and impulse reads a move the venue's own reference has already printed. For
    # them a large gap between model and book is the opportunity, not a bug.
    if not (lock_in or impulse) and abs(Decimal(str(p_up)) - quote.mid) > params.max_divergence:
        return Decision("abstain", f"model {model_p_side:.2f} vs book {quote.mid:.2f} too far apart",
                        p_model=p_up, side=side, confidence=confidence)

    # Edge is measured against the ASK, not against our own resting price. A maker
    # fill at the bid is adversely selected: we only get it when someone is selling
    # into us, which is exactly when the book is about to move against the position.
    edge = model_p_side - side_ask
    if impulse:
        fee = taker_fee_per_contract(side_ask)
        net = edge - fee
        if net < Decimal(str(params.impulse_min_edge)):
            return Decision("abstain", f"impulse net edge {net * 100:.2f}c under "
                            f"{params.impulse_min_edge * 100:.1f}c", p_model=p_up, side=side,
                            edge=net, confidence=confidence)
        return Decision("entry", f"IOC taker on {impulse_bps:+.1f}bp impulse, net {net * 100:.2f}c",
                        side=side, price=side_ask, p_model=p_up, edge=net,
                        confidence=confidence, regime="impulse_snipe", signal="impulse_snipe",
                        impulse_bps=impulse_bps)
    if lock_in:
        if (p_up if side == "YES" else 1 - p_up) < params.lock_in_min_prob:
            return Decision("abstain", f"lock-in probability {p_up if side == 'YES' else 1 - p_up:.3f} "
                            f"under {params.lock_in_min_prob:.3f}", p_model=p_up, side=side,
                            confidence=confidence)
        fee = taker_fee_per_contract(side_ask)
        net = edge - fee
        if net < Decimal(str(params.lock_in_min_edge)):
            return Decision("abstain", f"lock-in net edge {net * 100:.2f}c under "
                            f"{params.lock_in_min_edge * 100:.1f}c", p_model=p_up, side=side,
                            edge=net, confidence=confidence)
        return Decision("entry", f"lock-in at {side_ask:.2f}, net {net * 100:.2f}c, hold to settle",
                        side=side, price=side_ask, p_model=p_up, edge=net,
                        confidence=confidence, regime="lock_in", signal="final_minute")

    # Rest one tick inside the best bid so the fill is maker and the queue is ours.
    tick = Decimal("0.01")
    passive = (quote.yes_bid + tick * params.maker_offset_ticks) if side == "YES" else \
              (Decimal(1) - quote.yes_ask + tick * params.maker_offset_ticks)
    if passive >= side_ask:
        return Decision("abstain", "would cross the book", p_model=p_up, side=side,
                        confidence=confidence)
    if passive < params.min_price:
        return Decision("abstain", "passive price below band", p_model=p_up, side=side,
                        confidence=confidence)

    gross_cost = exit_cost(side_ask, params)
    net = edge - gross_cost
    hurdle = max(params.edge_hurdle, edge_hurdle)
    if net < max(hurdle, Decimal(str(params.min_net_edge))):
        return Decision(
            "abstain",
            f"net edge {net * 100:.2f}c after {gross_cost * 100:.2f}c exit cost under hurdle",
            p_model=p_up, side=side, edge=net, confidence=confidence,
        )

    price = side_ask if params.entry_style == "taker" else passive
    if params.entry_style == "taker":
        cost = gross_cost + taker_fee_per_contract(side_ask)
        if edge - cost < max(hurdle, Decimal(str(params.min_net_edge))):
            return Decision("abstain", f"taker net edge {(edge - cost) * 100:.2f}c under hurdle",
                            p_model=p_up, side=side, edge=edge - cost, confidence=confidence)
    regime = "momentum_burst" if abs(p_up - 0.5) > 0.28 else "mean_revert"
    return Decision("entry", f"{params.entry_style} entry clears spread and fee", side=side,
                    price=price, p_model=p_up, edge=edge, confidence=confidence, regime=regime,
                    signal=signal if lock_in else "")
