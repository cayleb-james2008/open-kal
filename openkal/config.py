"""One place for every runtime knob: credentials paths, venue, slots and safety gates.

Nothing here decides to trade. The gates are read by the strategy and the engine;
`ALLOW_LIVE` is the only switch that can put real money at risk, and it stays off
unless the operator sets it deliberately.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Live sizing ceiling, set to the evidence. `scripts/forward_oos.py` scored the
# frozen rule with MAX_QTY = 50 contracts per trade, and the live lane must not
# buy more than the study covered: a 24c NO became 170 contracts ($40.80 of
# premium) against a 50-contract study, and a binary that settles to zero loses
# the whole premium, not the 7c stop distance.
MAX_CONTRACTS = 50
MAX_TRADE_RISK_USD = Decimal("15.00")

# --- staged fractional-Kelly compounding ---------------------------------
# A lane that risks a flat $15 cannot compound: at $121.46 of equity that is 12.3%
# of the account, and at $350 it is 4.3% -- the program would shrink relative to
# the money it trades. The stages below hold the *fraction* of equity at risk
# roughly constant as the account grows, and cap each stage at the contract count
# that fraction buys at the price ceiling.
#
# A hold-to-settlement binary loses its whole premium, so a single entry can never
# risk 20% of equity: that loss would itself carry the account through the
# `MAX_PEAK_DRAWDOWN_PCT` floor. No stage here allows more than 12% of equity, and
# `openkal.live_exec.sized_contracts` clips that again to half the remaining
# distance to `0.80 * peak_equity`. At a fresh high-water mark the binding cap is
# therefore 10% of equity, and under a drawdown it shrinks with the buffer.
#
# Stage 1 is the sizing this build already ships: 10% of equity ($12.14 at
# $121.46) and the 50-contract study ceiling. The stages above it lift the contract
# ceiling so a validated lane can compound from $121.46 toward $1,000, $10,000 and
# $300,000 without stalling at a 250-contract wall, holding the risk fraction at
# 12% through stage 4 and tapering to 10% at stage 5 and 8% at stage 6. Nothing in
# any stage may lift `HARD_MAX_ENTRY_PRICE` or suspend the `MAX_PEAK_DRAWDOWN_PCT`
# trailing-peak breaker.
@dataclass(frozen=True)
class CompoundingStage:
    """One equity band and the risk it is allowed to put on a single entry."""

    name: str
    equity_floor: Decimal
    max_risk_pct: Decimal
    max_contracts: int


COMPOUND_STAGES: tuple[CompoundingStage, ...] = (
    CompoundingStage("stage1", Decimal("0.00"), Decimal("0.10"), MAX_CONTRACTS),
    CompoundingStage("stage2", Decimal("175.00"), Decimal("0.12"), 120),
    CompoundingStage("stage3", Decimal("350.00"), Decimal("0.12"), 300),
    CompoundingStage("stage4", Decimal("1000.00"), Decimal("0.12"), 1_000),
    CompoundingStage("stage5", Decimal("5000.00"), Decimal("0.10"), 2_500),
    CompoundingStage("stage6", Decimal("25000.00"), Decimal("0.08"), 5_000),
)

# The two invariants that outrank every stage. $0.62 is the cost ceiling the
# forward study scored (`openkal.sim_core.CAP`); a binary carried into settlement
# loses its whole premium, so the entry price *is* the tail bound. 20% off the
# equity high-water mark stops new risk rather than compounding a drawdown.
HARD_MAX_ENTRY_PRICE = Decimal("0.62")
MAX_PEAK_DRAWDOWN_PCT = Decimal("0.20")


def stage_for(equity: Decimal | float | str) -> CompoundingStage:
    """The compounding stage an account equity sits in (highest floor reached)."""
    value = Decimal(str(equity))
    stage = COMPOUND_STAGES[0]
    for candidate in COMPOUND_STAGES:
        if value >= candidate.equity_floor:
            stage = candidate
    return stage


def staged_risk_cap(equity: Decimal | float | str) -> Decimal:
    """Dollars of premium one entry may risk at this account equity.

    Rounded *down* to the cent: the cap is a ceiling, and a rounding step must
    never hand the lane a dollar more than the stage allows.
    """
    try:
        value = max(Decimal("0"), Decimal(str(equity)))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return (value * stage_for(value).max_risk_pct).quantize(Decimal("0.01"),
                                                            rounding=ROUND_DOWN)


def peak_drawdown_tripped(equity: Decimal | float | str | None,
                          peak_equity: Decimal | float | str | None) -> bool:
    """True once equity is `MAX_PEAK_DRAWDOWN_PCT` below its high-water mark."""
    try:
        current = Decimal(str(equity))
        peak = Decimal(str(peak_equity))
    except (InvalidOperation, TypeError, ValueError):
        return False
    if peak <= 0:
        return False
    return current <= peak * (Decimal("1") - MAX_PEAK_DRAWDOWN_PCT)


# --- rolling 48h regime gate ------------------------------------------------
# A 2-3 day edge is real but short: waiting for a 20-trade, month-long sample
# means trading it after it is gone. `scripts/forward_oos.py --regime-mode`
# scores a rolling 48h slice with the same frozen rule and the same execution
# model (T+1s entry, >= 50-contract print-through, $0.62 cap, settlement exits),
# and a bar that a smaller sample can clear. Nothing about the simulator, the
# sizing or the entry cap changes; only the evidence window and the counts it
# has to show.
REGIME_WINDOW_HOURS = 48
REGIME_MIN_TRADES = 12
REGIME_MIN_PROFIT_FACTOR = Decimal("1.25")
REGIME_MAX_ECE = Decimal("0.10")
REGIME_MIN_PROFITABLE_LANES = 2


# --- entry re-quote cooldown ------------------------------------------------
# One new entry quote per lane per 90 seconds. The per-window caps count entries,
# not time: a lane whose order is pulled unfilled (thesis gone, book moved) could
# re-quote on the next tick, and four lanes at 5Hz turned that into a stream of
# quotes nobody's study measured. 90s is the cooldown the frozen research scripts
# already use when they pick events (`pick(..., cooldown=90.0)`), so the live gate
# and the scored rule agree on what "the same trade" means.
REQUOTE_COOLDOWN_S = 90.0


def entry_quote_allowed(last_ts: float | None, now: float,
                        cooldown_s: float = REQUOTE_COOLDOWN_S) -> bool:
    """Whether a lane may issue a new entry quote at `now`.

    `last_ts` is when the lane last issued one (0.0 = never). The comparison is
    inclusive: a quote exactly `cooldown_s` old is allowed, one a tick younger is
    not. A clock that steps backwards (NTP) refuses rather than allowing, because
    a negative gap is not evidence that the cooldown elapsed.
    """
    try:
        last = float(last_ts or 0.0)
        current = float(now)
    except (TypeError, ValueError):
        return False
    if last <= 0.0:
        return True
    return 0.0 <= current - last >= cooldown_s


# --- live rolling-decay breaker ---------------------------------------------
# The forward gate is scored before arming; afterwards the epoch's own closed
# trades are the only evidence that the edge is still there. A rule that stops
# paying must hand the account back to paper without waiting for the 20%
# trailing balance floor: ten trades is a full sample for a 15-minute binary, and
# 12% is the drawdown a decayed edge produces before it is obvious.
DECAY_WINDOW_TRADES = 10
DECAY_MIN_PROFIT_FACTOR = Decimal("1.05")
DECAY_MAX_DRAWDOWN_PCT = Decimal("0.12")


def rolling_decay_breaker(pnls: Iterable[Decimal | float | str],
                          equity: Decimal | float | str, *,
                          window: int = DECAY_WINDOW_TRADES,
                          min_profit_factor: Decimal = DECAY_MIN_PROFIT_FACTOR,
                          max_drawdown_pct: Decimal = DECAY_MAX_DRAWDOWN_PCT) -> str | None:
    """The decay trip, as a pure function of the epoch's last closed trades.

    `pnls` is in chronological order and `equity` is the account after the last
    of them. Two independent ways to fail, one reason string: a profit factor
    below `min_profit_factor` (gross wins over gross losses) says the payoff has
    gone, and a peak-to-trough fall of more than `max_drawdown_pct` inside the
    window says the edge is costing more than the account can carry.

    Fewer than `window` trades is not a trip. A short sample cannot show decay,
    and a breaker that fired on the first two losers would disarm every lane
    before it had a chance to work. Returns None when the window is short or
    healthy.
    """
    try:
        trades = [Decimal(str(p)) for p in list(pnls)[-window:]]
        current = Decimal(str(equity))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if len(trades) < window or current <= 0:
        return None
    wins = sum((p for p in trades if p > 0), Decimal("0"))
    losses = -sum((p for p in trades if p < 0), Decimal("0"))
    pf = Decimal("Infinity") if losses == 0 else wins / losses
    # Walk the window forward from the equity it started with, so the drawdown is
    # measured against the running peak inside the window -- the same reading as
    # the trailing-peak breaker, just on ten trades instead of the epoch.
    running = peak = current - sum(trades, Decimal("0"))
    worst = Decimal("0")
    for pnl in trades:
        running += pnl
        peak = max(peak, running)
        if peak > 0:
            worst = max(worst, (peak - running) / peak)
    reasons: list[str] = []
    if pf < min_profit_factor:
        shown = "inf" if pf == Decimal("Infinity") else f"{pf:.2f}"
        reasons.append(f"last {window} trades PF {shown} < {min_profit_factor}")
    if worst > max_drawdown_pct:
        reasons.append(f"last {window} trades drawdown {worst * 100:.1f}% "
                       f"> {max_drawdown_pct * 100:.0f}%")
    return "; ".join(reasons) or None


def load_env(path: Path | None = None) -> dict[str, str]:
    """Read KEY=value lines. Values never leave the process; nothing is logged here."""
    path = path or ROOT / ".env"
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


@dataclass(frozen=True)
class SeriesConfig:
    """One orthogonal slot: a Kalshi 15-minute series and the spot feed that leads it."""

    series: str
    asset: str
    feed_symbol: str | None
    display: str

    @property
    def has_spot(self) -> bool:
        return self.feed_symbol is not None


# Kalshi's 15-minute crypto families. `feed_symbol` is a Coinbase product that
# tracks the same underlying the contract settles on (BRTI constituents), so the
# fair-value model and the venue are looking at the same asset.
SLOTS: tuple[SeriesConfig, ...] = (
    SeriesConfig("KXBTC15M", "BTC", "BTC-USD", "Bitcoin"),
    SeriesConfig("KXETH15M", "ETH", "ETH-USD", "Ethereum"),
    SeriesConfig("KXSOL15M", "SOL", "SOL-USD", "Solana"),
    SeriesConfig("KXXRP15M", "XRP", "XRP-USD", "XRP"),
    # KXDOGE15M is not a lane: four lanes at a quarter of equity each keeps every
    # dollar allocated, and DOGE's book was the thinnest of the five.
    # KXGOLD15M is deliberately not a lane: PAXG-USD is not the contract's settlement
    # reference, and pricing it against PAXG read 1.00 against a 0.40 book. A lane that
    # cannot validate its own input stays off rather than quoting a wrong number.
)

API_PREFIX = "/trade-api/v2"

VENUES = {
    "prod": "https://api.elections.kalshi.com",
    "demo": "https://demo-api.kalshi.co",
}


@dataclass
class Settings:
    env_name: str = "prod"
    allow_live: bool = False
    paper_equity: Decimal = Decimal("1000.00")
    # Live comes online only after the functional gate in `readiness.evaluate` passes:
    # an OOS-verified strategy, 8/8 controls, a healthy WS plane and a live canary.
    # There is no wall-clock soak any more — waiting was never evidence, and a paused
    # clock could satisfy it. OPENKAL_ARM_LIVE is permission to arm, not a bypass.
    arm_live: bool = False

    # Slot sizing: 100% of the slot's own bankroll per entry, never the whole account.
    slot_fraction: Decimal = Decimal("1.00")
    # Correlation guard: crypto lanes share one beta, so a single BTC shock must not
    # be able to hit every slot at once. At most two lanes may hold risk together,
    # and at most one of each direction.
    max_concurrent_positions: int = 4
    max_concurrent_crypto: int = 2
    max_same_direction_crypto: int = 2

    # Polling (seconds). Four series at 3s is ~13 read tokens/s, well under Basic.
    market_poll_s: float = 3.0
    tick_store_s: float = 1.0

    # Data quality gates.
    max_quote_age_s: float = 4.0
    max_spot_age_s: float = 5.0

    key_path: Path = ROOT / "keys" / "kalshi-private.pem"
    api_key_id: str = ""
    db_path: Path = ROOT / "data" / "openkal.db"
    dashboard_dir: Path = ROOT / "dashboard" / "dist"
    host: str = "127.0.0.1"
    port: int = 8787

    slots: tuple[SeriesConfig, ...] = SLOTS
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return VENUES.get(self.env_name, VENUES["demo"])

    @property
    def live(self) -> bool:
        """Whether the operator asked for live execution.

        Nothing in this build places a venue order, so the engine still reports paper
        fills. Mounting a broker in `Engine._execute` is the deliberate last step.
        """
        return self.allow_live and self.env_name == "prod"


def load_settings() -> Settings:
    env = load_env()
    env_name = (env.get("KALSHI_ENV") or "demo").lower()
    # A hard paper lock that outranks every other flag. `OPENKAL_ARM_LIVE=1` is
    # permission to arm *if the readiness gate passes*; this is the separate,
    # unambiguous instruction that no order may be sent this run, used by soak and
    # rehearsal runs so an armed .env can never put money at risk during them.
    force_paper = (env.get("OPENKAL_FORCE_PAPER") or os.environ.get("OPENKAL_FORCE_PAPER")
                   or "0").lower() in {"1", "true", "yes"}
    allow_live = ((env.get("ALLOW_LIVE") or "false").lower() in {"1", "true", "yes"}
                  and not force_paper)
    key_path = Path(env.get("KALSHI_PRIVATE_KEY_PATH") or ROOT / "keys" / "kalshi-private.pem")
    if not key_path.is_absolute():
        key_path = ROOT / key_path
    return Settings(
        env_name=env_name,
        allow_live=allow_live,
        paper_equity=Decimal(env.get("OPENKAL_PAPER_EQUITY") or os.environ.get("OPENKAL_PAPER_EQUITY") or "1000.00"),
        arm_live=((env.get("OPENKAL_ARM_LIVE") or os.environ.get("OPENKAL_ARM_LIVE") or "0").lower()
                  in {"1", "true", "yes"}) and not force_paper,
        api_key_id=env.get("KALSHI_API_KEY_ID") or env.get("KALSHI_KEY_ID", ""),
        key_path=key_path,
        db_path=Path(env.get("OPENKAL_DB") or ROOT / "data" / "openkal.db"),
    )
