"""One place for every runtime knob: credentials paths, venue, slots and safety gates.

Nothing here decides to trade. The gates are read by the strategy and the engine;
`ALLOW_LIVE` is the only switch that can put real money at risk, and it stays off
unless the operator sets it deliberately.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Live sizing ceiling, set to the evidence. `scripts/forward_oos.py` scored the
# frozen rule with MAX_QTY = 50 contracts per trade, and the live lane must not
# buy more than the study covered: a 24c NO became 170 contracts ($40.80 of
# premium) against a 50-contract study, and a binary that settles to zero loses
# the whole premium, not the 7c stop distance.
MAX_CONTRACTS = 50
MAX_TRADE_RISK_USD = Decimal("15.00")


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
