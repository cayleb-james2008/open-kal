"""Kalshi fee geometry in exact Decimal arithmetic.

Kalshi charges `ceil(0.07 * contracts * P * (1 - P))` rounded up to the next
cent, and rebates maker orders on most series (including the 15-minute crypto
families). That asymmetry is the whole strategy: a taker round-trip at $0.50
costs ~1.75c a side and eats the edge, a maker round-trip costs $0.

Floats are never used here; a cent of drift is a cent of edge.
"""
from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

CENT = Decimal("0.01")
ZERO = Decimal("0")


def taker_fee(price: Decimal, contracts: int) -> Decimal:
    """Fee for a taker fill at `price` (0..1) for `contracts` contracts."""
    if contracts <= 0:
        return ZERO
    p = Decimal(price)
    raw = Decimal("0.07") * Decimal(contracts) * p * (Decimal(1) - p)
    return (raw / CENT).to_integral_value(rounding=ROUND_CEILING) * CENT


def maker_fee(price: Decimal, contracts: int) -> Decimal:
    """Resting maker fills are free on every series this bot is allowlisted for."""
    return ZERO


def round_trip_cost(price: Decimal, contracts: int, both_taker: bool = True) -> Decimal:
    fee = taker_fee if both_taker else maker_fee
    return fee(price, contracts) * (2 if both_taker else 1)


def breakeven_move(price: Decimal, contracts: int = 1) -> Decimal:
    """How far the price must move for a taker round-trip to break even, per contract."""
    if contracts <= 0:
        return ZERO
    return round_trip_cost(price, contracts, True) / Decimal(contracts)
