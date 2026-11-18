"""The indicator registry the genome is built from.

PROVISIONAL (version `provisional-1`): the Decision Agent's indicator list was not available, so this is the engine's own feature
set computed on a one-second grid. Replacing it means bumping VERSION; every candidate records the version it was built with.
`signed` features can be the source of a follow/fade side; every feature can sit in a threshold condition.
"""
from __future__ import annotations

VERSION = "provisional-1"

# name -> (signed, description)
FEATURES: dict[str, tuple[bool, str]] = {
    "z":          (True,  "log(spot/strike) / (trailing sigma * sqrt(seconds left)); the favourite's direction is its sign"),
    "p_up":       (False, "normal-approximation probability that spot settles above strike, from z"),
    "delta_bps":  (True,  "(spot - strike) / spot in basis points"),
    "flow10":     (True,  "signed aggressive print flow over 10 s, in [-1, 1] (+ = takers buying YES)"),
    "flow30":     (True,  "signed aggressive print flow over 30 s"),
    "rev5":       (True,  "touch-revision imbalance over 5 s, bid upticks minus ask downticks, in [-1, 1]"),
    "shift5":     (True,  "mid change over the last 5 s, in dollars"),
    "mom10":      (True,  "spot return over 10 s, bps"),
    "mom30":      (True,  "spot return over 30 s, bps"),
    "btc_mom30":  (True,  "BTC spot return over 30 s, bps (the lead for the other coins)"),
    "btc_z":      (True,  "BTC's z at the same second"),
    "spread":     (False, "yes ask - yes bid, dollars"),
    "mid":        (False, "(yes bid + yes ask) / 2"),
    "l2imb":      (True,  "(bid size - ask size) / (bid size + ask size) at the touch; NaN before the touch sizes were recorded"),
    "log_touch":  (False, "log of the smaller touch size; NaN before the sizes were recorded"),
    "sigma_bps":  (False, "trailing per-sqrt-second vol of the settlement index, in bps"),
    "quote_age":  (False, "seconds since the book row in force was written"),
    "secs":       (False, "seconds left in the window"),
    "hod":        (False, "hour of day, fractional, local time"),
}
NAMES = list(FEATURES)
SIGNED = [n for n, (s, _) in FEATURES.items() if s]
