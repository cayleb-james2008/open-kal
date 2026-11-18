"""Shared plumbing for Track 4A/4B: the pinned tape, the cut, and the handoff files.

Both tracks read one frozen copy of the evolve cache (`data/track4-tape`, taken with
`track2c_true_l2_settle.snapshot_cache`, digests in pin.json) and never the live
directory, so an hourly cache rebuild cannot move the tape between the IS and OOS phases.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import alpha_discover as AD  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402
from scripts import track2c_true_l2_settle as C  # noqa: E402

TAPE = ROOT / "data" / "track4-tape"
MIN_WINDOWS = 217


def pin_tape(tape: Path = TAPE) -> dict:
    """Point every reader at a pinned tape after re-hashing it against its pin.json."""
    pin = json.loads((tape / "pin.json").read_text())
    if C.snapshot_digests(tape, list(pin["sha256"])) != pin["sha256"]:
        raise RuntimeError(f"{tape.name} digest mismatch; refusing to run on a changed tape")
    AD.CACHE = RC.CACHE = tape
    return pin


def refuse_if_spent(family: str, out: Path) -> None:
    """An IS run rewrites the receipt: never after the family has an OOS ledger line or a scored receipt."""
    if C.LEDGER.exists() and any(json.loads(line)["family"] == family
                                 for line in C.LEDGER.read_text().splitlines() if line.strip()):
        raise RuntimeError(f"{family} already has an OOS ledger entry; its IS receipt is frozen")
    if out.exists() and json.loads(out.read_text()).get("oos_scored") is not None:
        raise RuntimeError(f"{family} receipt already holds a scored OOS look; refusing to rewrite it")


def refuse_rescore() -> None:
    """The ledger's rescore override exists for the lead's manual use; these tracks never take a second look."""
    if os.environ.get("RND_OOS_RESCORE"):
        raise RuntimeError("RND_OOS_RESCORE is set: these tracks never rescore an OOS family")


def window_closes(tape: Path = TAPE) -> list[float]:
    """Distinct settled window closes across the four lanes, sorted."""
    meta = json.loads((tape / "markets.json").read_text())
    closes = set()
    for lane in RC.LANES:
        with np.load(tape / f"book_{lane}.npz", allow_pickle=False) as data:
            names = [str(t) for t in data["tickers"]]
        closes |= {float(meta[n]["close"]) for n in names
                   if n in meta and meta[n].get("y") in (0, 1) and meta[n].get("strike")}
    return sorted(closes)


def cut_of(closes: list[float]) -> float:
    """First OOS close: the same 60% rule as `rnd_common.global_cut`."""
    return closes[int(len(closes) * 0.6)]


def clean(obj):
    """A copy with non-finite floats as strings, so strict JSON (`allow_nan=False`) can write it."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def update_handoff(key: str, state: dict, header: str | None = None, lines: list[str] | None = None) -> None:
    """Merge `state` into NEXT.json; append a report section once when `header` is given."""
    nxt = json.loads(C.NEXT_JSON.read_text())
    nxt["rnd_overnight_shift"].setdefault(key, {}).update(clean(state))
    C.atomic_text(C.NEXT_JSON, json.dumps(nxt, indent=1, allow_nan=False) + "\n")
    if header is not None:
        report = C.RND_REPORT.read_text()
        if header not in report:
            C.atomic_text(C.RND_REPORT, report.rstrip() + "\n\n" + header + "\n" + "\n".join(lines) + "\n")


def scorecard(s: dict) -> str:
    pf = s["pf"] if s["trades"] else "n/a"
    pf = round(pf, 3) if isinstance(pf, float) else pf
    return (f"{s['trades']} trades, win {s['win_rate']:.3f}, net ${s['net']:+.2f}, PF {pf}, "
            f"{s['profitable_lanes']} profitable lanes, ex-XRP ${s['net_ex_xrp']:+.2f}, "
            f"max entry ${s['max_entry_cost']:.2f}, {s['crossed_fills']} crossed fills")


SHARED_TAPE_NOTE = ("- The 'sealed' OOS slice is sealed per family in the ledger only: earlier families on this tape "
                    "(Tracks 2B-2D, F2 variants) already looked at overlapping late windows.")
