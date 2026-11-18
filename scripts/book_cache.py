"""Build the sub-second book + print cache used by the research stack.

The 1 Hz `ticks` table is a decision log, not a fill tape: when the websocket book
loses its ask side the engine used to fall back to the REST snapshot, which can be
40 seconds stale. Every honest measurement therefore has to read `ws_book` (the
venue's own ladder, every message) and `ws_trade` (its print tape) directly.

Each `book_<LANE>.npz` carries, per lane:

    tickers, ts, bid, ask, tid            top of book, as recorded
    yes_bid_sz, yes_ask_sz, microprice    true resting size at the touch
    depth_bid_px/sz, depth_ask_px/sz      the top DEPTH_LEVELS of each side, YES terms
    trade_*                               the print tape
    tick_ts/bid/ask/tid                   the 1 Hz decision log
    gate_allow/confidence/latency_ms/     Laya's shadow verdict on that tick:
    available/note                        -1/NaN where none was recorded

Rows recorded before sizes and ladders were persisted carry NaN there: the cache
says what the tape holds instead of inventing a size. Every write goes beside the
live file and is renamed into place, so a rebuild killed halfway (a supervisor
restart, a full disk) can never leave a truncated book that the existence check
then skips forever.

    .venv/bin/python scripts/book_cache.py [--force] [LANE ...]
"""
from __future__ import annotations

import fcntl
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.micro_study import LANES, MARKETS, connect  # noqa: E402

CACHE = ROOT / "data" / "evolve-cache"
# One writer at a time on the live cache. `daemon_247.py` and the overnight
# supervisor both run this script against the same directory, and both write
# `book_<lane>.tmp.npz` before renaming it into place: a second writer truncating
# that temp file mid-save is the one race that can publish a corrupt book. A
# proof run into `--out` is private, so only the live directory is locked.
LOCK = ROOT / "data" / ".book-cache.lock"
DEPTH_LEVELS = 3
GATE_NOTE_CHARS = 48


def _f(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _depth(raw) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One recorded ladder row -> (bid_px, bid_sz, ask_px, ask_sz), NaN padded."""
    arrays = tuple(np.full(DEPTH_LEVELS, np.nan, dtype=np.float32) for _ in range(4))
    if not raw:
        return arrays
    try:
        ladder = json.loads(raw)
    except (TypeError, ValueError):
        return arrays
    for key, price_out, size_out in (("b", arrays[0], arrays[1]), ("a", arrays[2], arrays[3])):
        for i, level in enumerate((ladder.get(key) or [])[:DEPTH_LEVELS]):
            try:
                price_out[i], size_out[i] = float(level[0]), float(level[1])
            except (TypeError, ValueError, IndexError):
                break
    return arrays


def _gate(raw) -> tuple[int, float, float, int, str]:
    """(allow, confidence, latency_ms, available, note) from a recorded verdict.

    `allow` is 1/0 when Laya actually voted and -1 when the tick carries no verdict,
    so an ablation can tell an abstention from a missing answer.
    """
    if not raw:
        return -1, float("nan"), float("nan"), -1, ""
    try:
        verdict = json.loads(raw)
    except (TypeError, ValueError):
        return -1, float("nan"), float("nan"), -1, ""
    if not isinstance(verdict, dict):
        return -1, float("nan"), float("nan"), -1, ""
    allow = verdict.get("allow")
    return (1 if allow is True else 0 if allow is False else -1,
            _f(verdict.get("confidence")), _f(verdict.get("latency_ms")),
            1 if verdict.get("available") else 0,
            str(verdict.get("note") or "")[:GATE_NOTE_CHARS])


def build(only: list[str] | None = None, force: bool = False, limit: int = 0,
          out: Path | None = None) -> None:
    # `out` exists so a bounded proof build can run the same SQL and the same atomic
    # rename without replacing the production cache with a ten-ticker file.
    cache = out or CACHE
    cache.mkdir(parents=True, exist_ok=True)
    db = connect()
    markets = {}
    for m in json.loads(MARKETS.read_text()):
        if m.get("result") not in ("yes", "no"):
            continue
        markets[m["ticker"]] = m
    # The size/depth columns only exist once the recorder has persisted them; a tape
    # written before that still exports, with those fields NaN.
    cols = {r[1] for r in db.execute("PRAGMA table_info(ws_book)")}
    sized = {"yes_bid_sz", "yes_ask_sz"} <= cols
    has_depth = "depth" in cols
    select = "SELECT ts,yes_bid,yes_ask"
    if sized:
        select += ",yes_bid_sz,yes_ask_sz"
    if has_depth:
        select += ",depth"
    select += " FROM ws_book WHERE ticker=? ORDER BY ts"
    tickers = [r[0] for r in db.execute("SELECT DISTINCT ticker FROM ticks")]
    for lane in LANES:
        if only and lane not in only:
            continue
        if not force and (cache / f"book_{lane}.npz").exists():
            print(f"{lane}: cache exists, skipping", flush=True)
            continue
        names = sorted(t for t in tickers if t.startswith(lane + "-") and t in markets)
        if not names:
            # A lane with no settled market on the tape has nothing to export. This
            # used to reach np.concatenate([]) and kill the whole build, so the three
            # healthy lanes behind it were never written.
            print(f"{lane}: no settled tickers in the tape, skipping", flush=True)
            continue
        if limit:
            # A bounded build for proving the export path on a 26 GB tape: the SQL,
            # the atomic rename and the digests are identical, only the ticker count
            # is smaller. The scheduled production export passes no limit.
            names = names[:limit]
        rows_ts: list[np.ndarray] = []
        rows_bid: list[np.ndarray] = []
        rows_ask: list[np.ndarray] = []
        rows_tid: list[np.ndarray] = []
        rows_bid_sz: list[np.ndarray] = []
        rows_ask_sz: list[np.ndarray] = []
        rows_micro: list[np.ndarray] = []
        rows_dep: tuple[list[np.ndarray], ...] = ([], [], [], [])
        tr_ts: list[np.ndarray] = []
        tr_px: list[np.ndarray] = []
        tr_sz: list[np.ndarray] = []
        tr_ty: list[np.ndarray] = []
        tr_tid: list[np.ndarray] = []
        for i, name in enumerate(names):
            book = db.execute(select, (name,)).fetchall()
            # A ticker's rows are the peak of this loop: hand them to numpy and let
            # the row objects go before the next ticker is pulled.
            gc.collect()
            if not book:
                continue
            keys = book[0].keys()
            bid = np.array([_f(r["yes_bid"]) for r in book], dtype=np.float32)
            ask = np.array([_f(r["yes_ask"]) for r in book], dtype=np.float32)
            bid_sz = (np.array([_f(r["yes_bid_sz"]) for r in book], dtype=np.float32)
                      if "yes_bid_sz" in keys else np.full(len(book), np.nan, np.float32))
            ask_sz = (np.array([_f(r["yes_ask_sz"]) for r in book], dtype=np.float32)
                      if "yes_ask_sz" in keys else np.full(len(book), np.nan, np.float32))
            with np.errstate(invalid="ignore", divide="ignore"):
                total = bid_sz + ask_sz
                micro = np.where(total > 0, (bid * ask_sz + ask * bid_sz) / total,
                                 np.nan).astype(np.float32)
            rows_ts.append(np.array([r["ts"] for r in book], dtype=np.float64))
            rows_bid.append(bid)
            rows_ask.append(ask)
            rows_tid.append(np.full(len(book), i, dtype=np.int16))
            rows_bid_sz.append(bid_sz)
            rows_ask_sz.append(ask_sz)
            rows_micro.append(micro)
            if has_depth:
                parsed = [_depth(r["depth"]) for r in book]
                for side, block in enumerate(rows_dep):
                    block.append(np.vstack([p[side] for p in parsed]))
            else:
                for block in rows_dep:
                    block.append(np.full((len(book), DEPTH_LEVELS), np.nan, dtype=np.float32))
            trades = db.execute(
                "SELECT ts,yes_price,count,taker_side FROM ws_trade WHERE ticker=? ORDER BY ts",
                (name,)).fetchall()
            if trades:
                tr_ts.append(np.array([r["ts"] for r in trades], dtype=np.float64))
                tr_px.append(np.array([_f(r["yes_price"]) for r in trades], dtype=np.float32))
                tr_sz.append(np.array([_f(r["count"]) for r in trades], dtype=np.float32))
                tr_ty.append(np.array([1 if r["taker_side"] == "yes" else 0 for r in trades],
                                      dtype=np.int8))
                tr_tid.append(np.full(len(trades), i, dtype=np.int16))
            if i % 10 == 0:
                print(f"{lane}: {i}/{len(names)} tickers", flush=True)
        if not rows_ts:
            # Settled markets exist but none has a book row on this tape (a bounded
            # proof run can land on exactly such a slice). Skipping the lane keeps the
            # lanes behind it exporting instead of dying on np.concatenate([]).
            print(f"{lane}: {len(names)} settled ticker(s), no book rows, skipping",
                  flush=True)
            continue
        # Laya's shadow verdict lives on the 1 Hz decision log, not on the book, so
        # it is exported on its own time base with the ticker id it judged.
        tick_rows = db.execute(
            "SELECT ts,ticker,yes_bid,yes_ask,gate FROM ticks WHERE series=? ORDER BY ts",
            (lane,)).fetchall()
        tid_of = {name: i for i, name in enumerate(names)}
        gates = [_gate(r["gate"]) for r in tick_rows]
        tick_ts = np.array([r["ts"] for r in tick_rows], dtype=np.float64)
        tick_bid = np.array([_f(r["yes_bid"]) for r in tick_rows], dtype=np.float32)
        tick_ask = np.array([_f(r["yes_ask"]) for r in tick_rows], dtype=np.float32)
        tick_tid = np.array([tid_of.get(r["ticker"], -1) for r in tick_rows], dtype=np.int16)
        gate_allow = np.array([g[0] for g in gates], dtype=np.int8)
        gate_conf = np.array([g[1] for g in gates], dtype=np.float32)
        gate_lat = np.array([g[2] for g in gates], dtype=np.float32)
        gate_avail = np.array([g[3] for g in gates], dtype=np.int8)
        gate_note = np.array([g[4] for g in gates], dtype=f"<U{GATE_NOTE_CHARS}")
        # Write beside the live file and rename into place. A save that is killed
        # halfway (a supervisor restart, a full disk) used to leave a truncated
        # book_<lane>.npz that the existence check then skipped forever, and the
        # rebuild that moved the old file aside first left no tape at all.
        final = cache / f"book_{lane}.npz"
        tmp = cache / f"book_{lane}.tmp.npz"
        np.savez_compressed(
            tmp,
            tickers=np.array(names),
            ts=np.concatenate(rows_ts), bid=np.concatenate(rows_bid),
            ask=np.concatenate(rows_ask), tid=np.concatenate(rows_tid),
            yes_bid_sz=np.concatenate(rows_bid_sz),
            yes_ask_sz=np.concatenate(rows_ask_sz),
            microprice=np.concatenate(rows_micro),
            depth_bid_px=np.concatenate(rows_dep[0]),
            depth_bid_sz=np.concatenate(rows_dep[1]),
            depth_ask_px=np.concatenate(rows_dep[2]),
            depth_ask_sz=np.concatenate(rows_dep[3]),
            trade_ts=np.concatenate(tr_ts), trade_px=np.concatenate(tr_px),
            trade_sz=np.concatenate(tr_sz), trade_ty=np.concatenate(tr_ty),
            trade_tid=np.concatenate(tr_tid),
            tick_ts=tick_ts, tick_bid=tick_bid, tick_ask=tick_ask, tick_tid=tick_tid,
            gate_allow=gate_allow, gate_confidence=gate_conf, gate_latency_ms=gate_lat,
            gate_available=gate_avail, gate_note=gate_note,
        )
        os.replace(tmp, final)
        print(f"{lane}: {len(names)} tickers, {sum(len(x) for x in rows_ts):,} book rows, "
              f"{int(np.isfinite(np.concatenate(rows_bid_sz)).sum()):,} with L2 size, "
              f"{sum(len(x) for x in tr_ts):,} prints, "
              f"{len(tick_rows):,} ticks ({int((gate_allow >= 0).sum()):,} gate verdicts)",
              flush=True)
        del rows_ts, rows_bid, rows_ask, rows_tid, rows_bid_sz, rows_ask_sz, rows_micro
        del rows_dep, tr_ts, tr_px, tr_sz, tr_ty, tr_tid
        gc.collect()
    db.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    lanes: list[str] = []
    force = False
    limit = 0
    out: Path | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--force":
            force = True
        elif arg == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 1
        elif arg == "--out" and i + 1 < len(args):
            out = Path(args[i + 1])
            i += 1
        elif arg.startswith("--out="):
            out = Path(arg.split("=", 1)[1])
        elif arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif not arg.startswith("--"):
            lanes.append(arg)
        i += 1
    # Hold the lock for the whole build, not just the write: the multi-hour read
    # must not overlap another export either. A second invocation exits cleanly so
    # the daemon's rebuild and the supervisor's export never stack, and neither
    # treats the skip as a failure.
    target = CACHE if out is None else Path(out)
    lock_file = None
    if target.resolve() == CACHE.resolve():
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(LOCK, "a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("cache export already active, skipping", flush=True)
            lock_file.close()
            return 0
    try:
        build(lanes or None, force=force, limit=limit, out=out)
    finally:
        if lock_file is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
