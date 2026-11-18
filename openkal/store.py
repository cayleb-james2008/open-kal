"""SQLite store: tick ring buffer, fills, trades, equity curve, program database.

The ledger and the program database are what the verifier reads. They are written
only by the engine, never by a candidate strategy (the plan's immutable-ledger rule).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    ts REAL NOT NULL, series TEXT NOT NULL, ticker TEXT NOT NULL,
    yes_bid TEXT, yes_ask TEXT, last TEXT, spot REAL, p_model REAL,
    edge REAL, secs_left REAL, spread REAL, signal TEXT, gate TEXT
);
CREATE INDEX IF NOT EXISTS ticks_series_ts ON ticks(series, ts);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, series TEXT, ticker TEXT,
    side TEXT, kind TEXT, price TEXT, contracts INTEGER, fee TEXT, note TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, series TEXT NOT NULL, ticker TEXT,
    side TEXT, opened_ts REAL, closed_ts REAL, entry TEXT, exit_price TEXT,
    contracts INTEGER, fee TEXT, pnl TEXT, exit_reason TEXT, p_model REAL,
    spot REAL, strike REAL
);
CREATE INDEX IF NOT EXISTS trades_series ON trades(series, closed_ts);

CREATE TABLE IF NOT EXISTS equity (
    ts REAL PRIMARY KEY, equity TEXT, cash TEXT, realized TEXT
);

CREATE TABLE IF NOT EXISTS programs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, slot TEXT, name TEXT,
    params TEXT, status TEXT, score REAL, profit_factor REAL, ece REAL,
    note TEXT, generation INTEGER
);

CREATE TABLE IF NOT EXISTS forensics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, series TEXT, reason TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);

CREATE TABLE IF NOT EXISTS ws_index (
    ts REAL, index_id TEXT, value TEXT, avg60 TEXT, final_minute TEXT,
    final_ticks INTEGER, channel TEXT
);
CREATE INDEX IF NOT EXISTS ws_index_ts ON ws_index(index_id, ts);

CREATE TABLE IF NOT EXISTS ws_book (
    ts REAL, ticker TEXT, yes_bid TEXT, yes_ask TEXT, seq INTEGER,
    yes_bid_sz TEXT, yes_ask_sz TEXT, depth TEXT
);
CREATE INDEX IF NOT EXISTS ws_book_ts ON ws_book(ticker, ts);

CREATE TABLE IF NOT EXISTS ws_trade (
    ts REAL, ticker TEXT, yes_price TEXT, count TEXT, taker_side TEXT, raw TEXT
);
CREATE INDEX IF NOT EXISTS ws_trade_ts ON ws_trade(ticker, ts);

CREATE TABLE IF NOT EXISTS ws_misc (ts REAL, kind TEXT, payload TEXT);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        self._db.row_factory = sqlite3.Row
        # WAL: the recorder, the engine and the verifier all read this file, and the
        # websocket writer must never block the hot loop on a reader.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)
            columns = {r[1] for r in self._db.execute("PRAGMA table_info(trades)")}
            if "source" not in columns:
                self._db.execute("ALTER TABLE trades ADD COLUMN source TEXT DEFAULT 'live'")
            self._db.commit()
        # After the schema exists: on a brand-new file there is no `ticks` table yet.
        self._migrate()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _migrate(self) -> None:
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(ticks)")}
        # The book ladder's resting sizes. Without them the tape says where the
        # touch was but not how much was actually there, so no study can tell a
        # one-lot quote from a wall. `depth` holds the top levels as JSON and is
        # written on snapshots only; the touch sizes are on every row.
        book_cols = {row[1] for row in self._db.execute("PRAGMA table_info(ws_book)")}
        for name in ("yes_bid_sz", "yes_ask_sz", "depth"):
            if name not in book_cols:
                self._db.execute(f"ALTER TABLE ws_book ADD COLUMN {name} TEXT")
        for name, decl in (("signal", "TEXT"), ("can_open", "INTEGER"), ("stake", "TEXT"),
                           # The decision's inputs, recorded with the tick. Replay needs
                           # them to re-derive the same decision instead of guessing at
                           # freshness and the calibration-adjusted hurdle.
                           ("spot_age", "REAL"), ("hurdle", "TEXT"),
                           ("impulse_bps", "REAL"),
                           # Laya's shadow verdict for this trigger, as JSON. Without it
                           # an ablation can only re-run the model with hindsight; with
                           # it, the verdict the live lane actually saw is on the tape.
                           ("gate", "TEXT")):
            if name not in cols:
                self._db.execute(f"ALTER TABLE ticks ADD COLUMN {name} {decl}")
        self._db.commit()

    def slot_stake(self, series: str) -> Decimal:
        """The bankroll the live loop was actually sizing with for this lane.

        Fallback only: the live loop records its per-lane bankroll on every tick, and
        `replay_series` prefers that tape figure. `epoch_equity` is the epoch's opening
        account, which is what a lane was sliced from before window rebalancing.
        """
        base = self.get_meta("epoch_equity")
        try:
            equity = Decimal(str(base)) if base is not None else Decimal("0")
        except Exception:  # noqa: BLE001
            equity = Decimal("0")
        lanes = max(1, len([c for c in __import__("openkal.config", fromlist=["SLOTS"]).SLOTS]))
        return (equity / lanes) if equity > 0 else Decimal("50")

    def tick(self, *, ts: float, series: str, ticker: str, yes_bid: str, yes_ask: str,
             last: str, spot: float, p_model: float, edge: float, secs_left: float,
             spread: float, signal: str = "", can_open: bool = False,
             stake: str = "", spot_age: float = 0.0, hurdle: str = "",
             impulse_bps: float = 0.0, gate: str = "") -> None:
        self._exec(
            "INSERT INTO ticks(ts,series,ticker,yes_bid,yes_ask,last,spot,p_model,edge,secs_left,"
            "spread,signal,can_open,stake,spot_age,hurdle,impulse_bps,gate)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, series, ticker, yes_bid, yes_ask, last, spot, p_model, edge, secs_left, spread,
             signal, 1 if can_open else 0, stake, spot_age, hurdle, impulse_bps, gate),
        )

    def set_tick_gate(self, *, series: str, gate: str, ticker: str | None = None) -> None:
        """Attach a gate verdict to the newest tick for this lane.

        A shadow review is fire-and-forget, so its answer lands just after the tick
        row it judged was written. Updating the newest row keeps the verdict on the
        trigger it belongs to rather than the next one, which is the difference
        between an honest re-score and one that sees the future.

        `ticker` pins it to the contract the verdict was about. A review that is
        still in flight when the window rolls would otherwise land on the *next*
        contract's tick row, and an ablation reading that row would score the
        gate on a trade it never saw.
        """
        if ticker:
            self._exec(
                "UPDATE ticks SET gate=? WHERE rowid=("
                "SELECT rowid FROM ticks WHERE series=? AND ticker=? "
                "ORDER BY ts DESC LIMIT 1)",
                (gate, series, ticker),
            )
            return
        self._exec(
            "UPDATE ticks SET gate=? WHERE rowid=("
            "SELECT rowid FROM ticks WHERE series=? ORDER BY ts DESC LIMIT 1)",
            (gate, series),
        )

    def printed_qty(self, *, ticker: str, t0: float, t1: float, price: Decimal,
                    side: str) -> Decimal:
        """Contracts the venue printed at or through a passive limit in (t0, t1].

        A maker fill is evidence, not a touch. This is the print rule
        `scripts/alpha_search.py` scores every passive candidate with: a resting YES
        bid at `price` is lifted by an aggressor *selling* YES at or below it, and a
        resting NO bid at `price` (a YES offer at `1 - price`) is lifted by an
        aggressor *buying* YES at or above `1 - price`.
        """
        if t1 <= t0:
            return Decimal("0")
        taker = "no" if side == "YES" else "yes"
        limit = price if side == "YES" else Decimal("1") - price
        op = "<=" if side == "YES" else ">="
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(CAST(count AS REAL)),0) FROM ws_trade"
                f" WHERE ticker=? AND ts>? AND ts<=? AND taker_side=?"
                f" AND CAST(yes_price AS REAL) {op} ?",
                (ticker, t0, t1, taker, float(limit)),
            ).fetchone()
        return Decimal(str(row[0] or 0))

    def venue_entries(self, since_ts: float) -> list[dict[str, Any]]:
        """Maker entries the venue itself reported this epoch, oldest first.

        In live execution the exchange -- not a simulator -- decides which resting bids
        filled. Observed on this account: five ETH/SOL/XRP bids rested through prints
        at their own limit and were never matched, because the queue ahead of them was
        deeper than the print. A replay cannot see queue position, so a live epoch's
        entry fills come from `/portfolio/fills`; everything after the fill (size,
        exit, fees, P&L) is still re-derived from the recorded quotes.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, ticker, side, price, contracts, note FROM fills"
                " WHERE kind='entry' AND ts>=? ORDER BY ts ASC", (since_ts,)
            ).fetchall()
        return [dict(r) for r in rows if (r["note"] or "") == "live fill"]

    def venue_exits(self, since_ts: float) -> list[dict[str, Any]]:
        """Venue-reported exit prints this epoch, oldest first.

        The live epoch's exits leave a venue paper trail just like its entries. The
        venue is the only authority on the price it actually took us out at: a
        crash-recovery flatten, a maker take-profit or a carry settlement recorded on
        the book is not something a tape replay can re-derive from the ladder, and the
        replay cannot guess it. Included here only for live epochs; a paper fixture
        has no venue truth and keeps deriving exits from its tape.

        `venue reconciliation flatten` / `venue reconciliation (settlement carry)` are
        excluded to mirror the crash-recovery rows the repro control excludes on the
        ledger side -- both sides of the parity comparison must use the same boundary.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT ts, ticker, side, price, contracts, fee, note FROM fills"
                " WHERE kind='exit' AND ts>=? AND (note IN"
                " ('settled','live fill','take-profit reached')"
                # A filled post-only scratch is a venue exit like the others; left out,
                # the replay re-derived it from the tape and booked a take-profit the
                # exchange never gave (replay +$2.70 vs ledger $0.00, 18:55 BTC scratch).
                " OR note LIKE 'time barrier%')"
                " ORDER BY ts ASC", (since_ts,)
            ).fetchall()
        return [dict(r) for r in rows]

    def fill(self, *, ts: float, series: str, ticker: str, side: str, kind: str,
             price: str, contracts: int, fee: str, note: str = "") -> None:
        self._exec(
            "INSERT INTO fills(ts,series,ticker,side,kind,price,contracts,fee,note)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (ts, series, ticker, side, kind, price, contracts, fee, note),
        )

    def trade(self, **row: Any) -> None:
        self._exec(
            "INSERT INTO trades(series,ticker,side,opened_ts,closed_ts,entry,exit_price,contracts,fee,pnl,exit_reason,p_model,spot,strike,source)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (row["series"], row["ticker"], row["side"], row["opened_ts"], row["closed_ts"],
             str(row["entry"]), str(row["exit_price"]), row["contracts"], str(row["fee"]),
             str(row["pnl"]), row["exit_reason"], row["p_model"], row.get("spot"),
             row.get("strike"), row.get("source", "live")),
        )

    def equity(self, ts: float, equity: str, cash: str, realized: str) -> None:
        self._exec("INSERT OR REPLACE INTO equity(ts,equity,cash,realized) VALUES(?,?,?,?)",
                   (ts, equity, cash, realized))

    def recent_ticks(self, series: str, since_ts: float, limit: int = 2000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM ticks WHERE series=? AND ts>=? ORDER BY ts ASC LIMIT ?",
                (series, since_ts, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def epoch_stats(self, since_ts: float, series: str | None = None,
                    source: str | None = None) -> dict[str, Any]:
        """Ledger totals for one window, so parity compares like with like.

        `source="live"` restricts to venue-confirmed trades. A live epoch may have
        simulated paper fills written by the engine before the readiness gate flipped
        ARM_LIVE; those are not this epoch's program behaviour and the replay cannot
        reproduce them.
        """
        sql = ("SELECT COUNT(*) AS trades, COALESCE(SUM(CAST(pnl AS REAL)), 0) AS pnl"
               " FROM trades WHERE closed_ts >= ?")
        args: tuple = (since_ts,)
        if series:
            sql += " AND series=?"
            args += (series,)
        if source:
            sql += " AND source = ?"
            args += (source,)
        row = self._db.execute(sql, args).fetchone()
        return {"trades": int(row["trades"]), "pnl": row["pnl"]}

    def ws_batch(self, rows: list[tuple]) -> None:
        """Batched persistence for the websocket plane, outside its receive loop."""
        books, indices, trades, misc = [], [], [], []
        for row in rows:
            kind = row[0]
            if kind == "book":
                # Older producers send only the five top-of-book fields; they are
                # written with the sizes NULL rather than rejected.
                books.append((row[1], row[2], row[3], row[4], row[5],
                              row[6] if len(row) > 6 else None,
                              row[7] if len(row) > 7 else None,
                              row[8] if len(row) > 8 else None))
            elif kind == "index":
                indices.append((row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
            elif kind == "trade":
                trades.append((row[1], row[2], row[3], row[4], row[5], row[6]))
            elif kind == "indexlist":
                misc.append((row[1], kind, row[2]))
        with self._lock:
            if books:
                self._db.executemany(
                    "INSERT INTO ws_book(ts,ticker,yes_bid,yes_ask,seq,yes_bid_sz,"
                    "yes_ask_sz,depth) VALUES(?,?,?,?,?,?,?,?)", books)
            if indices:
                self._db.executemany(
                    "INSERT INTO ws_index(ts,index_id,value,avg60,final_minute,final_ticks,channel)"
                    " VALUES(?,?,?,?,?,?,?)", indices)
            if trades:
                self._db.executemany(
                    "INSERT INTO ws_trade(ts,ticker,yes_price,count,taker_side,raw)"
                    " VALUES(?,?,?,?,?,?)", trades)
            if misc:
                self._db.executemany("INSERT INTO ws_misc(ts,kind,payload) VALUES(?,?,?)", misc)
            self._db.commit()

    def recent_trades(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM trades ORDER BY closed_ts DESC LIMIT ?",
                                    (limit,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self, series: str | None = None) -> dict[str, Any]:
        sql = "SELECT COUNT(*) n, COALESCE(SUM(CAST(pnl AS REAL)),0) pnl FROM trades"
        args: tuple = ()
        if series:
            sql += " WHERE series=?"
            args = (series,)
        with self._lock:
            row = self._db.execute(sql, args).fetchone()
        return {"trades": int(row["n"] or 0), "pnl": float(row["pnl"] or 0.0)}

    def calibration_pairs(self, limit: int = 400) -> list[tuple[float, int]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT p_model, pnl FROM trades ORDER BY closed_ts DESC LIMIT ?", (limit,)
            ).fetchall()
        pairs: list[tuple[float, int]] = []
        for r in rows:
            if r["p_model"] is None:
                continue
            won = 1 if float(r["pnl"]) > 0 else 0
            pairs.append((float(r["p_model"]), won))
        return pairs

    def equity_curve(self, limit: int = 600) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT ts,equity FROM equity ORDER BY ts DESC LIMIT ?",
                                    (limit,)).fetchall()
        return [{"ts": r["ts"], "equity": float(r["equity"])} for r in reversed(rows)]

    def add_program(self, *, ts: float, slot: str, name: str, params: dict[str, str],
                    status: str, score: float, profit_factor: float, ece: float,
                    note: str, generation: int) -> int:
        cur = self._exec(
            "INSERT INTO programs(ts,slot,name,params,status,score,profit_factor,ece,note,generation)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (ts, slot, name, json.dumps(params), status, score, profit_factor, ece, note, generation),
        )
        return int(cur.lastrowid or 0)

    def programs(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM programs ORDER BY ts DESC LIMIT ?",
                                    (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["params"] = json.loads(d["params"] or "{}")
            out.append(d)
        return out

    def forensic(self, *, ts: float, series: str, reason: str, payload: dict[str, Any]) -> None:
        self._exec("INSERT INTO forensics(ts,series,reason,payload) VALUES(?,?,?,?)",
                   (ts, series, reason, json.dumps(payload, default=str)))

    def last_forensic(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM forensics ORDER BY ts DESC LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(row)
        d["payload"] = json.loads(d["payload"] or "{}")
        return d

    def set_meta(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO meta(k,v) VALUES(?,?)", (key, json.dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._db.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return json.loads(row["v"]) if row else default

    def prune(self, keep_s: float = 86400 * 3) -> None:
        cutoff = time.time() - keep_s
        self._exec("DELETE FROM ticks WHERE ts<?", (cutoff,))
