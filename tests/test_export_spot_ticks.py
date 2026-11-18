"""The spot_ticks exporter only reads, takes a bounded snapshot, and writes one file atomically."""
from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import export_spot_ticks as X

SCHEMA = """CREATE TABLE spot_ticks (ts REAL NOT NULL, symbol TEXT NOT NULL, source TEXT NOT NULL, price REAL, bid REAL, ask REAL);
CREATE INDEX spot_ticks_symbol_ts ON spot_ticks(symbol, ts);"""
ROWS = [(103.0, "BTC-USD", "coinbase", 100.3, 100.2, 100.4), (101.0, "BTC-USD", "kraken", 100.1, None, None),
        (102.0, "ETH-USD", "coinbase", 10.2, None, None), (100.0, "BTC-USD", "coinbase", 100.0, 99.9, 100.1),
        (104.0, "XRP-USD", "rest", 0.5, None, None)]


def make_db(directory: Path, rows=ROWS) -> tuple[Path, sqlite3.Connection]:
    path = directory / "prod.db"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.executescript(SCHEMA)
    writer.executemany("INSERT INTO spot_ticks VALUES (?,?,?,?,?,?)", rows)
    writer.commit()
    return path, writer                                         # the writer stays open, as the live engine's does


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"


class ReadOnly(unittest.TestCase):
    def test_the_connection_cannot_write_and_will_not_create_a_missing_database(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d))
            conn = X.open_ro(db)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO spot_ticks VALUES (1,'X','coinbase',1,1,1)")
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM spot_ticks")
            conn.close()
            writer.close()
            with self.assertRaises(sqlite3.OperationalError):
                X.open_ro(Path(d) / "missing.db")
            self.assertFalse((Path(d) / "missing.db").exists())

    def test_exporting_a_live_wal_database_leaves_its_files_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d))
            wal = Path(f"{db}-wal")
            before = (digest(db), digest(wal))
            dest = Path(d) / "out"
            dest.mkdir()
            X.export(db, dest)
            self.assertEqual((digest(db), digest(wal)), before)
            self.assertEqual(writer.execute("SELECT count(*) FROM spot_ticks").fetchone()[0], len(ROWS))
            writer.close()


class Snapshot(unittest.TestCase):
    def export_to(self, d, **kw):
        db, writer = make_db(Path(d))
        dest = Path(d) / "out"
        dest.mkdir(exist_ok=True)
        info = X.export(db, dest, **kw)
        return info, np.load(dest / X.OUT_NAME), db, writer, dest

    def test_rows_are_sorted_by_time_with_nulls_as_nan_and_the_venue_label_kept(self):
        with tempfile.TemporaryDirectory() as d:
            info, z, _, writer, _ = self.export_to(d)
            self.assertEqual((info["rows"], info["rows_by_source"]), (5, {"coinbase": 3, "kraken": 1, "rest": 1}))
            np.testing.assert_array_equal(z["ts"], [100.0, 101.0, 102.0, 103.0, 104.0])
            self.assertEqual(list(z["source"]), ["coinbase", "kraken", "coinbase", "coinbase", "rest"])
            self.assertTrue(np.isnan(z["bid"][1]) and z["bid"][0] == 99.9)
            self.assertEqual((int(z["meta_max_rowid"][0]), info["ts_min"], info["ts_max"]), (5, 100.0, 104.0))
            writer.close()

    def test_rows_written_after_the_snapshot_bound_are_left_for_the_next_export(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d))
            dest = Path(d) / "out"
            dest.mkdir()

            def late(bound):
                writer.execute("INSERT INTO spot_ticks VALUES (999, 'BTC-USD', 'coinbase', 1, 1, 1)")
                writer.commit()

            info = X.export(db, dest, on_snapshot=late)
            self.assertEqual((info["rows"], info["max_rowid"]), (5, 5))
            self.assertEqual(X.export(db, dest)["rows"], 6)                                   # the next export picks it up
            writer.close()

    def test_small_chunks_give_exactly_the_same_copy_as_one_big_read(self):
        with tempfile.TemporaryDirectory() as d:
            _, small, db, writer, dest = self.export_to(d, chunk=2)
            small = {k: small[k].copy() for k in ("ts", "symbol", "source", "price", "bid", "ask")}
            X.export(db, dest, chunk=1000)
            big = np.load(dest / X.OUT_NAME)
            for k, v in small.items():
                np.testing.assert_array_equal(v, big[k], err_msg=k)
            writer.close()

    def test_an_empty_table_exports_empty_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d), rows=[])
            dest = Path(d) / "out"
            dest.mkdir()
            info = X.export(db, dest)
            self.assertEqual((info["rows"], info["ts_min"]), (0, None))
            self.assertEqual(len(np.load(dest / X.OUT_NAME)["ts"]), 0)
            writer.close()


class Output(unittest.TestCase):
    def test_exactly_one_new_file_is_written_and_other_files_are_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d))
            dest = Path(d) / "out"
            dest.mkdir()
            (dest / "book_KXBTC15M.npz").write_bytes(b"someone else's export")
            before = digest(dest / "book_KXBTC15M.npz")
            X.export(db, dest)
            self.assertEqual(sorted(p.name for p in dest.iterdir()), ["book_KXBTC15M.npz", X.OUT_NAME])
            self.assertEqual(digest(dest / "book_KXBTC15M.npz"), before)
            writer.close()

    def test_a_failed_write_keeps_the_previous_copy_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d))
            dest = Path(d) / "out"
            dest.mkdir()
            X.export(db, dest)
            good = digest(dest / X.OUT_NAME)

            def broken(path, **arrays):
                Path(path).write_bytes(b"half a file")
                raise OSError("disk full")

            with patch.object(X.np, "savez_compressed", side_effect=broken):
                with self.assertRaises(OSError):
                    X.export(db, dest)
            self.assertEqual(digest(dest / X.OUT_NAME), good)
            self.assertEqual([p.name for p in dest.iterdir()], [X.OUT_NAME])
            writer.close()

    def test_it_refuses_a_missing_destination_and_never_writes_over_the_database(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d))
            with self.assertRaisesRegex(RuntimeError, "not a directory"):
                X.export(db, Path(d) / "nope")
            twin = Path(d) / X.OUT_NAME                                   # a database that is named like the output
            twin.write_bytes(db.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "production database"):
                X.check_dest(twin, Path(d))
            writer.close()


if __name__ == "__main__":
    unittest.main()
