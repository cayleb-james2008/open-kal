"""The spot_ticks exporter only reads, accumulates by rowid watermark, and writes one file atomically."""
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


def setup(d: Path, rows=ROWS):
    db, writer = make_db(d, rows)
    dest = d / "out"
    dest.mkdir()
    return db, writer, dest


class ReadOnly(unittest.TestCase):
    def test_the_connection_cannot_write_is_query_only_and_will_not_create_a_missing_database(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer = make_db(Path(d))
            conn = X.open_ro(db)
            self.assertEqual(conn.execute("PRAGMA query_only").fetchone()[0], 1)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO spot_ticks VALUES (1,'X','coinbase',1,1,1)")
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM spot_ticks")
            conn.close()
            writer.close()
            with self.assertRaises(sqlite3.OperationalError):
                X.open_ro(Path(d) / "missing.db")
            self.assertFalse((Path(d) / "missing.db").exists())

    def test_a_live_wal_database_keeps_its_main_file_and_wal_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            wal, shm = Path(f"{db}-wal"), Path(f"{db}-shm")
            self.assertTrue(wal.exists() and shm.exists())              # a running engine's database
            before = (digest(db), digest(wal))
            X.export(db, dest)
            self.assertEqual((digest(db), digest(wal)), before)         # the shm is reader coordination and may change
            self.assertTrue(wal.exists() and shm.exists())
            self.assertEqual(writer.execute("SELECT count(*) FROM spot_ticks").fetchone()[0], len(ROWS))
            writer.close()

    def test_a_database_without_wal_files_is_refused_and_nothing_is_created_beside_it(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            writer.close()                                              # the last connection closing removes -wal and -shm
            files = sorted(p.name for p in Path(d).iterdir())
            self.assertNotIn("prod.db-wal", files)
            with self.assertRaisesRegex(RuntimeError, "engine is stopped"):
                X.export(db, dest)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), files)
            self.assertEqual(list(dest.iterdir()), [])


class Snapshot(unittest.TestCase):
    def test_rows_are_sorted_by_time_with_nulls_as_nan_and_the_venue_label_kept(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            info = X.export(db, dest)
            z = np.load(dest / X.OUT_NAME)
            self.assertEqual((info["rows"], info["new_rows"], info["rows_by_source"]), (5, 5, {"coinbase": 3, "kraken": 1, "rest": 1}))
            np.testing.assert_array_equal(z["ts"], [100.0, 101.0, 102.0, 103.0, 104.0])
            self.assertEqual(list(z["source"]), ["coinbase", "kraken", "coinbase", "coinbase", "rest"])
            self.assertTrue(np.isnan(z["bid"][1]) and z["bid"][0] == 99.9)
            self.assertEqual((int(z["meta_max_rowid"][0]), info["ts_min"], info["ts_max"]), (5, 100.0, 104.0))
            writer.close()

    def test_long_symbols_are_not_truncated_and_nulls_become_empty_strings(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d), rows=[(1.0, "A-VERY-LONG-PRODUCT-NAME", "coinbase", 1.0, None, None)])
            writer.execute("DROP TABLE spot_ticks")
            writer.execute("CREATE TABLE spot_ticks (ts REAL, symbol TEXT, source TEXT, price REAL, bid REAL, ask REAL)")
            writer.executemany("INSERT INTO spot_ticks VALUES (?,?,?,?,?,?)", [(1.0, "A-VERY-LONG-PRODUCT-NAME", "coinbase", 1.0, None, None),
                                                                              (2.0, None, None, 2.0, None, None)])
            writer.commit()
            X.export(db, dest)
            z = np.load(dest / X.OUT_NAME)
            self.assertEqual((z["symbol"][0], z["symbol"][1], z["source"][1]), ("A-VERY-LONG-PRODUCT-NAME", "", ""))
            writer.close()

    def test_rows_written_after_the_snapshot_bound_are_left_for_the_next_export(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))

            def late(bound):
                writer.execute("INSERT INTO spot_ticks VALUES (999, 'BTC-USD', 'coinbase', 1, 1, 1)")
                writer.commit()

            info = X.export(db, dest, on_snapshot=late)
            self.assertEqual((info["rows"], info["max_rowid"]), (5, 5))
            self.assertEqual(X.export(db, dest)["new_rows"], 1)                              # the next export picks it up
            writer.close()

    def test_the_read_is_done_in_chunks_each_its_own_short_query(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            seen = []
            real = X.open_ro

            def spy(path):
                conn = real(path)
                conn.set_trace_callback(lambda sql: seen.append((sql, conn.in_transaction)))
                return conn

            with patch.object(X, "open_ro", spy):
                X.export(db, dest, chunk=2)
            chunks = [s for s, _ in seen if "ORDER BY rowid LIMIT" in s]
            self.assertEqual(len(chunks), 3)                                                 # 2 + 2 + 1 rows, never one big query
            self.assertTrue(all("LIMIT 2" in c for c in chunks))
            self.assertTrue(all(not in_tx for sql, in_tx in seen if "ORDER BY rowid LIMIT" in sql))     # no transaction left open
            writer.close()

    def test_small_chunks_give_exactly_the_same_copy_as_one_big_read(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            X.export(db, dest, chunk=2)
            small = {k: np.load(dest / X.OUT_NAME)[k] for k in X.FIELDS}
            (dest / X.OUT_NAME).unlink()
            X.export(db, dest, chunk=1000)
            big = np.load(dest / X.OUT_NAME)
            for k, v in small.items():
                np.testing.assert_array_equal(v, big[k], err_msg=k)
            writer.close()

    def test_an_empty_table_exports_empty_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d), rows=[])
            info = X.export(db, dest)
            self.assertEqual((info["rows"], info["ts_min"]), (0, None))
            self.assertEqual(len(np.load(dest / X.OUT_NAME)["ts"]), 0)
            writer.close()


class Accumulation(unittest.TestCase):
    def test_each_export_appends_only_the_new_rows_and_keeps_the_archive_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            self.assertEqual(X.export(db, dest)["new_rows"], 5)
            writer.executemany("INSERT INTO spot_ticks VALUES (?,?,?,?,?,?)", [(99.5, "BTC-USD", "coinbase", 5.0, None, None),
                                                                              (105.0, "BTC-USD", "coinbase", 6.0, None, None)])
            writer.commit()
            info = X.export(db, dest)
            self.assertEqual((info["rows"], info["new_rows"], info["max_rowid"]), (7, 2, 7))
            ts = np.load(dest / X.OUT_NAME)["ts"]
            np.testing.assert_array_equal(ts, np.sort(ts))
            before = digest(dest / X.OUT_NAME)
            again = X.export(db, dest)
            self.assertEqual((again["new_rows"], again["rows"], again["max_rowid"]), (0, 7, 7))      # nothing new: nothing appended
            self.assertEqual(digest(dest / X.OUT_NAME), before)                                       # and the file is not rewritten
            writer.close()

    def test_rows_the_engine_later_prunes_stay_in_the_archive(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            X.export(db, dest)
            writer.execute("DELETE FROM spot_ticks WHERE ts < 103")                          # Store.prune() at shutdown
            writer.execute("INSERT INTO spot_ticks VALUES (200, 'BTC-USD', 'coinbase', 9, NULL, NULL)")
            writer.commit()
            info = X.export(db, dest)
            self.assertEqual((info["rows"], info["new_rows"]), (6, 1))                       # 5 archived + 1 new, none lost
            self.assertEqual(float(np.load(dest / X.OUT_NAME)["ts"].min()), 100.0)
            writer.close()

    def test_a_rebuilt_table_is_refused_instead_of_merged(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            X.export(db, dest)
            before = digest(dest / X.OUT_NAME)
            writer.execute("DROP TABLE spot_ticks")
            writer.executescript(SCHEMA)
            writer.execute("INSERT INTO spot_ticks VALUES (300, 'BTC-USD', 'coinbase', 1, NULL, NULL)")
            writer.commit()
            with self.assertRaisesRegex(RuntimeError, "rebuilt"):
                X.export(db, dest)
            self.assertEqual(digest(dest / X.OUT_NAME), before)
            writer.close()

    def test_rows_that_vanish_during_the_export_fail_the_run_and_write_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))

            def prune(last):
                writer.execute("DELETE FROM spot_ticks WHERE rowid = 5")                     # a prune between chunks
                writer.commit()

            with self.assertRaisesRegex(RuntimeError, "vanished"):
                X.export(db, dest, chunk=2, on_chunk=prune)
            self.assertEqual(list(dest.iterdir()), [])
            writer.close()


class Output(unittest.TestCase):
    def test_exactly_one_new_file_is_written_and_other_files_are_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            (dest / "book_KXBTC15M.npz").write_bytes(b"someone else's export")
            before = digest(dest / "book_KXBTC15M.npz")
            X.export(db, dest)
            self.assertEqual(sorted(p.name for p in dest.iterdir()), ["book_KXBTC15M.npz", X.OUT_NAME])
            self.assertEqual(digest(dest / "book_KXBTC15M.npz"), before)
            writer.close()

    def test_the_file_is_synced_before_it_is_renamed_into_place(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            order = []
            fsync, replace = X.os.fsync, X.os.replace
            with patch.object(X.os, "fsync", lambda fd: (order.append("fsync"), fsync(fd))[1]), \
                    patch.object(X.os, "replace", lambda a, b: (order.append("replace"), replace(a, b))[1]):
                X.export(db, dest)
            self.assertEqual(order, ["fsync", "replace"])
            writer.close()

    def test_a_failed_write_keeps_the_previous_copy_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as d:
            db, writer, dest = setup(Path(d))
            X.export(db, dest)
            good = digest(dest / X.OUT_NAME)
            writer.execute("INSERT INTO spot_ticks VALUES (500, 'BTC-USD', 'coinbase', 1, NULL, NULL)")
            writer.commit()

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
                X.check_paths(twin, Path(d))
            writer.close()


if __name__ == "__main__":
    unittest.main()
