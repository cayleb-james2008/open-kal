"""Deterministic tests for the 24/7 supervisor's git sync, drawdown floor, the 18:00
freeze, and the mc_drawdown floor-breach model.

Isolation: git_sync runs only against a throwaway repo with a bare local remote;
the real origin, the engine, the venue and the breaker are never reached. The
module-level log/state paths are redirected into a temp directory.

Run: .venv/bin/python -m unittest tests/test_daemon_247.py -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import daemon_247 as D  # noqa: E402
from scripts import day_supervisor as DS  # noqa: E402
from scripts import mc_drawdown as MC  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, check=True)
    return proc.stdout


class GitSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="daemon-sync-"))
        self.bare = self.tmp / "origin.git"
        self.repo = self.tmp / "work"
        subprocess.run(["git", "init", "-q", "--bare", "-b", "master", str(self.bare)], check=True)
        self.repo.mkdir()
        _git(self.repo, "init", "-q", "-b", "master")
        _git(self.repo, "config", "user.email", "test@example.invalid")
        _git(self.repo, "config", "user.name", "test")
        _git(self.repo, "remote", "add", "origin", str(self.bare))
        # Mirror the real allowlist: git add fails outright if any GIT_PATHS entry is missing.
        for rel in D.GIT_PATHS:
            self._write(rel + ("/seed.txt" if rel in ("openkal", "scripts", "tests", "dashboard", "docs") else ""), "seed\n")
        for rel in ("scripts/a.py", "openkal/x.py", "data/evening-report.md"):
            self._write(rel, "seed\n")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "seed")
        _git(self.repo, "push", "-q", "origin", "master")

        self._patches = [
            mock.patch.object(D, "ROOT", self.repo),
            mock.patch.object(DS, "LOG", self.tmp / "sync.log"),
            mock.patch.object(D, "LOG", self.tmp / "sync.log"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel: str, text: str | None = None, size: int | None = None) -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if size is not None:
            path.write_bytes(b"0" * size)
        else:
            path.write_text(text or "")

    def _head(self) -> str:
        return _git(self.repo, "rev-parse", "HEAD").strip()

    def _remote_head(self) -> str:
        return _git(self.bare, "rev-parse", "master").strip()

    def _staged(self) -> list[str]:
        return [p for p in _git(self.repo, "diff", "--cached", "--name-only").splitlines() if p]

    def test_allowlisted_change_is_committed_and_pushed_others_are_not(self) -> None:
        self._write("scripts/a.py", "changed\n")
        self._write("misc/outside.txt", "not on the allowlist\n")
        before = self._head()
        D._git_sync_work("test-clean")
        self.assertNotEqual(self._head(), before, "allowlisted change should be committed")
        self.assertEqual(self._remote_head(), self._head(), "commit should reach origin")
        committed = _git(self.repo, "show", "--name-only", "--format=", "HEAD").split()
        self.assertEqual(committed, ["scripts/a.py"])
        self.assertTrue((self.repo / "misc" / "outside.txt").exists(), "untouched on disk")

    def test_database_under_openkal_is_refused_and_index_reset(self) -> None:
        self._write("openkal/state.db", "sqlite-ish\n")
        before = self._head()
        D._git_sync_work("test-db")
        self.assertEqual(self._head(), before, "refused sync must not commit")
        self.assertEqual(self._staged(), [], "refusal resets the index")
        self.assertTrue((self.repo / "openkal" / "state.db").exists(), "working tree untouched")

    def test_oversized_file_over_5mb_is_refused(self) -> None:
        self._write("scripts/big.py", size=5_000_001)
        before = self._head()
        D._git_sync_work("test-big")
        self.assertEqual(self._head(), before)
        self.assertEqual(self._staged(), [])

    def test_file_at_exactly_5mb_is_not_refused_as_oversized(self) -> None:
        self._write("scripts/edge.py", size=5_000_000)
        before = self._head()
        D._git_sync_work("test-edge")
        self.assertNotEqual(self._head(), before, "5,000,000 bytes is at the limit, not over it")

    def test_keys_directory_inside_allowlisted_path_is_refused(self) -> None:
        self._write("scripts/keys/venue.pem", "-----BEGIN-----\n")
        before = self._head()
        D._git_sync_work("test-keys")
        self.assertEqual(self._head(), before)
        self.assertEqual(self._staged(), [])

    def test_root_dotenv_is_never_staged(self) -> None:
        self._write(".env", "KALSHI=redacted\n")
        before = self._head()
        D._git_sync_work("test-env-root")
        self.assertEqual(self._head(), before, ".env is outside GIT_PATHS, so nothing stages")

    def test_data_directory_other_than_report_is_refused(self) -> None:
        # Only data/evening-report.md is allowed under data/; anything else there is data.
        self._write("data/day-state.json", "{}\n")
        before = self._head()
        D._git_sync_work("test-data")
        self.assertEqual(self._head(), before)

    @unittest.expectedFailure
    def test_nested_dotenv_is_refused(self) -> None:
        # KNOWN GAP (reported, not fixed here): _forbidden matches only the root
        # ".env". A secrets file nested under an allowlisted path is pushed.
        self._write("scripts/.env", "KALSHI=redacted\n")
        before = self._head()
        D._git_sync_work("test-env-nested")
        self.assertEqual(self._head(), before, "a nested .env must be refused")


class ForbiddenPathTests(unittest.TestCase):
    def test_table(self) -> None:
        cases = {
            ".env": True,
            "keys/venue.pem": True,
            "scripts/keys/k.pem": True,
            "scripts/x.key": True,
            "data/anything.json": True,
            "data/evening-report.md": False,
            "openkal/prod.db": True,
            "openkal/prod.db-wal": True,
            "scripts/cache.npz": True,
            "scripts/__pycache__/m.cpython-314.pyc": True,
            "dashboard/node_modules/x.js": True,
            ".remember/now.md": True,
            "scripts/daemon_247.py": False,
            "openkal/strategy.py": False,
            "tests/test_daemon_247.py": False,
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(D._forbidden(path), expected)


class ForbiddenVenvTests(unittest.TestCase):
    @unittest.expectedFailure
    def test_root_venv_is_forbidden_in_the_check(self) -> None:
        # KNOWN GAP (reported, not fixed here): _forbidden matches "/.venv/" only
        # mid-path, so a root-level ".venv/..." passes the check. The allowlist keeps
        # .venv out of staging today, so this is defence in depth, not a live leak.
        self.assertTrue(D._forbidden(".venv/bin/python"))


class DrawdownFloorTests(unittest.TestCase):
    def test_floor_is_twenty_percent_below_peak_with_105_cap(self) -> None:
        twenty = Decimal("0.20")
        cases = [
            ("164.00", "131.20"),
            ("131.25", "105.00"),   # exact boundary: 0.8 * 131.25 = 105.00
            ("120.00", "105.00"),   # 0.8 * 120 = 96 is below the hard cap
            ("200.005", "160.00"),  # 160.004 rounds to cents
            ("0", "105.00"),
        ]
        for peak, expected in cases:
            with self.subTest(peak=peak):
                self.assertEqual(D.floor_for({"peak_balance": peak}, twenty), Decimal(expected))

    def test_missing_peak_defaults_to_the_cap(self) -> None:
        self.assertEqual(D.floor_for({}, Decimal("0.20")), Decimal("105.00"))

    def test_mc_float_floor_agrees_with_daemon_decimal_floor(self) -> None:
        self.assertEqual(MC.FLOOR_MIN, float(D.floor_for({}, Decimal("0.20"))))
        self.assertEqual(MC.TRAIL, 0.20)
        for cents in range(10_000, 30_001, 7):
            peak = cents / 100
            want = float(D.floor_for({"peak_balance": str(peak)}, Decimal("0.20")))
            got = max(MC.FLOOR_MIN, (1.0 - MC.TRAIL) * peak)
            self.assertLess(abs(want - got), 0.006, f"peak={peak}")


class EveningFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="evening-"))
        self.final = self.tmp / "evening-final.json"
        self.clock = {"t": datetime(2026, 11, 17, 17, 59)}
        self._patches = [
            mock.patch.object(D, "EVENING_FINAL", self.final),
            mock.patch.object(D, "now", lambda: self.clock["t"]),
            mock.patch.object(DS, "LOG", self.tmp / "log"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _snap(self, equity: float) -> dict:
        return D.afternoon_snapshot({"last_balance": "170.0", "floor": "131.2"},
                                    {"fills": []}, {"equity": equity, "realized": 6.0}, [])

    def test_before_1800_is_live_and_writes_nothing(self) -> None:
        snap = self._snap(170.0)
        self.assertFalse(snap["final"])
        self.assertFalse(self.final.exists())

    def test_at_1800_freezes_once_and_later_calls_return_the_same_numbers(self) -> None:
        self.clock["t"] = datetime(2026, 11, 17, 18, 0)
        first = self._snap(170.0)
        self.assertTrue(first["final"])
        self.assertTrue(self.final.exists())
        on_disk = json.loads(self.final.read_text())
        self.assertEqual(on_disk["equity"], 170.0)

        self.clock["t"] = datetime(2026, 11, 17, 18, 20)
        later = self._snap(999.0)            # a later window must not replace the 18:00 number
        self.assertEqual(later, first)
        self.assertEqual(json.loads(self.final.read_text()), on_disk)

    def test_other_dates_return_empty_when_nothing_is_frozen(self) -> None:
        self.clock["t"] = datetime(2026, 11, 18, 18, 30)
        self.assertEqual(self._snap(170.0), {})
        self.assertFalse(self.final.exists())


class McDrawdownTests(unittest.TestCase):
    PATHS = 2000
    SEED = 7

    def _breach(self, p_win: float, slots: int = 2, arrival: str = "observed") -> float:
        return MC.simulate(slots, p_win, 14, self.PATHS, self.SEED, arrival)["breach_prob"]

    def test_certain_wins_never_breach(self) -> None:
        self.assertEqual(self._breach(1.0), 0.0)

    def test_certain_losses_breach_on_the_observed_schedule(self) -> None:
        # 3 losses (-19.5) then 2 more (-13) breach 131.20 by the third window.
        self.assertEqual(self._breach(0.0), 1.0)

    def test_breach_probability_is_non_increasing_in_win_rate(self) -> None:
        grid = [0.0, 0.5, 0.8, 0.9, 0.944, 1.0]
        for arrival in ("observed", "independent", "simultaneous"):
            with self.subTest(arrival=arrival):
                probs = [self._breach(p, arrival=arrival) for p in grid]
                for lo, hi in zip(probs, probs[1:]):
                    self.assertGreaterEqual(lo + 1e-12, hi, probs)

    def test_observed_schedule_ignores_the_slot_cap(self) -> None:
        two = MC.simulate(2, 0.9, 14, self.PATHS, self.SEED, "observed")
        four = MC.simulate(4, 0.9, 14, self.PATHS, self.SEED, "observed")
        for key in ("breach_prob", "median_final", "p05_final", "median_max_dd", "median_trades"):
            self.assertEqual(two[key], four[key], key)


if __name__ == "__main__":
    unittest.main()
