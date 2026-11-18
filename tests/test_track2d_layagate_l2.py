import hashlib
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import track2c_true_l2_settle as C


class Track2D(unittest.TestCase):
    def test_latest_gate_is_same_ticker_causal_and_missing_verdict_resets_it(self):
        from scripts import track2d_layagate_l2 as D

        gates = D.load_gates({
            "tick_ts": np.array([8., 9., 10., 11., 12., 13.]),
            "tick_tid": np.array([0, 1, 0, 0, 1, 0]),
            "gate_available": np.array([1, 1, 0, 1, 1, 1]),
            "gate_allow": np.array([1, 1, 1, 0, 1, 1]),
            "gate_confidence": np.array([.1, .9, .9, .55, .9, .1], dtype=np.float32),
        })
        available, allow, confidence = D.gate_at(gates[0], np.array([7., 9., 10., 12.]))
        np.testing.assert_array_equal(available, [False, True, False, True])
        np.testing.assert_array_equal(allow, [False, True, True, False])
        np.testing.assert_array_equal(confidence >= .55, [False, False, True, True])

    def test_requires_joint_available_gate_raw_obi_price_band_and_own_side_edge(self):
        from scripts import track2d_layagate_l2 as D

        n = 10
        pack = {
            "tid": np.arange(n), "t": np.arange(n, dtype=float),
            "l2_obi": np.array([.20, -.20, .8, .19, -.8, np.nan, .8, .8, .8, -.8]),
            "p_yes": np.array([.65, .35, .65, .65, .7, .65, .45, .65, .65, .35]),
            "p_no": np.array([.35, .65, .35, .35, .3, .35, .55, .35, .35, .65]),
            "cost_yes_taker": np.array([.60, .55, .45, .45, .45, .45, .45, .61, .34, .60]),
            "cost_no_taker": np.array([.55, .35, .55, .55, .55, .55, .55, .55, .55, .40]),
        }
        available = np.array([True, True, False] + [True] * 7)
        allow = np.array([True, False] + [True] * 7 + [False])
        confidence = np.array([np.nan, .55, .99] + [.1] * 6 + [.549])
        entries = D.find_entries(pack, available, allow, confidence)
        self.assertEqual([(e[0], e[1]) for e in entries], [(0, "YES"), (1, "NO")])

    def test_joint_coverage_uses_live_gate_and_true_depth_at_same_decision(self):
        from scripts import track2d_layagate_l2 as D

        lane = "KXBTC15M"
        with tempfile.TemporaryDirectory() as temp, patch.object(C.RC, "LANES", (lane,)):
            root = Path(temp)
            names = ["one", "two", "three", "four"]
            closes = [1000, 1900, 2800, 3700]
            (root / "markets.json").write_text(json.dumps(
                {name: {"close": close, "strike": 100, "y": 1} for name, close in zip(names, closes)}))
            times = np.array(closes) - 780.
            np.savez(root / f"book_{lane}.npz",
                     tickers=names, ts=times, tid=np.arange(4),
                     yes_bid_sz=[3., 3., 3., 3.], yes_ask_sz=[2., 2., 2., 2.],
                     microprice=[.5, .5, .5, np.nan], tick_tid=np.arange(4),
                     tick_ts=times + [0., 3., 0., 0.], gate_available=[1, 1, 0, 1],
                     gate_allow=[0, 1, 1, 1], gate_confidence=[.1, .9, .9, .9])
            self.assertEqual(D.joint_coverage(root, set(closes)), {lane: [1000]})

    def test_canonical_execution_uses_arrival_price_and_binary_settlement(self):
        from scripts import track2d_layagate_l2 as D

        lane = SimpleNamespace(
            lane="KXBTC15M", tickers=["winner", "loser", "too_expensive"],
            meta={"winner": {"close": 900, "y": 1}, "loser": {"close": 900, "y": 0},
                  "too_expensive": {"close": 900, "y": 1}},
            ts=np.tile([100., 101.], 3), bid=np.tile([.39, .39], 3),
            ask=np.array([.40, .60, .40, .35, .40, .61]),
            book_bounds=[0, 2, 4, 6], trade_bounds=[0, 0, 0, 0],
            trade_ts=np.array([]), trade_px=np.array([]), trade_sz=np.array([]),
            trade_ty=np.array([], dtype=int),
        )
        lane.bid[3] = .34
        pack = dict(lane=lane, split="is", tid=np.arange(3), t=np.full(3, 100.),
                    secs=np.full(3, 800.), close=np.full(3, 900.),
                    l2_obi=np.full(3, .2), p_yes=np.full(3, .7), p_no=np.full(3, .3),
                    cost_yes_taker=np.full(3, .4), cost_no_taker=np.full(3, .61))
        rows, skips = D.execute(pack, np.ones(3, bool), np.ones(3, bool), np.full(3, np.nan))
        self.assertEqual([(r["ticker"], r["order_ts"], r["exit_ts"], Decimal(r["entry"]),
                           Decimal(r["exit_value"]), Decimal(r["fee_exit"]), r["reason"]) for r in rows],
                         [("winner", 101., 900., Decimal(".60"), 1, 0, "settle"),
                          ("loser", 101., 900., Decimal(".35"), 0, 0, "settle")])
        self.assertEqual(dict(skips), {"cost_band": 1})

    def test_track2c_receipt_exists_before_follow_on_and_both_use_identical_pin(self):
        from scripts import track2d_layagate_l2 as D

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pin = root / "pin"
            audit = {"windows_with_l2": 24}
            manifest = {"markets.json": hashlib.sha256(b"frozen bytes").hexdigest()}
            c_result = {"total_l2_windows": 24, "completed_at": "now",
                        "frozen_input_sha256": manifest}
            receipt = root / "c.json"

            def freeze(path):
                (path / "markets.json").write_text("frozen bytes")
                return manifest

            def follow_on(cache, c_obj):
                self.assertEqual(cache, pin)
                self.assertEqual(json.loads(receipt.read_text()), c_obj)
                self.assertEqual((cache / "markets.json").read_text(), "frozen bytes")
                return {"status": "IS fail; OOS unspent"}

            with patch.dict("os.environ", {"RND_OOS_UNSEAL": "lead", "RND_OOS_RESCORE": ""}), \
                    patch.object(C, "LEDGER", root / "ledger.jsonl"), \
                    patch.object(C, "PINNED_CACHE", pin), patch.object(C, "OUT", receipt), \
                    patch.object(C, "snapshot_cache", side_effect=freeze) as snapshot, \
                    patch.object(C, "audit_l2_depth", return_value=audit), \
                    patch.object(C, "run_full_l2_sweep", return_value=c_result) as sweep, \
                    patch.object(C, "update_handoff"), patch.object(D, "run_pinned", side_effect=follow_on):
                self.assertEqual(C.run_pinned_tracks(), (c_result, {"status": "IS fail; OOS unspent"}))
                C.run_pinned_tracks()
                snapshot.assert_called_once()
                sweep.assert_called_once_with(audit, pin, manifest)
                (pin / "markets.json").write_text("changed")
                with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                    C.run_pinned_tracks()

    def test_full_pair_seals_each_family_once_and_publishes_both_to_a_private_remote(self):
        from scripts import track2d_layagate_l2 as D

        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            source = data / "source"
            source.mkdir()
            remote = root / "remote.git"
            private = data / "rnd.git"

            def git(*args):
                return subprocess.check_output(["git", f"--git-dir={private}", f"--work-tree={root}", *args],
                                               text=True, cwd=root, stderr=subprocess.PIPE).strip()

            git("init", "-b", C.BRANCH)
            subprocess.check_output(["git", "init", "-q", "--bare", str(remote)])
            git("remote", "add", "origin", str(remote))
            git("config", "user.name", "Track Research Fixture")
            git("config", "user.email", "fixture@example.invalid")
            git("config", "commit.gpgsign", "false")

            lanes = ("KXBTC15M", "KXETH15M")
            closes = [10_000 + i * 900 for i in range(24)]
            markets = {}
            rng = np.random.default_rng(19)
            n = len(closes)
            for name in lanes:
                tickers = [f"{name}-{i:02d}" for i in range(n)]
                markets.update({ticker: {"close": close, "strike": 100, "y": int(i % 5 != 0)}
                                for i, (ticker, close) in enumerate(zip(tickers, closes))})
                t = np.concatenate([np.arange(close - 800., close - 58.) for close in closes])
                ids = np.repeat(np.arange(n), 742)
                np.savez(source / f"book_{name}.npz", tickers=tickers, ts=t, tid=ids,
                         bid=np.full(len(t), .39), ask=np.full(len(t), .40),
                         yes_bid_sz=np.tile([3., 9., 5., 7., 6., 4., 8.], len(t) // 7),
                         yes_ask_sz=np.full(len(t), 2.), microprice=np.full(len(t), .395),
                         trade_ts=[], trade_px=[], trade_sz=[], trade_ty=[], trade_tid=[],
                         tick_ts=t, tick_tid=ids, gate_available=np.ones(len(t), int),
                         gate_allow=np.zeros(len(t), int), gate_confidence=np.full(len(t), .55, dtype=np.float32))
                its = np.arange(closes[0] - 1200., closes[-1])
                value = 105. + np.cumsum(rng.normal(0, .003, len(its)))
                np.savez(source / f"index_{C.AD.INDEX_BY_SERIES[name]}.npz", ts=its, value=value)
            (source / "markets.json").write_text(json.dumps(markets))
            (root / "NEXT.json").write_text('{"rnd_overnight_shift":{"track2c_true_l2_loop":{}}}\n')
            (root / "RND_REPORT.md").write_text("# Fixture\n")
            git("add", "--", "NEXT.json", "RND_REPORT.md")
            git("commit", "-m", "fixture")
            git("push", "origin", C.BRANCH)

            for module, names in (
                (C, ["OUT", "TRADES", "OOS_TRADES", "LEDGER", "RND_REPORT", "NEXT_JSON"]),
                (D, ["OUT", "IS_TRADES", "OOS_TRADES"]),
            ):
                old_root = module.ROOT
                for name in names:
                    stack.enter_context(patch.object(module, name, root / getattr(module, name).relative_to(old_root)))
                stack.enter_context(patch.object(module, "ROOT", root))
            for module, name, value in (
                (C, "PRIVATE_GIT", private), (C, "PINNED_CACHE", data / "pinned"),
                (C, "EVOLVE_CACHE", source), (C.RC, "LEDGER", C.LEDGER), (C.RC, "LANES", lanes),
                (C.RC, "CACHE", C.RC.CACHE), (C.AD, "CACHE", C.AD.CACHE),
            ):
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.dict("os.environ", {"RND_OOS_UNSEAL": "lead", "RND_OOS_RESCORE": ""}))

            eth_book = source / "book_KXETH15M.npz"
            complete = eth_book.read_bytes()
            eth_book.write_bytes(complete[:100])
            self.assertIsNone(C.run_pinned_tracks())
            self.assertFalse(C.PINNED_CACHE.exists())
            eth_book.write_bytes(complete)
            with np.load(eth_book, allow_pickle=False) as archive:
                uneven = dict(archive)
            uneven["yes_ask_sz"] = uneven["yes_ask_sz"][:-1]
            np.savez(eth_book, **uneven)
            self.assertIsNone(C.run_pinned_tracks())
            self.assertFalse(C.PINNED_CACHE.exists())
            eth_book.write_bytes(complete)

            atomic_text = C.atomic_text

            def interrupted(path, text):
                if path == D.OUT:
                    raise OSError("simulated receipt write failure after ledger commit")
                atomic_text(path, text)

            with patch.object(C, "atomic_text", side_effect=interrupted):
                with self.assertRaisesRegex(OSError, "after ledger commit"):
                    C.run_pinned_tracks()
            self.assertFalse(D.OUT.exists())
            ledger = C.LEDGER.read_text()
            self.assertIn('"status": "scored"', ledger.splitlines()[-1])
            hook = remote / "hooks/pre-receive"
            hook.write_text("#!/bin/sh\nexit 1\n")
            hook.chmod(0o755)
            with self.assertRaises(subprocess.CalledProcessError):
                C.publish_results(json.loads(C.OUT.read_text()))
            c_only = git("rev-parse", "HEAD")
            c, d = C.run_pinned_tracks()
            self.assertEqual(d["frozen_input_sha256"], c["frozen_input_sha256"])
            self.assertEqual(d["fit_params"], c["fit_params"])
            self.assertEqual((d["joint_windows"], d["is_windows"], d["oos_windows"], d["cut"]),
                             (24, 14, 10, closes[14]))
            self.assertEqual((d["is"]["trades"], d["oos_scored"]["oos"]["trades"]), (56, 40))
            oos = [json.loads(line) for line in D.OOS_TRADES.read_text().splitlines()]
            self.assertTrue(all(r["close"] >= closes[14] and r["order_ts"] == r["trigger_ts"] + 1
                                and r["gate_available"] and not r["gate_allow"]
                                and abs(r["l2_obi"]) >= .20 for r in oos))
            self.assertEqual(C.LEDGER.read_text(), ledger)
            self.assertEqual([(r["family"], r["status"]) for r in map(json.loads, ledger.splitlines())],
                             [(C.FAMILY, "started"), (C.FAMILY, "scored"),
                              (D.FAMILY, "started"), (D.FAMILY, "scored")])
            C.LEDGER.write_text(ledger + '{"family":')
            with self.assertRaisesRegex(RuntimeError, "malformed OOS ledger"):
                C.publish_results(c, d)
            C.LEDGER.write_text(ledger)
            with patch.object(C, "time", SimpleNamespace(sleep=lambda _: hook.unlink())):
                C.publish_with_retry(c, d)
            self.assertEqual(git("rev-parse", "HEAD^"), c_only)
            self.assertEqual(git("ls-remote", "origin", f"refs/heads/{C.BRANCH}").split()[0], git("rev-parse", "HEAD"))
            _, retry = C.run_pinned_tracks()
            self.assertEqual(retry, d)
            self.assertEqual(C.LEDGER.read_text(), ledger)
            self.assertEqual(C.RND_REPORT.read_text().count("### Track 2D"), 1)
            state = json.loads(C.NEXT_JSON.read_text())["rnd_overnight_shift"]["track2d_layagate_l2"]
            self.assertTrue(state["oos_spent"])
            self.assertEqual(state["oos"]["trades"], 40)
            published = git("rev-parse", "HEAD")
            (root / "unrelated.txt").write_text("unrelated")
            git("add", "--", "unrelated.txt")
            git("commit", "-m", "unrelated")
            with self.assertRaisesRegex(RuntimeError, "unpublished commit"):
                C.publish_results(c, d)
            self.assertEqual(git("ls-remote", "origin", f"refs/heads/{C.BRANCH}").split()[0], published)


if __name__ == "__main__":
    unittest.main()
