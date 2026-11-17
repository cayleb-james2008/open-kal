#!/usr/bin/env python3
"""Temp-file tests for the noon settled-market readiness gate.

No production database, venue call, scorer, or forward OOS artifact is used.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "refresh_markets_ready", ROOT / "scripts" / "refresh_markets_ready.py")
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)

LANES = mod.DEFAULT_LANES
assert LANES == ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M"), LANES
TARGET = 1794934800  # 2026-11-17 12:00:00 EST = 17:00:00Z
CLOSE = datetime.fromtimestamp(TARGET, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
TICKER = "{lane}-26NOV171200-15"  # the actual settled ticker shape



def pass_success() -> None:
    with tempfile.TemporaryDirectory(prefix="openkal-settle-ok-") as td:
        path = Path(td) / "settled.json"
        path.write_text("[]")
        calls = []

        def refresh(remaining: float) -> tuple[int, str]:
            calls.append(remaining)
            data = [{"ticker": TICKER.format(lane=lane), "close_time": CLOSE}
                    for lane in LANES]
            path.write_text(json.dumps(data))
            return 0, "stub refreshed"

        ok, missing = mod.wait_for_latest(path, LANES, TARGET, timeout=1.0,
                                          poll=0.01, refresh=refresh)
        assert ok, missing
        assert not missing
        assert calls, "stub refresh was not called"
        print(f"PASS success: {CLOSE} present on all lanes after "
              f"{len(calls)} stub refresh")


def pass_shortfall() -> None:
    with tempfile.TemporaryDirectory(prefix="openkal-settle-missing-") as td:
        path = Path(td) / "settled.json"
        path.write_text("[]")
        calls = []

        def refresh(remaining: float) -> tuple[int, str]:
            calls.append(remaining)
            return 0, "stub returned no target"

        ok, missing = mod.wait_for_latest(path, LANES, TARGET, timeout=0.05,
                                          poll=0.005, refresh=refresh)
        assert not ok
        assert set(missing) == set(LANES), missing
        assert calls, "stub refresh was not called"
        print(f"PASS shortfall: {CLOSE} missing on all lanes; "
              f"function returns false after {len(calls)} stub refresh")


def cli_call(path: Path, *extra: str) -> tuple[int, str, str]:
    old_file, old_refresh, old_argv = mod.DEFAULT_FILE, mod.refresh_once, sys.argv
    mod.DEFAULT_FILE = path
    mod.refresh_once = lambda pages, timeout: (0, "stub returned no target")
    sys.argv = ["refresh_markets_ready.py", "--as-of", str(TARGET), *extra]
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = mod.main()
    finally:
        mod.DEFAULT_FILE, mod.refresh_once, sys.argv = old_file, old_refresh, old_argv
    return rc, out.getvalue(), err.getvalue()


def pass_cli_success() -> None:
    with tempfile.TemporaryDirectory(prefix="openkal-settle-cli-ok-") as td:
        path = Path(td) / "settled.json"
        data = [{"ticker": TICKER.format(lane=lane), "close_time": CLOSE}
                for lane in LANES]
        path.write_text(json.dumps(data))
        rc, out, err = cli_call(path, "--timeout", "1")
        assert rc == 0, (rc, out, err)
        assert "present on all lanes" in out
        print("PASS cli success: exit 0 when the target window is present")


def pass_cli_shortfall() -> None:
    with tempfile.TemporaryDirectory(prefix="openkal-settle-cli-bad-") as td:
        path = Path(td) / "settled.json"
        path.write_text("[]")
        rc, out, err = cli_call(path, "--timeout", "0.02", "--poll", "0.001")
        assert rc == 1, (rc, out, err)
        assert "MARKET REFRESH SHORTFALL" in err, err
        assert "refusing to rebuild data/evolve-cache" in err, err
        print("PASS cli shortfall: exit 1 and loud refusal when absent")


if __name__ == "__main__":
    pass_success()
    pass_shortfall()
    pass_cli_success()
    pass_cli_shortfall()
    print("4/4 settled-market readiness checks passed")
