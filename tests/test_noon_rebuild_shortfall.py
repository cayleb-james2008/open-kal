#!/usr/bin/env python3
"""Prove a market-refresh shortfall is not swallowed by day_supervisor.

A temporary ROOT contains a stub rebuild_cache.sh that prints the same loud
shortfall line as scripts/refresh_markets_ready.py and exits 1. The real
day_supervisor.evaluate_forward is called with temp state/log/forward paths;
it must see rc=1, log the failure, and return None before any scorer runs.
No production database or forward OOS file is touched.
"""
from __future__ import annotations

import importlib.util
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "day_supervisor_shortfall", ROOT / "scripts" / "day_supervisor.py")
ds = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ds)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="openkal-shortfall-") as td:
        root = Path(td)
        (root / "scripts").mkdir()
        (root / "data").mkdir()
        stub = root / "scripts" / "rebuild_cache.sh"
        stub.write_text(
            "#!/bin/bash\n"
            "echo 'MARKET REFRESH SHORTFALL: latest closed 12:00 missing; "
            "refusing cache build' >&2\n"
            "exit 1\n"
        )
        stub.chmod(0o755)

        old_root, old_state = ds.ROOT, ds.STATE
        old_forward, old_log = ds.FORWARD_PATH, ds.LOG
        ds.ROOT = root
        ds.STATE = root / "data" / "day-state.json"
        ds.FORWARD_PATH = root / "data" / "forward-oos.json"
        ds.LOG = root / "data" / "day-supervisor.log"
        try:
            state: dict = {}
            slot = datetime.fromisoformat("2026-11-17T12:00:00-05:00")
            result = ds.evaluate_forward(1794909600, slot=slot, state=state)
            log_text = ds.LOG.read_text()
        finally:
            ds.ROOT, ds.STATE = old_root, old_state
            ds.FORWARD_PATH, ds.LOG = old_forward, old_log

        assert result is None, result
        assert state.get("slot_attempt", {}).get("slot") == slot.isoformat()
        assert "rebuild rc=1" in log_text, log_text
        assert "MARKET REFRESH SHORTFALL" in log_text, log_text
        assert "rebuild failed; this slot is spent without a score" in log_text, log_text
        print("PASS shortfall reached evaluate_forward: rc=1 logged, result=None, "
              "scorer not reached")
        print("PASS sandbox forward artifact exists:", (root / "data" / "forward-oos.json").exists())
        print("1/1 rebuild-shortfall propagation check passed")


if __name__ == "__main__":
    main()
