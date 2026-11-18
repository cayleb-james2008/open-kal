#!/usr/bin/env python3
"""Arm live execution for the running engine -- a person's act, recorded.

The engine will not send a venue order because `.env` says so. It requires
`data/live-arm.json`, and this script is the only thing that writes one. The record
names who armed it and when; the engine accepts it only if it was written after
this boot started, deletes it on the first successful arm, and discards any record
left over from a previous boot. A verdict, a promotion, a restart and an
environment variable therefore cannot arm anything.

    .venv/bin/python scripts/arm_live.py --by "Cayleb" --note "Track 9 canary"
    .venv/bin/python scripts/arm_live.py --status
    .venv/bin/python scripts/arm_live.py --disarm
    .venv/bin/python scripts/arm_live.py --lane gas-no --by "Cayleb"

The crypto engine reads `data/live-arm.json`; `--lane` writes that lane's own
record (for example `data/live-arm-gas-no.json`) and never touches the crypto
file. Arming one lane cannot arm another.

Arming is still not trading: the readiness gate must also pass, and it stays red
until the strategy verdict, the controls, the WS plane and the live canary are all
green.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import arming  # noqa: E402


def _stamp(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arm_live", description=__doc__.splitlines()[0])
    parser.add_argument("--by", default=os.environ.get("OPENKAL_OPERATOR")
                        or getpass.getuser(),
                        help="the person arming live execution (recorded verbatim)")
    parser.add_argument("--note", default="", help="why, in a few words")
    parser.add_argument("--lane", default=None,
                        help="arm a separate lane (e.g. gas-no); writes its own record")
    parser.add_argument("--status", action="store_true", help="show the arm record, if any")
    parser.add_argument("--disarm", action="store_true", help="remove the arm record")
    args = parser.parse_args(argv)

    path = arming.arm_path(args.lane)
    if args.status:
        record = arming.read(path)
        if record is None:
            print(f"no arm record at {path.relative_to(ROOT)}: the engine stays paper")
            return 0
        print(f"arm record: {record.get('by')} at "
              f"{_stamp(float(record.get('ts') or 0.0))}"
              f" ({args.note or record.get('note') or 'no note'})"
              f" lane {record.get('lane') or 'crypto'}")
        return 0

    if args.disarm:
        record = arming.consume(path)
        print("disarmed" if record is not None else "no arm record to remove")
        return 0

    try:
        record = arming.operator_arm(args.by, note=args.note, lane=args.lane)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2))
    if args.lane:
        print(f"Armed lane {args.lane} only, at {path.relative_to(ROOT)}. "
              "The crypto record is untouched.")
    else:
        print("Armed for this engine boot only. The engine arms on its next readiness "
              "pass; `--status` shows whether the record was consumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
