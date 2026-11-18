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
    parser.add_argument("--status", action="store_true", help="show the arm record, if any")
    parser.add_argument("--disarm", action="store_true", help="remove the arm record")
    args = parser.parse_args(argv)

    if args.status:
        record = arming.read()
        if record is None:
            print(f"no arm record at {arming.ARM_PATH.relative_to(ROOT)}: the engine stays paper")
            return 0
        print(f"arm record: {record.get('by')} at "
              f"{_stamp(float(record.get('ts') or 0.0))}"
              f" ({args.note or record.get('note') or 'no note'})")
        return 0

    if args.disarm:
        record = arming.consume()
        print("disarmed" if record is not None else "no arm record to remove")
        return 0

    try:
        record = arming.operator_arm(args.by, note=args.note)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(record, indent=2))
    print("Armed for this engine boot only. The engine arms on its next readiness pass; "
          "`--status` shows whether the record was consumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
