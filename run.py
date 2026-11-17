#!/usr/bin/env python3
"""`python run.py` starts the Open-Kal engine and its dashboard.

The engine needs the project virtualenv. On this box `/usr/bin/python` carries
`websockets` 10.4, whose `connect()` has no `additional_headers` argument: every
authenticated connect raised a `TypeError` that the reconnect loop swallowed, so
`KalshiWS` sat at 0 messages / 0 reconnects under `python run.py` while
`.venv/bin/python scripts/soak.py` had it live. Re-exec into `.venv` before any
Open-Kal import so both invocations are the same run.
"""
import os
import sys
from pathlib import Path

_VENV = Path(__file__).resolve().parent / ".venv" / "bin" / "python"

if (os.name != "nt" and _VENV.exists()
        and Path(sys.executable).resolve() != _VENV.resolve()
        and not os.environ.get("OPENKAL_NO_REEXEC")):
    os.execv(str(_VENV), [str(_VENV), *sys.argv])

from openkal.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
