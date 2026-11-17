"""Entry point: one process, two planes.

Plane A (asyncio) owns market data, decisions and fills.
Plane B (a background thread) serves the snapshot to the dashboard.
The SHR verifier runs inside plane A but has no order path and reads only the store.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import threading
import time

from openkal.api import SnapshotBox, serve
from openkal.config import load_settings
from openkal.engine import Engine
from openkal.store import Store


def _crash_log_path():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent / "data" / "engine.log"


def _write_traceback(header: str, exc, tb=None) -> None:
    import traceback

    try:
        with open(_crash_log_path(), "a", buffering=1) as fh:
            fh.write(f"\n===== {header} {type(exc).__name__ if exc else ''} =====\n")
            if tb is not None:
                traceback.print_exception(type(exc), exc, tb, file=fh)
            else:
                traceback.print_exception(exc if exc else ValueError("unknown"), file=fh)
            fh.write("\n")
    except Exception:  # noqa: BLE001 - never raise inside the crash logger
        traceback.print_exception(type(exc) if exc else Exception, exc, getattr(exc, "__traceback__", None))


def _install_crash_logging() -> None:
    """Flush any unexpected exit to data/engine.log with a full traceback."""
    def _hook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        _write_traceback("sys.excepthook", exc, tb)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


def _crash_handler(loop, context) -> None:
    exc = context.get("exception") if isinstance(context, dict) else None
    if exc is not None:
        _write_traceback("asyncio exception handler", exc, exc.__traceback__)
    else:
        detail = context.get("message", "event loop logged error") if isinstance(context, dict) else str(context)
        _write_traceback("asyncio exception handler", RuntimeError(detail), None)
    loop.default_exception_handler(context)


async def _amain(args: argparse.Namespace) -> int:
    _install_crash_logging()
    asyncio.get_running_loop().set_exception_handler(_crash_handler)
    settings = load_settings()
    settings.host = args.host
    settings.port = args.port
    store = Store(settings.db_path)
    engine = Engine(settings, store)
    box = SnapshotBox()

    server = None
    if not args.no_dashboard:
        server = serve(settings.host, settings.port, box, settings.dashboard_dir)
        threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
        print(f"Open-Kal dashboard  http://{settings.host}:{settings.port}", flush=True)
    gate = "open" if (settings.arm_live and settings.live) else "closed"
    print(f"Open-Kal engine     env={settings.env_name} execution=paper "
          f"(live gate {gate}, arms only when the functional readiness gate passes) "
          f"slots={len(engine.slots)} equity=${settings.paper_equity or 'venue balance'}", flush=True)

    async def publish() -> None:
        while True:
            box.put(engine.snapshot())
            await asyncio.sleep(0.5)

    publisher = asyncio.create_task(publish())
    try:
        if args.duration:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(engine.run(), timeout=args.duration)
        else:
            await engine.run()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        engine.stop()
        publisher.cancel()
        with contextlib.suppress(Exception):
            await publisher
        if server:
            server.shutdown()
        store.prune()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="open-kal", description="Open-Kal 15-minute Kalshi engine")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (testing)")
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
