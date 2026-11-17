"""Dashboard transport: one JSON snapshot, one static directory, no framework server.

stdlib only, on loopback by default. The engine publishes an immutable snapshot
string after every pass; the handler never touches engine internals.
"""
from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


class SnapshotBox:
    def __init__(self) -> None:
        self._payload = json.dumps({"mode": "booting"}).encode()
        self._lock = threading.Lock()

    def put(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, default=str).encode()
        with self._lock:
            self._payload = data

    def get(self) -> bytes:
        with self._lock:
            return self._payload


def make_handler(box: SnapshotBox, static_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the terminal readable
            return

        def _send(self, status: int, body: bytes, content_type: str,
                  cache: str = "no-store") -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _route(self, path: str) -> tuple[bytes, str, str]:
            """(body, content-type, cache-control) for one request path."""
            if path in {"/api/state", "/api/health"}:
                return box.get(), "application/json", "no-store"
            if path == "/api/ping":
                return b'{"ok":true}', "application/json", "no-store"
            target = (static_dir / path.lstrip("/")).resolve()
            if static_dir.exists() and str(target).startswith(str(static_dir.resolve())) and target.is_file():
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                cache = "public, max-age=31536000, immutable" if "/assets/" in str(target) else "no-cache"
                return target.read_bytes(), ctype, cache
            index = static_dir / "index.html"
            if index.exists():
                return index.read_bytes(), "text/html; charset=utf-8", "no-store"
            return (b"<h1>Open-Kal</h1><p>Dashboard not built yet. Run <code>npm run build</code> in dashboard/.</p>",
                    "text/html; charset=utf-8", "no-store")

        def _send_head(self, body: bytes, content_type: str, cache: str) -> None:
            self.send_response(200)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", cache)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            body, ctype, cache = self._route(self.path.split("?")[0])
            self._send(200, body, ctype, cache)

        def do_HEAD(self) -> None:  # noqa: N802
            # A HEAD is a GET without the body: `curl -I` on an asset is a real check, not a 501.
            body, ctype, cache = self._route(self.path.split("?")[0])
            self._send_head(body, ctype, cache)

    return Handler


def serve(host: str, port: int, box: SnapshotBox, static_dir: Path,
          ready: Callable[[], None] | None = None) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(box, static_dir))
    server.daemon_threads = True
    if ready:
        ready()
    return server
