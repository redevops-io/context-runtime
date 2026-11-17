"""The container health probe follows PUBLIC_BASE_PATH: it requests <prefix>/healthz and nothing else.

Stdlib only: a loopback HTTP server on 127.0.0.1 answers only the path under test; the probe script runs as a
subprocess exactly as the image's HEALTHCHECK runs it.
"""
from __future__ import annotations

import http.server
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[1] / "deploy" / "geo-demo" / "healthcheck.py"


class _Server:
    def __init__(self, ok_path: str):
        seen: list[str] = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.path)
                self.send_response(200 if self.path == ok_path else 404)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass

        self.seen = seen
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def probe(port: int, base: str | None) -> int:
    env = {k: v for k, v in os.environ.items() if k != "PUBLIC_BASE_PATH"}
    env["PORT"] = str(port)
    if base is not None:
        env["PUBLIC_BASE_PATH"] = base
    return subprocess.run([sys.executable, str(PROBE)], env=env, timeout=30).returncode


@pytest.mark.parametrize("base", [None, "", "/geo", "/shared/geo"])
def test_probe_requests_the_prefixed_health_path(base):
    srv = _Server((base or "") + "/healthz")
    try:
        assert probe(srv.port, base) == 0
        assert srv.seen == [(base or "") + "/healthz"]
    finally:
        srv.close()


def test_probe_fails_when_only_the_other_path_answers():
    root_only, prefixed_only = _Server("/healthz"), _Server("/geo/healthz")
    try:
        assert probe(root_only.port, "/geo") == 1            # a prefix is set but only the root answers
        assert probe(prefixed_only.port, None) == 1          # no prefix but only the prefixed path answers
    finally:
        root_only.close()
        prefixed_only.close()


@pytest.mark.parametrize("base", ["geo", "/geo/", "/", "/a/../b", "/geo bar", "/."])
def test_probe_fails_on_a_value_the_app_would_refuse(base):
    srv = _Server("/healthz")
    try:
        assert probe(srv.port, base) == 1
        assert srv.seen == []                                # it did not probe some other path instead
    finally:
        srv.close()
