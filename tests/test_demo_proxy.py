"""The public demo stays usable through its proxy while the control plane stays strict.

Runs the real control plane (uvicorn) behind the unmodified `deploy/proxmox-demo/Caddyfile` with a
stub standing in for LibreChat, all on loopback, in a scrubbed environment, with synthetic keys.
The proxy holds only the read-only tenant key. A client without any credential can use the routes the
demo UI calls (/v1/models, /v1/chat/completions, /librechat/retrieve|compare|explain|policy); ingest and
every other control-plane route is refused at the proxy; the app itself answers nothing but /health to
a caller without a credential. Skipped when caddy is absent.
Set CR_DEMO_PROXY_TRANSCRIPT=<file> to write the request/response transcript.
"""
from __future__ import annotations

import http.client
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

ROOT = Path(__file__).resolve().parent.parent
CADDYFILE = ROOT / "deploy" / "proxmox-demo" / "Caddyfile"
ADMIN_KEY = "synthetic-ADMIN-key-7f3a91c2d4e6"
READ_KEY = "synthetic-READ-key-1b8e5d0a9c47"
SENTENCE = "The synthetic project codename is BLUE-HERON-4471."
STUB_BODY = "librechat-stub"

pytestmark = pytest.mark.skipif(shutil.which("caddy") is None, reason="caddy is not installed")


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait(port: int, path: str = "/health", tries: int = 150) -> None:
    for _ in range(tries):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", path)
            c.getresponse().read()
            c.close()
            return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(f"nothing answered on {port}")


class _Stub(http.server.BaseHTTPRequestHandler):
    def _any(self):
        self.rfile.read(int(self.headers.get("content-length") or 0))
        self.send_response(200)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(STUB_BODY)))
        self.end_headers()
        self.wfile.write(STUB_BODY.encode())
    do_GET = do_POST = _any

    def log_message(self, *a):
        pass


class Stack:
    def __init__(self, tmp: Path):
        self.log: list[str] = []
        self.app_port, self.proxy_port, self.stub_port = _port(), _port(), _port()
        self.root = tmp / "corpus-root"
        for t in ("a", "b", "served"):
            (self.root / t).mkdir(parents=True)
        (self.root / "served" / "note.md").write_text(f"# Note\n\n{SENTENCE}\n")
        home = tmp / "home"
        home.mkdir()
        self.procs: list[subprocess.Popen] = []
        self.stub = http.server.HTTPServer(("127.0.0.1", self.stub_port), _Stub)
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        base = {"HOME": str(home), "PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1", "XDG_DATA_HOME": str(home / "d"), "XDG_CONFIG_HOME": str(home / "c")}
        app_env = {**base, "CONTEXT_RUNTIME_HOME": str(home / "cr"),
                   "CONTEXT_RUNTIME_API_KEY": ADMIN_KEY, "CR_TENANT_KEYS": f"{READ_KEY}=*",
                   "CONTEXT_RUNTIME_CORPUS_ROOT": str(self.root),
                   "CR_TENANTS": f"demo-finance={self.root / 'a'} | demo-medical={self.root / 'b'}"}
        self.procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "context_runtime.control_plane.app:app", "--host", "127.0.0.1",
             "--port", str(self.app_port), "--log-level", "warning"],
            env=app_env, cwd=str(home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        wrapper = tmp / "Caddyfile.test"
        wrapper.write_text(f"{{\n\tadmin off\n\tauto_https off\n}}\nimport {CADDYFILE}\n")
        caddy_env = {**base, "CR_PROXY_LISTEN": f"http://127.0.0.1:{self.proxy_port}",
                     "CR_UPSTREAM": f"127.0.0.1:{self.app_port}", "LIBRECHAT_UPSTREAM": f"127.0.0.1:{self.stub_port}",
                     "CR_PUBLIC_READ_KEY": READ_KEY}      # the proxy gets the read key and nothing else
        self.procs.append(subprocess.Popen(
            [shutil.which("caddy"), "run", "--config", str(wrapper), "--adapter", "caddyfile"],
            env=caddy_env, cwd=str(home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        _wait(self.app_port)
        _wait(self.proxy_port, "/anything")

    def req(self, port: int, method: str, path: str, body=None, headers=None, label: str = "") -> tuple[int, str]:
        h = dict(headers or {})
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        text = r.read().decode("utf8", "replace")
        c.close()
        who = "proxy" if port == self.proxy_port else "app  "
        cred = ("admin-key" if h.get("X-API-Key") == ADMIN_KEY else "read-key" if h.get("X-API-Key") == READ_KEY
                else "bad-credential" if ("X-API-Key" in h or "Authorization" in h) else "no credential")
        self.log.append(f"{who} {method:4} {path:42} [{cred:14}] -> {r.status}  {text[:90]!r}")
        return r.status, text

    def proxy(self, method, path, body=None, headers=None):
        return self.req(self.proxy_port, method, path, body, headers)

    def app(self, method, path, body=None, headers=None):
        return self.req(self.app_port, method, path, body, headers)

    def close(self):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        self.stub.shutdown()


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    s = Stack(tmp_path_factory.mktemp("demoproxy"))
    try:
        # operator step, directly on the app with the administrative key
        st, body = s.app("POST", "/librechat/ingest", {"corpus": "served"}, {"X-API-Key": ADMIN_KEY})
        assert st == 200, body
        yield s
    finally:
        s.close()
        out = os.environ.get("CR_DEMO_PROXY_TRANSCRIPT")
        if out:
            Path(out).write_text("\n".join(s.log) + "\n")


def test_demo_ui_routes_work_through_the_proxy_without_a_client_credential(stack):
    st, body = stack.proxy("GET", "/v1/models")
    assert st == 200 and {m["id"] for m in json.loads(body)["data"]} == {"demo-finance", "demo-medical"}
    st, body = stack.proxy("POST", "/librechat/retrieve", {"request": "What is the synthetic project codename?"})
    assert st == 200 and "BLUE-HERON-4471" in json.loads(body)["context"]
    st, body = stack.proxy("POST", "/librechat/compare", {"request": "synthetic project codename", "k": 3})
    assert st == 200
    st, body = stack.proxy("POST", "/librechat/explain", {"request": "synthetic project codename", "k": 3})
    assert st == 200
    st, body = stack.proxy("GET", "/librechat/policy")
    assert st == 200 and "policy" in json.loads(body)
    st, body = stack.proxy("GET", "/librechat/image?chunk_id=none")
    assert st == 404 and "librechat-stub" not in body                      # reached the app (no such image)
    st, body = stack.proxy("POST", "/v1/chat/completions",
                           {"messages": [{"role": "user", "content": "What is the synthetic project codename?"}]})
    assert st == 200 and "BLUE-HERON-4471" in json.loads(body)["choices"][0]["message"]["content"]
    st, body = stack.proxy("POST", "/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "codename"}]})
    assert st == 200 and "[DONE]" in body


def test_other_tenants_are_selectable_with_the_proxy_key(stack):
    st, body = stack.proxy("POST", "/librechat/retrieve", {"request": "codename", "model": "demo-medical"})
    assert st == 200


def test_the_proxy_replaces_whatever_credential_the_client_sends(stack):
    for h in ({"X-API-Key": "wrong"}, {"Authorization": "Bearer wrong"}, {"X-API-Key": ADMIN_KEY},
              {"Authorization": f"Bearer {ADMIN_KEY}"}):
        st, _ = stack.proxy("GET", "/v1/models", headers=h)
        assert st == 200
    # a client that offers the admin key still only has reader rights: ingestion is not forwarded
    st, body = stack.proxy("POST", "/librechat/ingest", {"corpus": "served"}, {"X-API-Key": ADMIN_KEY})
    assert st == 404 and "ingested" not in body


ADMIN_PATHS = ["/librechat/ingest", "/librechat/judge", "/librechat/feedback", "/librechat/chat"]


@pytest.mark.parametrize("path", ADMIN_PATHS + [p + "/" for p in ADMIN_PATHS] + ["/librechat/Ingest", "/LIBRECHAT/INGEST",
                         "/librechat/%69ngest", "/librechat/./ingest", "/librechat/x/../ingest", "//librechat/ingest",
                         "/librechat", "/v1", "/v1/", "/v1/other", "/librechat/other"])
def test_ingest_and_every_other_control_plane_path_is_refused_at_the_proxy(stack, path):
    st, body = stack.proxy("POST", path, {"corpus": "a", "request": "x", "score": 0.5, "signal": "kept"})
    # Caddy's own empty 404, or the LibreChat side of the proxy: never an answer of the control plane
    assert (st == 404 and body == "") or body == STUB_BODY, (path, st, body[:80])


@pytest.mark.parametrize("path", ["/health", "/approvals", "/status", "/api/fleet", "/notify/status", "/modules", "/jobs/job-00001",
                                  "/agent/policy/control-tower", "/vibex/criteria", "/vibex/scoreboard", "/m/billing", "/overview",
                                  "/docs", "/openapi.json", "/up", "/down", "/dispatch", "/agent/run", "/agent/outcome"])
def test_control_plane_routes_are_not_reachable_through_the_proxy(stack, path):
    for method in ("GET", "POST"):
        st, body = stack.proxy(method, path, {} if method == "POST" else None)
        assert body == STUB_BODY, (method, path, st, body[:80])        # LibreChat's side of the proxy, not the control plane


def test_the_app_answers_nothing_but_health_without_a_credential(stack):
    assert stack.app("GET", "/health")[0] == 200
    for method, path in [("GET", "/v1/models"), ("POST", "/librechat/retrieve"), ("POST", "/librechat/compare"),
                         ("POST", "/librechat/explain"), ("POST", "/v1/chat/completions"), ("GET", "/approvals"),
                         ("GET", "/status"), ("GET", "/api/fleet"), ("POST", "/librechat/ingest"), ("POST", "/dispatch")]:
        assert stack.app(method, path, {} if method == "POST" else None)[0] == 401, path


def test_the_read_key_cannot_open_administrative_routes_on_the_app(stack):
    h = {"X-API-Key": READ_KEY}
    for method, path in [("GET", "/approvals"), ("GET", "/status"), ("POST", "/librechat/ingest"), ("POST", "/librechat/feedback"),
                         ("POST", "/librechat/judge"), ("POST", "/librechat/chat"), ("POST", "/up"), ("GET", "/vibex/criteria")]:
        assert stack.app(method, path, {} if method == "POST" else None, h)[0] == 401, path
    assert stack.app("GET", "/v1/models", headers=h)[0] == 200
