"""Control plane: proxied module paths are validated; tenants are bound to credentials."""
from __future__ import annotations

import http.server
import threading

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from context_runtime.control_plane import app as A
from context_runtime.control_plane.security import UnsafePath, safe_api_path, upstream_api_url

KEY = "hardening-test-control-plane-key"


@pytest.fixture(autouse=True)
def _configured_server(monkeypatch):
    """The control plane fails closed: these tests run against a deployment with a key configured."""
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", KEY)


def _client() -> TestClient:
    return TestClient(A.app, headers={"X-API-Key": KEY})


# ───────────────────────────── module proxy paths ─────────────────────────────

@pytest.fixture()
def upstream(monkeypatch):
    seen: list[str] = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": 1}')
        do_POST = do_GET

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    name = next(iter(A.MODULE_SERVICES))
    monkeypatch.setitem(A.MODULE_SERVICES, name, f"http://127.0.0.1:{srv.server_port}")
    yield name, seen
    srv.shutdown()


@pytest.mark.parametrize("bad", [
    "..%2fagent%2frun", "%2e%2e/agent/run", "%252e%252e/agent/run", "%25252e%25252e/x",
    "a%5c..%5cagent", "a\\b", "a%00b", "%2e%2e%2fagent", "a//b", "a%20b",
])
def test_rejects_unsafe_module_paths(upstream, bad):
    name, seen = upstream
    r = _client().get(f"/m/{name}/api/{bad}")
    assert r.status_code == 400
    assert seen == []


@pytest.mark.parametrize("bad", ["a/../../agent", "a%2fb", "./x", "a/./b", "a%0ab", "a/%2E/b"])
def test_validator_rejects_dot_and_encoded_separators(bad):
    with pytest.raises(UnsafePath):
        safe_api_path(bad)


@pytest.mark.parametrize("good", ["chat", "v1/items/42", "items/", "a-b_c.d/e~f", "x?q=1"])
def test_forwards_plain_module_paths(upstream, good):
    name, seen = upstream
    r = _client().get(f"/m/{name}/api/{good}")
    assert r.status_code == 200
    assert seen and seen[-1].startswith("/api/")


def test_upstream_url_stays_under_api():
    assert upstream_api_url("http://h:1", "a/b") == "http://h:1/api/a/b"
    assert upstream_api_url("http://h:1/", "") == "http://h:1/api/"
    with pytest.raises(UnsafePath):
        upstream_api_url("http://h:1", "../x")
    with pytest.raises(UnsafePath):
        safe_api_path("a/%2e%2e/b")


# ───────────────────────────── tenant binding ─────────────────────────────

@pytest.fixture()
def tenants(monkeypatch):
    base = A._TENANTS[A._DEFAULT_TENANT_ID]
    monkeypatch.setattr(A, "_TENANTS", {"pub": base, "fin": base, "med": base})
    monkeypatch.setattr(A, "_DEFAULT_TENANT_ID", "pub")
    monkeypatch.delenv("CR_ALLOW_ANONYMOUS_TENANTS", raising=False)
    monkeypatch.setenv("CR_TENANT_KEYS", "k-fin=fin;k-all=*")
    return TestClient(A.app)


# The control-plane key (KEY) is not in CR_TENANT_KEYS: for tenant selection it is an unbound
# credential and gets the default tenant only. Bind it with ``KEY=*`` to use every tenant.
def _retrieve(c, model, key=None):
    h = {"x-api-key": key} if key else {}
    return c.post("/librechat/retrieve", json={"request": "hello", "model": model}, headers=h)


def test_unbound_callers_cannot_select_other_tenants(tenants):
    assert _retrieve(tenants, "fin").status_code == 401             # no credential: refused outright
    assert _retrieve(tenants, "fin", "wrong").status_code == 401    # a credential nobody issued
    assert _retrieve(tenants, "fin", KEY).status_code == 403        # authenticated but not bound to "fin"
    assert _retrieve(tenants, "pub", KEY).status_code == 200
    assert _retrieve(tenants, "unknown-model", KEY).status_code == 200   # falls back to default


def test_bound_credential_selects_only_its_tenants(tenants):
    assert _retrieve(tenants, "fin", "k-fin").status_code == 200
    assert _retrieve(tenants, "med", "k-fin").status_code == 403
    assert _retrieve(tenants, "pub", "k-fin").status_code == 403   # default not in its binding
    assert _retrieve(tenants, "med", "k-all").status_code == 200


def test_bearer_token_is_accepted_as_credential(tenants):
    r = tenants.post("/librechat/retrieve", json={"request": "hi", "model": "fin"},
                     headers={"authorization": "Bearer k-fin"})
    assert r.status_code == 200


def test_models_list_only_shows_permitted_tenants(tenants):
    ids = lambda h: {m["id"] for m in tenants.get("/v1/models", headers=h).json()["data"]}  # noqa: E731
    assert ids({"x-api-key": KEY}) == {"pub"}
    assert ids({"x-api-key": "k-fin"}) == {"fin"}
    assert ids({"x-api-key": "k-all"}) == {"pub", "fin", "med"}
    assert tenants.get("/v1/models").status_code == 401


def test_chat_and_policy_enforce_binding(tenants):
    body = {"model": "fin", "messages": [{"role": "user", "content": "hi"}]}
    assert tenants.post("/v1/chat/completions", json=body).status_code == 401
    assert tenants.post("/v1/chat/completions", json=body, headers={"x-api-key": KEY}).status_code == 403
    assert tenants.get("/librechat/policy", params={"model": "fin"},
                       headers={"x-api-key": KEY}).status_code == 403
    assert tenants.get("/librechat/policy", params={"model": "fin"},
                       headers={"x-api-key": "k-fin"}).status_code == 200


def test_multi_tenant_mode_for_authenticated_callers_requires_explicit_opt_in(tenants, monkeypatch):
    monkeypatch.setenv("CR_ALLOW_ANONYMOUS_TENANTS", "1")
    assert _retrieve(tenants, "fin", KEY).status_code == 200
    assert _retrieve(tenants, "fin").status_code == 401             # the flag never opens the gate


def test_tenant_key_does_not_open_control_plane_routes(tenants):
    for path in ("/approvals", "/status", "/jobs/job-00001", "/api/fleet", "/agent/policy/control-tower"):
        assert tenants.get(path, headers={"x-api-key": "k-all"}).status_code == 401
    assert tenants.post("/librechat/ingest", json={"corpus": "x"}, headers={"x-api-key": "k-all"}).status_code == 401
