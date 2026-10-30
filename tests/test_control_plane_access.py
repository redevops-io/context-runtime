"""Control plane access: fail closed without configured authentication, every route classified.

Covers: a deployment with no key (503 on everything but /health), a deployment with a key (401
without or with a wrong key, success with the key), anonymous reads of every route that used to be
open, unauthorised writes, approval decisions, the tenant-key relationship, and a route-table drift
check (a new route that is not classified, or a new open route, fails here).
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from starlette.routing import Mount

from context_runtime.control_plane import app as A

KEY = "access-test-control-plane-key"
TENANT_KEY = "access-test-tenant-key"

PUBLIC, READER, ADMIN = "public", "reader", "admin"

# The expected class of every route, written out independently of the table in app.py.
# A new route must be added here on purpose; a route that is open and not listed as PUBLIC fails.
EXPECTED: dict[tuple[str, str], str] = {
    ("GET", "/health"): PUBLIC,
    ("GET", "/"): ADMIN, ("GET", "/overview"): ADMIN, ("GET", "/modules"): ADMIN, ("GET", "/status"): ADMIN,
    ("GET", "/api/fleet"): ADMIN, ("GET", "/notify/status"): ADMIN,
    ("GET", "/m/{name}"): ADMIN, ("GET", "/m/{name}/raw"): ADMIN,
    ("GET", "/m/{name}/api/{path:path}"): ADMIN, ("POST", "/m/{name}/api/{path:path}"): ADMIN,
    ("POST", "/up"): ADMIN, ("POST", "/down"): ADMIN, ("GET", "/approvals"): ADMIN,
    ("POST", "/approvals/{approval_id}/{decision}"): ADMIN, ("POST", "/dispatch"): ADMIN,
    ("GET", "/jobs/{job_id}"): ADMIN, ("POST", "/agent/run"): ADMIN, ("POST", "/agent/outcome"): ADMIN,
    ("GET", "/agent/policy/{module}"): ADMIN,
    ("GET", "/vibex/criteria"): ADMIN, ("POST", "/vibex/scenario"): ADMIN, ("POST", "/vibex/plan"): ADMIN,
    ("POST", "/vibex/score"): ADMIN, ("GET", "/vibex/scoreboard"): ADMIN,
    ("POST", "/librechat/ingest"): ADMIN, ("POST", "/librechat/judge"): ADMIN,
    ("POST", "/librechat/feedback"): ADMIN, ("POST", "/librechat/chat"): ADMIN,
    ("POST", "/librechat/retrieve"): READER, ("POST", "/librechat/compare"): READER,
    ("POST", "/librechat/explain"): READER, ("GET", "/librechat/image"): READER,
    ("GET", "/librechat/policy"): READER, ("GET", "/v1/models"): READER,
    ("POST", "/v1/chat/completions"): READER,
}


def _live_routes() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for r in A.app.routes:
        if isinstance(r, Mount):
            out.add(("*", r.path))
            continue
        for m in sorted((getattr(r, "methods", None) or {"GET"}) - {"HEAD", "OPTIONS"}):
            out.add((m, r.path))
    return out


def _url(path: str) -> str:
    for param, value in (("{name}", "nope"), ("{path:path}", "x"), ("{approval_id}", "ap-none"),
                         ("{decision}", "approve"), ("{job_id}", "job-00001"), ("{module}", "control-tower")):
        path = path.replace(param, value)
    return path


@pytest.fixture()
def no_key(monkeypatch):
    for name in ("CONTEXT_RUNTIME_API_KEY", "AGENTIC_OS_API_KEY", "CR_TENANT_KEYS"):
        monkeypatch.delenv(name, raising=False)
    return TestClient(A.app)


@pytest.fixture()
def keyed(monkeypatch):
    monkeypatch.delenv("AGENTIC_OS_API_KEY", raising=False)
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", KEY)
    monkeypatch.setenv("CR_TENANT_KEYS", f"{TENANT_KEY}={A._DEFAULT_TENANT_ID}")
    return TestClient(A.app)


# ───────────────────────────── route table ─────────────────────────────

def test_every_route_has_an_expected_access_class():
    live = _live_routes()
    assert live - set(EXPECTED) == set(), "new route without an expected access class (open by accident?)"
    assert set(EXPECTED) - live == set(), "expected route that no longer exists"


def test_only_health_is_public():
    assert [k for k, v in EXPECTED.items() if v == PUBLIC] == [("GET", "/health")]


def test_access_table_in_the_app_matches_the_expected_classes():
    for (method, path), cls in EXPECTED.items():
        assert A.route_access(method, path).value == cls, (method, path)
    # the optional static mount (only present when the assets directory exists) is classified too
    assert A.route_access("GET", "/assets") is A.Access.ADMIN
    assert {k: v.value for k, v in A.ACCESS.items() if k[0] != "*"} == EXPECTED


def test_generated_documentation_routes_are_not_served(keyed):
    for path in ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"):
        assert keyed.get(path, headers={"X-API-Key": KEY}).status_code == 404


def test_no_websocket_routes():
    assert not [r for r in A.app.routes if type(r).__name__.startswith("WebSocket")]


# ───────────────────────────── no key configured: fail closed ─────────────────────────────

def test_health_is_public_and_minimal(no_key):
    r = no_key.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("method,path", sorted(k for k, v in EXPECTED.items() if v != PUBLIC and k[0] != "*"))
def test_without_a_configured_key_every_protected_route_is_503(no_key, method, path):
    r = no_key.request(method, _url(path))
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"]
    # supplying a credential does not help when the server has none configured
    assert no_key.request(method, _url(path), headers={"X-API-Key": "anything"}).status_code == 503


def test_anonymous_requests_without_a_key_change_nothing(no_key):
    """The demonstrated no-key actions: dispatch, approve, up, agent/outcome, ingest."""
    before = len(A.fleet.context.pending())
    assert no_key.post("/dispatch", json={"module": "agentic-billing", "agent": "checkout",
                                          "action": "refund", "prompt": "SYNTHETIC"}).status_code == 503
    assert len(A.fleet.context.pending()) == before
    assert no_key.post("/up", json=["agentic-billing"]).status_code == 503
    assert no_key.post("/agent/outcome", json={"module": "control-tower", "question": "q",
                                               "success": False}).status_code == 503
    assert no_key.post("/librechat/ingest", json={"path": "/tmp"}).status_code == 503


def test_whitespace_only_key_counts_as_not_configured(no_key, monkeypatch):
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", "   ")
    assert no_key.get("/approvals", headers={"X-API-Key": "   "}).status_code == 503


def test_agentic_os_key_variable_is_still_honoured(no_key, monkeypatch):
    monkeypatch.setenv("AGENTIC_OS_API_KEY", "legacy-name-key")
    assert no_key.get("/approvals").status_code == 401
    assert no_key.get("/approvals", headers={"X-API-Key": "legacy-name-key"}).status_code == 200


# ───────────────────────────── key configured ─────────────────────────────

# Every route that used to answer an anonymous caller (reads, and the POSTs that were open without a key).
FORMERLY_OPEN_READS = ["/approvals", "/jobs/job-00001", "/status", "/api/fleet", "/notify/status",
                       "/agent/policy/control-tower", "/modules", "/", "/overview", "/m/billing",
                       "/m/billing/raw", "/m/billing/api/x", "/vibex/criteria", "/vibex/scoreboard",
                       "/v1/models", "/librechat/policy", "/librechat/image?chunk_id=x"]


@pytest.mark.parametrize("path", FORMERLY_OPEN_READS)
def test_anonymous_reads_are_refused(keyed, path):
    assert keyed.get(path).status_code == 401
    assert keyed.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert keyed.get(path, headers={"Authorization": "Bearer wrong"}).status_code == 401


@pytest.mark.parametrize("method,path", sorted(k for k, v in EXPECTED.items() if v != PUBLIC and k[0] != "*"))
def test_with_a_key_configured_no_or_wrong_credential_is_401(keyed, method, path):
    assert keyed.request(method, _url(path)).status_code == 401
    assert keyed.request(method, _url(path), headers={"X-API-Key": "wrong"}).status_code == 401
    assert keyed.request(method, _url(path), headers={"X-API-Key": KEY + "x"}).status_code == 401
    assert keyed.request(method, _url(path), headers={"X-API-Key": KEY[:-1]}).status_code == 401


@pytest.mark.parametrize("method,path", sorted(k for k, v in EXPECTED.items() if v == ADMIN and k[0] != "*"))
def test_a_tenant_key_does_not_open_control_plane_routes(keyed, method, path):
    assert keyed.request(method, _url(path), headers={"X-API-Key": TENANT_KEY}).status_code == 401


def test_authorised_caller_reaches_the_handlers(keyed):
    h = {"X-API-Key": KEY}
    assert keyed.get("/health").status_code == 200
    assert keyed.get("/approvals", headers=h).status_code == 200
    assert keyed.get("/status", headers=h).status_code == 200
    assert keyed.get("/api/fleet", headers=h).status_code == 200
    assert keyed.get("/notify/status", headers=h).status_code == 200
    assert keyed.get("/agent/policy/control-tower", headers=h).status_code == 200
    assert keyed.get("/jobs/job-99999", headers=h).status_code == 404         # handler ran
    assert keyed.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_reader_routes_accept_the_tenant_key_and_the_control_plane_key(keyed):
    for cred in (KEY, TENANT_KEY):
        r = keyed.post("/librechat/retrieve", json={"request": "hello"}, headers={"X-API-Key": cred})
        assert r.status_code == 200, cred


def test_approval_reads_and_decisions_need_the_key(keyed):
    h = {"X-API-Key": KEY}
    d = keyed.post("/dispatch", json={"module": "agentic-billing", "agent": "checkout", "action": "refund",
                                      "prompt": "SYNTHETIC approval request text"}, headers=h)
    assert d.status_code == 200 and d.json()["kind"] == "approval"
    aid = d.json()["id"]
    # anonymous read of the request text, anonymous decisions
    assert keyed.get("/approvals").status_code == 401
    assert keyed.post(f"/approvals/{aid}/approve").status_code == 401
    assert keyed.post(f"/approvals/{aid}/reject").status_code == 401
    assert keyed.post(f"/approvals/{aid}/approve", headers={"X-API-Key": TENANT_KEY}).status_code == 401
    still = [a for a in keyed.get("/approvals", headers=h).json() if a["id"] == aid]
    assert still, "the approval must still be pending after the refused decisions"
    ok = keyed.post(f"/approvals/{aid}/approve", headers=h)
    assert ok.status_code == 200 and ok.json()["status"] == "approved"


def test_unauthorised_writes_change_nothing(keyed):
    before = {m["name"]: m["deployed"] for m in keyed.get("/status", headers={"X-API-Key": KEY}).json()}
    policy_before = keyed.get("/agent/policy/control-tower", headers={"X-API-Key": KEY}).json()
    assert keyed.post("/down", json=["agentic-billing"]).status_code == 401
    assert keyed.post("/agent/outcome", json={"module": "control-tower", "question": "q", "success": False}).status_code == 401
    assert keyed.post("/agent/run", json={"module": "control-tower", "question": "q"}).status_code == 401
    after = {m["name"]: m["deployed"] for m in keyed.get("/status", headers={"X-API-Key": KEY}).json()}
    assert after == before
    assert keyed.get("/agent/policy/control-tower", headers={"X-API-Key": KEY}).json() == policy_before


def test_preflight_requests_do_not_reach_the_handlers(keyed):
    r = keyed.options("/approvals", headers={"Origin": "http://localhost:8080",
                                             "Access-Control-Request-Method": "GET"})
    assert r.status_code == 200 and "approvals" not in r.text


def test_unknown_paths_are_plain_404(keyed):
    assert keyed.get("/nope").status_code == 404


# ───────────────────────────── comparisons ─────────────────────────────

def test_credentials_are_compared_in_constant_time(monkeypatch):
    """No equality operator on a credential in the control plane; the comparison goes through hmac.compare_digest."""
    import inspect

    from context_runtime.control_plane import security
    src = inspect.getsource(security) + inspect.getsource(A)
    assert "hmac.compare_digest" in inspect.getsource(security.digest_equal)
    for needle in ("x_api_key !=", "x_api_key ==", "expected !=", "expected =="):
        assert needle not in src
    seen = []
    monkeypatch.setattr(security.hmac, "compare_digest", lambda a, b: seen.append((a, b)) or a == b)
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", KEY)
    monkeypatch.setenv("CR_TENANT_KEYS", "k1=a;k2=b")
    assert security.authorize(security.Access.READER, "k2") is None
    assert len(seen) == 3                       # the control-plane key and both tenant keys, every time
    seen.clear()
    assert security.authorize(security.Access.ADMIN, "nope") == (401, "invalid or missing credential")
    assert len(seen) == 1
    seen.clear()
    assert security.allowed_tenants("k1", security.parse_tenant_keys("k1=a;k2=b"), ["a", "b"], "a", False) == {"a"}
    assert len(seen) == 2
