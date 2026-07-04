"""Control plane: the API is drop-in compatible and the fleet IS the ModuleTenant fleet."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")          # control-plane extra; skip if not installed
from fastapi.testclient import TestClient

from context_runtime.control_plane.app import app

client = TestClient(app)


def test_health_reports_registry():
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["modules"] >= 14


def test_all_modules_deployed_as_tenants():
    # (b): standing the fleet up flips every business module to deployed=true
    st = client.get("/status").json()
    assert len(st) >= 14
    assert all(m["deployed"] for m in st)
    assert any(m["detail"] == "context-runtime tenant" for m in st)


def test_modules_registry_shape():
    mods = client.get("/modules").json()
    names = {m["name"] for m in mods}
    assert {"agentic-billing", "edge-sentinel", "control-tower", "sidekick"} <= names


def test_agent_run_routes_through_the_tenant():
    r = client.post("/agent/run", json={"module": "control-tower", "question": "why did revenue fall?"})
    j = r.json()
    assert j["module"] == "control-tower"
    assert j["sources"] and j["plan_id"].startswith("plan_")


def test_money_action_is_approval_gated():
    d = client.post("/dispatch", json={"module": "agentic-billing", "agent": "checkout",
                                       "action": "refund", "prompt": "refund duplicate charge"})
    assert d.json()["kind"] == "approval"


def test_down_up_cycle_changes_deployed_state():
    client.post("/down", json=["agentic-billing"])
    after = {m["name"]: m["deployed"] for m in client.get("/status").json()}
    assert after["agentic-billing"] is False
    client.post("/up", json=["agentic-billing"])
    back = {m["name"]: m["deployed"] for m in client.get("/status").json()}
    assert back["agentic-billing"] is True


def test_unknown_module_404():
    assert client.post("/agent/run", json={"module": "nope", "question": "x"}).status_code == 404


def test_mutating_endpoints_require_api_key(monkeypatch):
    # With an API key configured, protected POSTs demand a matching X-API-Key header (else 401).
    # The rest of this suite runs with the key UNSET, so auth is disabled there — this is the only
    # coverage of the 401 branch that guards every money/infra/ingest mutation.
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", "s3cret")
    protected = [
        ("/up", {}),
        ("/dispatch", {"module": "agentic-billing", "agent": "checkout", "action": "refund", "prompt": "x"}),
        ("/agent/run", {"module": "control-tower", "question": "q"}),
    ]
    for path, body in protected:
        assert client.post(path, json=body).status_code == 401              # no header → 401
        assert client.post(path, json=body, headers={"X-API-Key": "nope"}).status_code == 401  # wrong key → 401
    # correct key → auth passes (handler runs; status is anything but 401)
    ok = client.post("/agent/run", json={"module": "control-tower", "question": "q"},
                     headers={"X-API-Key": "s3cret"})
    assert ok.status_code != 401


def test_agent_outcome_closes_learning_loop():
    r = client.post("/agent/outcome", json={"module": "control-tower",
                                            "question": "why did revenue fall?", "success": True})
    j = r.json()
    assert j["module"] == "control-tower" and "reward" in j and isinstance(j["policy"], dict)
    # undeployed module → 404
    assert client.post("/agent/outcome", json={"module": "nope", "question": "q",
                                               "success": True}).status_code == 404


def test_approvals_resolve_and_validation():
    client.post("/dispatch", json={"module": "agentic-billing", "agent": "checkout",
                                   "action": "refund", "prompt": "refund a duplicate charge"})
    pending = client.get("/approvals").json()
    assert pending, "money action should have created a pending approval"
    aid = pending[0]["id"]
    assert client.post(f"/approvals/{aid}/frobnicate").status_code == 400   # bad decision
    assert client.post("/approvals/ap-does-not-exist/approve").status_code == 404  # unknown id
    ok = client.post(f"/approvals/{aid}/approve")
    assert ok.status_code == 200 and ok.json()["id"] == aid


def test_v1_chat_abstain_branch_shapes_response(monkeypatch):
    # the module tenant has no corpus, so drive the abstain branch of _chat_core directly by
    # returning an abstaining retrieval context — asserts the response shaping, not re-testing
    # the calibrate/abstain decision (covered in test_librechat).
    import types

    import context_runtime.control_plane.app as cpapp
    strat = cpapp.librechat.strategies[0]

    class _AbstainCtx:
        abstain = True
        context = ""
        hits = ()
        probs = ()
        max_p_rel = 0.1
        strategy = strat
        plan = types.SimpleNamespace(id="plan_test_abstain")

    monkeypatch.setattr(cpapp.librechat, "retrieve", lambda text: _AbstainCtx())
    r = client.post("/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "what is alpha"}]})
    j = r.json()
    assert j["context_runtime"]["abstained"] is True
    assert "enough relevant context" in j["choices"][0]["message"]["content"].lower()
    assert j["usage"]["total_tokens"] == 0     # abstaining skips the upstream forward entirely
    assert j["id"] == "chatcmpl-plan_test_abstain"
