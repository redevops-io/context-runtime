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
