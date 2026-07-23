"""Serving surface (§2.4 auth + MCP tool, §2.5 Bedrock /v1 upstream).

Uses the control-plane extra + monkeypatch so module globals never leak between tests.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from context_runtime.control_plane import app as appmod
from context_runtime.control_plane import mcp_server
from context_runtime.types import ModelResult

client = TestClient(appmod.app)


# ── §2.4 auth on the read routes (opt-in: enforced only when a key is set) ───────────────────────
def test_retrieve_open_when_no_key_set(monkeypatch):
    monkeypatch.delenv("CONTEXT_RUNTIME_API_KEY", raising=False)
    monkeypatch.delenv("AGENTIC_OS_API_KEY", raising=False)
    r = client.post("/librechat/retrieve", json={"request": "hello"})
    assert r.status_code != 401         # unchanged default: open on localhost


@pytest.mark.parametrize("route", ["/librechat/retrieve", "/librechat/compare", "/librechat/explain"])
def test_read_routes_require_key_when_set(monkeypatch, route):
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", "secret")
    assert client.post(route, json={"request": "x"}).status_code == 401           # missing header
    ok = client.post(route, json={"request": "x"}, headers={"X-API-Key": "secret"})
    assert ok.status_code != 401                                                  # correct header passes auth


# ── §2.5 Bedrock as the /v1 upstream ─────────────────────────────────────────────────────────────
class FakeBedrockModel:
    def __init__(self):
        self.seen = None

    def complete(self, req):
        self.seen = req
        return ModelResult(text="bedrock says hi", model="amazon.nova-lite-v1:0", tier="chat",
                           prompt_tokens=7, completion_tokens=3)


def test_forward_prefers_bedrock_when_configured(monkeypatch):
    fake = FakeBedrockModel()
    monkeypatch.setattr(appmod, "_BEDROCK_UPSTREAM", fake)
    monkeypatch.setattr(appmod, "_BEDROCK_UPSTREAM_BUILT", True)
    messages = [{"role": "system", "content": "RETRIEVED CONTEXT: ..."}, {"role": "user", "content": "hi"}]
    req = appmod.ChatCompletionReq(messages=[appmod.ChatMessage(role="user", content="hi")])
    answer, usage, model = appmod._forward_to_upstream(messages, req)
    assert answer == "bedrock says hi"
    assert usage["total_tokens"] == 10 and model == "amazon.nova-lite-v1:0"
    # system prompt is lifted into ModelRequest.system; conversation excludes the system turn
    assert fake.seen.system == "RETRIEVED CONTEXT: ..."
    assert all(m["role"] != "system" for m in fake.seen.messages)
    assert fake.seen.max_tokens >= 2048       # the shim's max-tokens floor is applied


def test_bedrock_upstream_lazy_builder_off_by_default(monkeypatch):
    monkeypatch.setattr(appmod, "_BEDROCK_UPSTREAM", None)
    monkeypatch.setattr(appmod, "_BEDROCK_UPSTREAM_BUILT", False)
    monkeypatch.delenv("CR_UPSTREAM_PROVIDER", raising=False)
    monkeypatch.delenv("CR_BEDROCK_MODEL", raising=False)
    assert appmod._bedrock_upstream_model() is None      # not configured → no Bedrock, use other paths


# ── §2.4 the MCP tool logic (fastmcp-free core) ──────────────────────────────────────────────────
def test_mcp_retrieve_context_returns_bundle():
    ctx = SimpleNamespace(request="q", context="CTX", hits=[],
                          strategy=SimpleNamespace(key="hybrid:cheap", method="hybrid", final_k=5),
                          plan=SimpleNamespace(id="p1"))
    tenant = SimpleNamespace(retrieve=lambda r: ctx, suggest=lambda r: "try graph")
    out = mcp_server.retrieve_context("q", tenant_resolver=lambda m: tenant)
    assert out["method"] == "hybrid" and out["plan_id"] == "p1"
    assert out["context"] == "CTX" and out["suggestion"] == "try graph" and out["hits"] == []
