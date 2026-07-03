"""The OpenAI-compatible shim: LibreChat (or any OpenAI client) → self-learning chat.
Skips when FastAPI isn't installed (the [control-plane] extra)."""
from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

os.environ.setdefault("CONTEXT_RUNTIME_HOME", tempfile.mkdtemp(prefix="cr-shim-test-"))
from fastapi.testclient import TestClient  # noqa: E402

from context_runtime.control_plane.app import app  # noqa: E402

client = TestClient(app)


def _ingest_small_corpus() -> None:
    d = tempfile.mkdtemp()
    for name, text in {
        "steroid.txt": "Steroid profile: testosterone, cortisol and DHEA measured by LC-MS/MS.",
        "lipid.txt": "Lipid profile: total cholesterol, LDL, HDL and triglycerides.",
    }.items():
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    assert client.post("/librechat/ingest", json={"path": d}).status_code == 200


def test_v1_models_lists_the_shim_model():
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert any(m["id"] == "context-runtime" for m in data["data"])


def test_v1_chat_completion_is_openai_shaped_and_grounded():
    _ingest_small_corpus()
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "what is in the steroid profile testosterone panel?"}]})
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion"
    assert j["choices"][0]["finish_reason"] == "stop"
    assert j["choices"][0]["message"]["role"] == "assistant"
    # the answer is grounded in the retrieved context (offline stub echoes it)
    assert "testosterone" in j["choices"][0]["message"]["content"].lower()
    # Context Runtime metadata rides along: strategy, judged score, reward, citations.
    cr = j["context_runtime"]
    assert cr["strategy"] and 0.0 <= cr["retrieval_score"] <= 1.0
    assert cr["citations"], "expected retrieval citations"


def test_v1_chat_completion_streams_openai_deltas():
    _ingest_small_corpus()
    r = client.post("/v1/chat/completions", json={
        "stream": True, "messages": [{"role": "user", "content": "lipid cholesterol panel"}]})
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
    assert lines and lines[-1].strip() == "data: [DONE]"
    # first chunk carries an assistant content delta
    import json
    first = json.loads(lines[0][len("data:"):])
    assert first["object"] == "chat.completion.chunk"
    assert "content" in first["choices"][0]["delta"]


def test_shim_chat_is_self_learning():
    _ingest_small_corpus()
    req = "steroid profile hormone results"
    for _ in range(12):
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": req}]})
    from context_runtime.control_plane.app import _drain_learning
    _drain_learning()   # learning is now off the response path — wait for it before asserting
    assert client.get("/librechat/policy").json()["policy"], "the shim should learn a retrieval policy"
