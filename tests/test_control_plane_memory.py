"""The narrow /memory/* routes: a thin HTTP pass-through to TemporalStore, for an external
consumer (e.g. DeerFlow) that should never need to import this package directly."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from context_runtime.control_plane import app as A
from context_runtime.control_plane import memory_store

KEY = "memory-test-control-plane-key"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", KEY)
    monkeypatch.setenv("CONTEXT_RUNTIME_HOME", str(tmp_path))
    memory_store.reset_for_tests()
    with TestClient(A.app) as c:
        yield c
    memory_store.reset_for_tests()


def _h(key: str | None = KEY) -> dict:
    return {"X-API-Key": key} if key else {}


def test_write_requires_admin_key(client):
    body = {"namespace": "t1", "subject": "svc", "predicate": "owned_by", "object": "team-a"}
    assert client.post("/memory/fact", json=body).status_code == 401
    r = client.post("/memory/fact", json=body, headers=_h())
    assert r.status_code == 200
    assert r.json() == {"subject": "svc", "predicate": "owned_by", "object": "team-a",
                        "valid_from": "", "valid_to": None, "recorded_at": "", "meta": {}}


def test_write_then_read_roundtrip_with_provenance(client):
    body = {"namespace": "t1", "subject": "svc", "predicate": "owned_by", "object": "team-a",
            "valid_from": "2026-01-01", "recorded_at": "2026-01-01",
            "meta": {"source": "slack-export", "confidence": 0.9}}
    client.post("/memory/fact", json=body, headers=_h())
    r = client.get("/memory/fact", params={"namespace": "t1", "query": "svc owned"}, headers=_h())
    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 1
    assert hits[0]["meta"]["subject"] == "svc"
    assert hits[0]["meta"]["source"] == "slack-export"   # provenance survives the round trip
    assert hits[0]["meta"]["confidence"] == 0.9


def test_as_of_point_in_time_query(client):
    ns = "t2"
    client.post("/memory/fact", json={"namespace": ns, "subject": "svc", "predicate": "owner",
                                       "object": "alice", "valid_from": "2026-01-01",
                                       "valid_to": "2026-06-01"}, headers=_h())
    client.post("/memory/fact", json={"namespace": ns, "subject": "svc", "predicate": "owner",
                                       "object": "bob", "valid_from": "2026-06-01"}, headers=_h())
    before = client.get("/memory/fact", params={"namespace": ns, "query": "svc owner", "at": "2026-03-01"}, headers=_h()).json()
    after = client.get("/memory/fact", params={"namespace": ns, "query": "svc owner", "at": "2026-07-01"}, headers=_h()).json()
    assert before[0]["meta"]["object"] == "alice"
    assert after[0]["meta"]["object"] == "bob"


def test_changes_since_until(client):
    ns = "t3"
    client.post("/memory/fact", json={"namespace": ns, "subject": "svc", "predicate": "status",
                                       "object": "degraded", "valid_from": "2026-02-01"}, headers=_h())
    changes = client.get("/memory/changes", params={"namespace": ns, "since": "2026-01-01",
                                                      "until": "2026-03-01"}, headers=_h()).json()
    assert any(c.get("subject") == "svc" for c in changes)


def test_namespaces_are_isolated(client):
    client.post("/memory/fact", json={"namespace": "ns-a", "subject": "x", "predicate": "y",
                                       "object": "z"}, headers=_h())
    r = client.get("/memory/fact", params={"namespace": "ns-b", "query": "x y"}, headers=_h())
    assert r.json() == []


def test_persists_across_process_cache_reset(client, tmp_path):
    client.post("/memory/fact", json={"namespace": "durable", "subject": "svc", "predicate": "owner",
                                       "object": "carol"}, headers=_h())
    memory_store.reset_for_tests()   # drops the in-process cache; disk file remains
    r = client.get("/memory/fact", params={"namespace": "durable", "query": "svc owner"}, headers=_h())
    assert r.json()[0]["meta"]["object"] == "carol"
