"""Ingestion is limited to an explicitly configured corpus root.

The behaviour fixed here: POST /librechat/ingest accepted any server-side directory and
/librechat/retrieve then returned text from it. Now no corpus root means ingestion is disabled, a
corpus is named by identifier, and traversal, absolute paths, encoded or Windows-style separators
and symlinks that leave the corpus are refused. Synthetic data only.
"""
from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from context_runtime.control_plane import app as A

KEY = "ingest-test-control-plane-key"
SENTENCE = "The synthetic project codename is BLUE-HERON-4471 and it is not part of any served corpus."
INSIDE = "The served corpus mentions the synthetic codename AMBER-FINCH-2208 only here."


@pytest.fixture()
def world(tmp_path, monkeypatch):
    """root/served (the corpus), outside/ (a directory that is NOT under the root)."""
    root = tmp_path / "corpus-root"
    served = root / "served"
    served.mkdir(parents=True)
    (served / "note.md").write_text(f"# Note\n\n{INSIDE}\n")
    (served / "sub").mkdir()
    (served / "sub" / "nested.txt").write_text("nested synthetic text KESTREL-77")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "synthetic-note.md").write_text(f"# Synthetic note\n\n{SENTENCE}\n")
    (outside / "b.txt").write_text("synthetic text")
    monkeypatch.setenv("CONTEXT_RUNTIME_API_KEY", KEY)
    monkeypatch.setenv("CONTEXT_RUNTIME_CORPUS_ROOT", str(root))
    monkeypatch.delenv("CR_TENANT_KEYS", raising=False)
    tenant = A._TENANTS[A._DEFAULT_TENANT_ID]
    saved = list(tenant.retriever.docs)
    yield type("W", (), {"root": root, "served": served, "outside": outside, "tmp": tmp_path})
    tenant.retriever.docs[:] = saved
    tenant.retriever.__dict__.pop("_bm25_cache", None)


@pytest.fixture()
def client():
    return TestClient(A.app, headers={"X-API-Key": KEY})


def _ingest(client, **body):
    return client.post("/librechat/ingest", json=body)


def _retrieved_text(client, query="What is the synthetic project codename BLUE-HERON-4471 AMBER-FINCH-2208 KESTREL-77?") -> str:
    r = client.post("/librechat/retrieve", json={"request": query})
    assert r.status_code == 200
    body = r.json()                                  # the answer, not the echoed question
    return body["context"] + " ".join(h["text"] for h in body["hits"])


# ───────────────────────────── the demonstrated issue ─────────────────────────────

def test_a_directory_outside_the_corpus_root_is_not_indexed_or_retrievable(world, client):
    """Exactly the demonstrated sequence: ingest an absolute directory outside any served corpus, then retrieve."""
    r = _ingest(client, path=str(world.outside))
    assert r.status_code == 400
    assert "BLUE-HERON-4471" not in _retrieved_text(client)


def test_same_with_the_corpus_field(world, client):
    assert _ingest(client, corpus=str(world.outside)).status_code == 400
    assert "BLUE-HERON-4471" not in _retrieved_text(client)


def test_a_corpus_inside_the_root_is_ingested_and_retrievable(world, client):
    r = _ingest(client, corpus="served")
    assert r.status_code == 200
    assert r.json()["ingested"]["files"] == 2
    assert "AMBER-FINCH-2208" in _retrieved_text(client)


def test_older_clients_may_send_a_path_relative_to_the_root(world, client):
    assert _ingest(client, path="served").status_code == 200
    assert _ingest(client, path="served/sub").status_code == 200


# ───────────────────────────── fail closed ─────────────────────────────

def test_without_a_corpus_root_ingestion_is_disabled(world, client, monkeypatch):
    monkeypatch.delenv("CONTEXT_RUNTIME_CORPUS_ROOT")
    r = _ingest(client, corpus="served")
    assert r.status_code == 503 and "disabled" in r.json()["detail"]
    assert _ingest(client, path=str(world.outside)).status_code == 503
    monkeypatch.setenv("CONTEXT_RUNTIME_CORPUS_ROOT", "   ")
    assert _ingest(client, corpus="served").status_code == 503


def test_a_corpus_root_that_is_not_a_directory_disables_ingestion(world, client, monkeypatch):
    monkeypatch.setenv("CONTEXT_RUNTIME_CORPUS_ROOT", str(world.root / "missing"))
    assert _ingest(client, corpus="served").status_code == 503
    monkeypatch.setenv("CONTEXT_RUNTIME_CORPUS_ROOT", str(world.served / "note.md"))
    assert _ingest(client, corpus="x").status_code == 503


def test_ingest_needs_the_control_plane_key(world):
    anonymous = TestClient(A.app)
    assert anonymous.post("/librechat/ingest", json={"corpus": "served"}).status_code == 401
    assert anonymous.post("/librechat/ingest", json={"corpus": "served"},
                          headers={"X-API-Key": "wrong"}).status_code == 401


def test_a_request_without_a_reference_is_400(world, client):
    assert _ingest(client).status_code == 400
    assert _ingest(client, corpus="").status_code == 400


# ───────────────────────────── traversal and encodings ─────────────────────────────

BAD_REFERENCES = [
    "..", ".", "../outside", "served/../../outside", "served/..", "./served", "served/./sub",
    "/etc", "/", "//served", "/tmp", "served//sub",
    "%2e%2e/outside", "%2E%2E%2Foutside", "..%2foutside", "served%2f..%2f..%2foutside", "%252e%252e/outside",
    "served%00", "served\x00", "served\n", "served\t",
    "..\\outside", "served\\..\\..\\outside", "C:\\Windows", "C:/Windows", "\\\\host\\share", "c:",
    ".hidden", "served/.hidden", "served.", "a b", "a;b", "a:b", "caf\u00e9",
    "x" * 600,
]


@pytest.mark.parametrize("ref", BAD_REFERENCES)
def test_malformed_or_hostile_references_are_refused(world, client, ref):
    for field in ("corpus", "path"):
        r = _ingest(client, **{field: ref})
        assert r.status_code in (400, 404), (field, ref, r.status_code)
    assert "BLUE-HERON-4471" not in _retrieved_text(client)


def test_the_root_itself_is_not_a_corpus(world, client):
    assert _ingest(client, corpus=".").status_code == 400
    assert _ingest(client, path=str(world.root)).status_code == 400


def test_a_nested_path_is_not_a_corpus_identifier(world, client):
    assert _ingest(client, corpus="served/sub").status_code == 400
    assert _ingest(client, path="served/sub").status_code == 200


def test_an_unknown_corpus_and_a_file_are_404(world, client):
    assert _ingest(client, corpus="nope").status_code == 404
    assert _ingest(client, path="served/note.md").status_code == 404


# ───────────────────────────── symlinks ─────────────────────────────

pytestmark_symlink = pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "symlink"), reason="needs symlinks")


@pytestmark_symlink
def test_a_symlinked_directory_that_leaves_the_root_is_refused(world, client):
    os.symlink(world.outside, world.root / "linked-outside")
    assert _ingest(client, corpus="linked-outside").status_code == 404
    assert _ingest(client, path="linked-outside").status_code == 404
    assert "BLUE-HERON-4471" not in _retrieved_text(client)


@pytestmark_symlink
def test_a_symlinked_file_inside_a_corpus_pointing_outside_is_refused(world, client):
    os.symlink(world.outside / "synthetic-note.md", world.served / "leak.md")
    assert _ingest(client, corpus="served").status_code == 404
    assert "BLUE-HERON-4471" not in _retrieved_text(client)


@pytestmark_symlink
def test_a_symlinked_directory_inside_a_corpus_pointing_outside_is_refused(world, client):
    os.symlink(world.outside, world.served / "sub" / "escape")
    assert _ingest(client, corpus="served").status_code == 404
    assert "BLUE-HERON-4471" not in _retrieved_text(client)


@pytestmark_symlink
def test_a_dangling_symlink_is_refused(world, client):
    os.symlink(world.tmp / "does-not-exist", world.served / "dangling.md")
    assert _ingest(client, corpus="served").status_code == 404


@pytestmark_symlink
def test_a_symlink_to_another_corpus_in_the_root_is_refused(world, client):
    other = world.root / "other"
    other.mkdir()
    (other / "x.md").write_text("another corpus under the same root")
    os.symlink(other, world.served / "other-link")
    assert _ingest(client, corpus="served").status_code == 404


@pytestmark_symlink
def test_a_symlink_that_stays_inside_the_corpus_is_allowed(world, client):
    os.symlink(world.served / "note.md", world.served / "sub" / "alias.md")
    assert _ingest(client, corpus="served").status_code == 200


@pytestmark_symlink
def test_a_symlink_to_the_root_is_refused(world, client):
    os.symlink(world.root, world.served / "up")
    assert _ingest(client, corpus="served").status_code == 404


# ───────────────────────────── unit level ─────────────────────────────

def test_resolve_corpus_dir_returns_the_resolved_directory(world):
    from context_runtime.control_plane.security import resolve_corpus_dir
    got = resolve_corpus_dir("served", None)
    assert got == world.served.resolve()
    assert resolve_corpus_dir(None, "served/sub/") == (world.served / "sub").resolve()
