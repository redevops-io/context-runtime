"""Deterministic corpus enumeration (cross-OS golden output) and junction / reparse-point refusal.

The golden files under ``tests/golden`` are committed expected values. The same tests run on the Linux,
macOS and Windows CI lanes and compare *bytes*, so a passing run on all three is the evidence that the
enumeration, labels, chunk ids and BM25 ranking are identical across operating systems.
Regenerate after an intended change with ``CR_UPDATE_GOLDEN=1 pytest tests/test_corpus_files.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from context_runtime.control_plane import security
from context_runtime.control_plane.security import CorpusError, resolve_corpus_dir
from context_runtime.ingest import corpus_files
from context_runtime.ingest.corpus_files import iter_corpus_files

GOLDEN = Path(__file__).resolve().parent / "golden"
MANIFEST = json.loads((GOLDEN / "corpus_manifest.json").read_text(encoding="utf-8"))
UPDATE = os.environ.get("CR_UPDATE_GOLDEN") == "1"
QUERIES = ["revenue", "alpha", "incident outage", "café", "umlaut", "japanese", "hidden"]
TEXT = (".md", ".txt", ".rst")


def _materialise(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in MANIFEST["text_files"].items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))                   # bytes: no newline translation on any OS
    for rel, hexed in MANIFEST["binary_files_hex"].items():
        (root / rel).write_bytes(bytes.fromhex(hexed))
    return root


def _canonical(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _check_golden(name: str, obj) -> None:
    data = _canonical(obj)
    path = GOLDEN / name
    if UPDATE:
        path.write_bytes(data)
    assert path.read_bytes() == data, f"{name} differs from the committed golden output"


@pytest.fixture()
def corpus(tmp_path):
    return _materialise(tmp_path / "corpus")


# ───────────────────────── ordering, labels, filtering ─────────────────────────

def test_golden_listing_is_byte_identical_on_every_os(corpus):
    listing = {
        "all": [c.rel for c in iter_corpus_files(corpus)],
        "text": [c.rel for c in iter_corpus_files(corpus, TEXT)],
        "names": [c.name for c in iter_corpus_files(corpus, TEXT)],
    }
    _check_golden("corpus_listing.json", listing)


def test_order_is_codepoint_order_by_component_not_filesystem_order(corpus):
    rels = [c.rel for c in iter_corpus_files(corpus)]
    assert rels == sorted(rels, key=lambda r: tuple(r.split("/")))        # component-wise codepoint order
    assert rels.index("B.md") < rels.index("C.md") < rels.index("a.md")   # upper case sorts before lower case
    assert rels.index("a/x.md") < rels.index("a-b.md") < rels.index("a.md")   # the directory a before a-b, a.md


def test_labels_are_nfc_and_posix(corpus):
    cfs = list(iter_corpus_files(corpus))
    assert all("\\" not in c.rel for c in cfs)
    assert all(c.rel == unicodedata.normalize("NFC", c.rel) for c in cfs)
    nfd_made = next(c for c in cfs if c.rel == "café.md")          # created decomposed, labelled composed
    assert nfd_made.path.read_text(encoding="utf-8").startswith("café")
    assert "dir with space/file name.md" in {c.rel for c in cfs}


def test_suffix_filter_is_case_insensitive_and_skips_other_types(corpus):
    rels = {c.rel for c in iter_corpus_files(corpus, TEXT)}
    assert "UPPER.TXT" in rels and "image.png" not in rels and "notes.bin" not in rels
    assert {c.rel for c in iter_corpus_files(corpus, (".png",))} == {"image.png"}


def test_a_single_file_root_and_a_missing_root(corpus, tmp_path):
    one = list(iter_corpus_files(corpus / "B.md"))
    assert [c.rel for c in one] == ["B.md"] and one[0].path == corpus / "B.md"
    assert list(iter_corpus_files(tmp_path / "does-not-exist")) == []


def test_every_directory_is_visited_in_place(corpus):
    rels = [c.rel for c in iter_corpus_files(corpus)]
    assert "alpha/nested/deep.rst" in rels and "mixed/lines.md" in rels and ".hidden.md" in rels


# ───────────────────────── golden outputs of the stores and sources ─────────────────────────

def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_golden_inmemory_store_index_and_ranking(corpus):
    from context_runtime.adapters.store_inmemory import InMemoryStore
    store = InMemoryStore()
    report = store.index(str(corpus))
    out = {
        "report": report,
        "docs": [{"chunk_id": d["chunk_id"], "filename": d["filename"], "sha256": _digest(d["text"])}
                 for d in store.docs],
        "search": {q: [[h.chunk_id, round(h.score, 6)] for h in store.search(q, 5, "bm25")] for q in QUERIES},
    }
    _check_golden("inmemory_store.json", out)


def test_golden_local_folder_source_labels(corpus):
    from context_runtime.sources.local import LocalFolderSource
    assets = list(LocalFolderSource(str(corpus)).read())
    out = [{"id": a.id, "label": a.label,
            "sha256": hashlib.sha256(Path(a.uri).read_bytes()).hexdigest()} for a in assets]
    _check_golden("local_folder_source.json", out)


def test_other_text_stores_use_the_same_files_and_order(corpus):
    from context_runtime.adapters.store_community import CommunityRetriever
    from context_runtime.adapters.store_inmemory import InMemoryStore
    ref = InMemoryStore()
    ref.index(str(corpus))
    expected = [d["chunk_id"] for d in ref.docs]
    com = CommunityRetriever()
    com.index(str(corpus))
    assert [d["chunk_id"] for d in com.docs] == expected


def test_golden_duckdb_store_without_the_extension(corpus, tmp_path):
    pytest.importorskip("duckdb")
    from context_runtime.adapters.store_duckdb import DuckDBStore
    store = DuckDBStore(fts_extension=str(tmp_path / "absent.duckdb_extension"))   # in-memory BM25 path
    report = store.index(str(corpus))
    out = {"report": report,
           "search": {q: [h.chunk_id for h in store.search(q, 5, "bm25")] for q in QUERIES}}
    _check_golden("duckdb_store_fallback.json", out)


def test_agent_harness_index_files_registers_files_in_the_same_order(corpus, tmp_path):
    from context_runtime.integrations.agent_harness import ContextRuntimeManager
    mgr = ContextRuntimeManager(archive_dir=str(tmp_path / "arch"))
    n = mgr.index_files(corpus)
    tools = [e.tool for e in mgr.archive.entries.values()]
    expected = [c.name for c in iter_corpus_files(corpus, (".txt", ".md", ".csv", ".json", ".rst"))]
    assert n == len(expected) and tools == expected


# ───────────────────────── junctions and reparse points ─────────────────────────

def test_a_directory_that_looks_like_a_junction_is_not_entered(corpus, monkeypatch):
    """Simulated on every OS: the scan skips whatever the junction predicate flags, and says so."""
    real = corpus_files.is_junction
    monkeypatch.setattr(corpus_files, "is_junction", lambda p: Path(p).name == "alpha" or real(p))
    rels = [c.rel for c in iter_corpus_files(corpus)]
    assert "alpha/nested/deep.rst" not in rels and "a/x.md" in rels


def test_a_directory_with_the_reparse_attribute_is_not_entered(corpus, monkeypatch):
    monkeypatch.setattr(corpus_files, "is_reparse_point", lambda p: Path(p).name == "Zeta")
    rels = [c.rel for c in iter_corpus_files(corpus)]
    assert "Zeta/z.md" not in rels and "a/x.md" in rels


def test_ingestion_refuses_a_junction_inside_a_corpus_even_when_it_points_inside(tmp_path, monkeypatch):
    root = tmp_path / "root"
    served = _materialise(root / "served")
    monkeypatch.setattr(security.corpus_files, "is_junction", lambda p: Path(p).name == "alpha")
    with pytest.raises(CorpusError) as e:
        resolve_corpus_dir("served", None, str(root))
    assert e.value.status == 404 and served.is_dir()


def test_ingestion_refuses_a_junction_as_the_corpus_directory(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "plain").mkdir(parents=True)
    (root / "plain" / "a.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(security.corpus_files, "is_junction", lambda p: Path(p).name == "plain")
    with pytest.raises(CorpusError) as e:
        resolve_corpus_dir("plain", None, str(root))
    assert e.value.status == 404


def test_ingestion_still_accepts_an_ordinary_corpus(tmp_path):
    root = tmp_path / "root"
    _materialise(root / "served")
    assert resolve_corpus_dir("served", None, str(root)) == (root / "served").resolve()


def _mklink_junction(link: Path, target: Path) -> None:
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="NTFS junctions only exist on Windows")


@windows_only
def test_windows_junction_is_a_junction_not_a_symlink_and_is_not_entered(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("outside text", encoding="utf-8")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ok.md").write_text("inside text", encoding="utf-8")
    _mklink_junction(corpus / "jn", outside)
    assert (corpus / "jn").is_dir() and not (corpus / "jn").is_symlink()     # why is_symlink() is not enough
    assert corpus_files.is_junction(corpus / "jn")
    assert [c.rel for c in iter_corpus_files(corpus)] == ["ok.md"]


@windows_only
def test_windows_ingestion_refuses_a_junction_leading_outside_the_corpus(tmp_path):
    root = tmp_path / "root"
    served = root / "served"
    served.mkdir(parents=True)
    (served / "ok.md").write_text("inside text", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("outside text", encoding="utf-8")
    _mklink_junction(served / "jn", outside)
    with pytest.raises(CorpusError) as e:
        resolve_corpus_dir("served", None, str(root))
    assert e.value.status == 404


@windows_only
def test_windows_ingestion_refuses_a_corpus_that_is_itself_a_junction(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("outside text", encoding="utf-8")
    _mklink_junction(root / "linked", outside)
    with pytest.raises(CorpusError) as e:
        resolve_corpus_dir("linked", None, str(root))
    assert e.value.status == 404


@windows_only
def test_windows_ingestion_refuses_a_junction_that_stays_inside_the_root(tmp_path):
    root = tmp_path / "root"
    (root / "served").mkdir(parents=True)
    (root / "served" / "ok.md").write_text("inside text", encoding="utf-8")
    _mklink_junction(root / "alias", root / "served")
    with pytest.raises(CorpusError) as e:
        resolve_corpus_dir("alias", None, str(root))
    assert e.value.status == 404
