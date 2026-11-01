"""File-name portability: Windows device names and component length (checked on every OS)."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from context_runtime.compression.transcript import Archive
from context_runtime.control_plane.security import CorpusError, corpus_segments
from context_runtime.ingest.extractors import MultimodalExtractor
from context_runtime.pathnames import MAX_COMPONENT, is_windows_reserved, portable_component
from context_runtime.policy.store import RuleStore, _safe
from context_runtime.types import RawAsset

RESERVED = ["con", "CON", "prn", "aux", "nul", "com0", "COM1", "COM9", "lpt0", "lpt1", "LPT9", "con.txt", "nul.tar.gz",
            "com¹", "conin$"]


@pytest.mark.parametrize("name", RESERVED)
def test_reserved_names_are_recognised(name):
    assert is_windows_reserved(name)


@pytest.mark.parametrize("name", ["console", "com", "com10", "lpt", "aux1", "a.con", "corpus-a", "nul_"])
def test_ordinary_names_are_not_reserved(name):
    assert not is_windows_reserved(name)


@pytest.mark.parametrize("name", [n for n in RESERVED if n.isascii() and "$" not in n])
def test_corpus_references_refuse_reserved_names(name):
    for ref in (name, f"ok/{name}"):
        with pytest.raises(CorpusError) as e:
            corpus_segments(ref)
        assert e.value.status == 400


def test_corpus_references_still_accept_ordinary_names():
    assert corpus_segments("finance/q3-2026") == ["finance", "q3-2026"]


@pytest.mark.parametrize("scope", ["con", "NUL", "com1", "lpt3"])
def test_policy_store_never_uses_a_device_name_as_a_file_name(scope, tmp_path):
    assert not is_windows_reserved(_safe(scope))
    store = RuleStore(str(tmp_path))
    rule = store.add(scope, "rule", "no external calls")
    assert store.get(rule.id, scope=scope) is not None
    for p in tmp_path.rglob("*"):
        assert not is_windows_reserved(p.name), p


def test_policy_store_caps_component_length_and_keeps_distinct_scopes_distinct(tmp_path):
    a, b = "x" * 300 + "a", "x" * 300 + "b"
    assert len(_safe(a)) <= MAX_COMPONENT and len(_safe(b)) <= MAX_COMPONENT
    assert _safe(a) != _safe(b)
    store = RuleStore(str(tmp_path))
    store.add(a, "rule", "one")
    store.add(b, "rule", "two")
    assert [r.text for r in store.list(scope=a)] == ["one"]
    assert [r.text for r in store.list(scope=b)] == ["two"]


def test_portable_component_leaves_short_ordinary_names_alone():
    assert portable_component("abc_def-1") == "abc_def-1"


def test_extractor_can_reopen_its_temporary_file_by_name():
    # On Windows a NamedTemporaryFile that is still open cannot be opened a second time by name.
    out, kind = MultimodalExtractor().extract(RawAsset(id="a", data=b"plain bytes asset", label="a.txt"))
    assert "plain bytes asset" in out and kind == "text"


def test_default_archive_directory_is_removed_when_it_is_released():
    arch = Archive()
    d = Path(arch.directory)
    assert d.is_dir()
    del arch
    gc.collect()
    assert not d.exists()


def test_explicit_archive_directory_is_kept(tmp_path):
    arch = Archive(tmp_path / "keep")
    del arch
    gc.collect()
    assert (tmp_path / "keep").is_dir()


def test_the_name_mapping_is_injective_for_reserved_names_and_long_names():
    # "con" and "con_" must stay two different files (an escape marker no ordinary name contains is used)
    assert _safe("con") != _safe("con_") and _safe("con") == "con~" and _safe("con_") == "con_"
    assert portable_component("nul.txt") == "nul~.txt"
    names = ["con", "con_", "nul", "nul_", "com0", "com0_", "lpt0", "x" * 200, "x" * 200 + "y", "x" * 128]
    assert len({_safe(n) for n in names}) == len(names)
    assert not any(is_windows_reserved(_safe(n)) for n in names)


def test_rules_for_con_and_con_underscore_are_separate(tmp_path):
    store = RuleStore(str(tmp_path))
    store.add("con", "rule", "for con")
    store.add("con_", "rule", "for con underscore")
    assert [r.text for r in store.list(scope="con")] == ["for con"]
    assert [r.text for r in store.list(scope="con_")] == ["for con underscore"]
