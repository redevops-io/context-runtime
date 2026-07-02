"""Ingestion plugin surface: SourcePlugin → QualityPlugin → ExtractorPlugin → corpus.

The core path (local folder + multimodal text + heuristic quality) is dependency-free
and always runs; the dlt connector is exercised only when the [connectors] extra is
installed, so default CI stays green.
"""
from __future__ import annotations

import importlib.util

import pytest

from context_runtime.ingest.extractors import MultimodalExtractor
from context_runtime.ingest.pipeline import ingest_corpus
from context_runtime.ingest.quality import HeuristicQuality, LLMQuality
from context_runtime.sources.local import LocalFolderSource
from context_runtime.types import RawAsset


def _write(dirpath, name, text):
    p = dirpath / name
    p.write_text(text, encoding="utf-8")
    return p


def test_local_folder_source_yields_assets(tmp_path):
    _write(tmp_path, "a.txt", "steroid profile testosterone cortisol dhea")
    _write(tmp_path, "b.md", "lipid profile cholesterol ldl hdl")
    assets = list(LocalFolderSource(str(tmp_path)).read())
    assert {a.label for a in assets} == {"a.txt", "b.md"}
    assert all(a.uri and a.id for a in assets)
    info = LocalFolderSource(str(tmp_path)).info()
    assert info.kind == "source"


def test_pipeline_builds_corpus_from_plugins(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _write(src_dir, "a.txt", "steroid profile testosterone cortisol dhea androstenedione")
    _write(src_dir, "b.txt", "lipid profile cholesterol ldl hdl triglycerides")
    out = tmp_path / "corpus"
    stats = ingest_corpus(LocalFolderSource(str(src_dir)), str(out),
                          extractor=MultimodalExtractor(), quality=HeuristicQuality())
    assert stats.written == 2
    txts = sorted(p.name for p in out.glob("*.txt"))
    assert len(txts) == 2
    # manifest lives BESIDE the corpus dir so a folder-index ingests only the .txt docs
    assert (out.parent / "corpus.manifest.jsonl").exists()
    body = (out / txts[0]).read_text(encoding="utf-8")
    assert body.startswith("[source:")


def test_quality_dedups_and_drops_short(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _write(src_dir, "a.txt", "steroid profile testosterone cortisol dhea")
    _write(src_dir, "dup.txt", "steroid profile testosterone cortisol dhea")  # identical → deduped
    _write(src_dir, "tiny.txt", "hi")  # below min_chars → dropped
    out = tmp_path / "corpus"
    stats = ingest_corpus(LocalFolderSource(str(src_dir)), str(out), quality=HeuristicQuality())
    assert stats.written == 1
    assert stats.dropped_quality == 1   # the duplicate
    assert stats.skipped_empty == 1     # "hi" is < _MIN_CHARS at extraction


def test_llm_quality_is_fail_open():
    class BoomModel:
        def complete(self, req):
            raise RuntimeError("upstream down")
    q = LLMQuality(BoomModel(), mode="clean")
    text = "steroid profile testosterone cortisol dhea androstenedione panel"
    assert q.review(text, RawAsset(id="x")) == text  # error → original text, never drops good data


def test_build_corpus_still_works(tmp_path):
    from context_runtime.ingest.multimodal import build_corpus
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    _write(src_dir, "a.txt", "steroid profile testosterone cortisol dhea androstenedione")
    stats = build_corpus([str(src_dir)], str(tmp_path / "corpus"))
    assert stats.written == 1
    assert "dropped_quality" in stats.as_dict()


@pytest.mark.skipif(importlib.util.find_spec("dlt") is None, reason="dlt extra not installed")
def test_dlt_source_maps_records():
    from context_runtime.sources.dlt_source import DltSource
    records = [
        {"id": 1, "title": "Steroid panel", "body": "testosterone cortisol dhea"},
        {"id": 2, "title": "Lipid panel", "body": "cholesterol ldl hdl"},
    ]
    assets = list(DltSource(records, text_fields=["title", "body"], id_field="id").read())
    assert [a.id for a in assets] == ["1", "2"]
    assert "testosterone" in assets[0].text
    assert assets[0].mime == "application/json"


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="ocr extra not installed")
def test_paddleocr_extractor_available():
    from context_runtime.ingest.paddle_ocr import PaddleOCRExtractor
    ext = PaddleOCRExtractor()
    assert ext.info().kind == "extractor"
    assert ext.supports(RawAsset(id="x", label="scan.png"))
    assert not ext.supports(RawAsset(id="y", label="notes.txt"))
