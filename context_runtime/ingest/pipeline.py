"""ingest_corpus — the pluggable ingestion pipeline (SPEC §4.8).

    SourcePlugin.read()  →  [QualityPlugin.review()]  →  ExtractorPlugin.extract()
                         →  chunk_text()  →  <out_dir>/<id>.txt  (+ manifest.jsonl)

This is the plugin form of what build_corpus() did inline. Any SourcePlugin (local
folder, dlt connector), any ExtractorPlugin (multimodal, PaddleOCR), and an optional
QualityPlugin compose here, and the output corpus is byte-for-byte what the Python
InMemoryStore and the Go control-plane both index — so ingestion became first-class and
swappable without changing the retrieval side.
"""
from __future__ import annotations

import json
from pathlib import Path

from .multimodal import _MIN_CHARS, CorpusStats, _safe_id, chunk_text


def ingest_corpus(source, out_dir: str, *, extractor=None, quality=None,
                  chunk_chars: int = 900, verbose: bool = False,
                  availability: dict | None = None) -> CorpusStats:
    """Run one source through the pipeline into a normalized text corpus.

    source:    a SourcePlugin (has .read() -> Iterable[RawAsset])
    extractor: an ExtractorPlugin (default: MultimodalExtractor)
    quality:   an optional QualityPlugin (clean/reject before indexing)
    """
    if extractor is None:
        from .extractors import MultimodalExtractor
        extractor = MultimodalExtractor()

    stats = CorpusStats(out_dir=out_dir, availability=availability or {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # manifest lives BESIDE the corpus dir so a folder-index ingests only the .txt docs.
    manifest = (out.parent / f"{out.name}.manifest.jsonl").open("w", encoding="utf-8")

    n = 0
    parquet_rows: list[dict] = []   # collected for a columnar corpus.parquet (fast bulk-load)
    for asset in source.read():
        text, kind = extractor.extract(asset)
        if kind == "unsupported":
            continue
        stats.by_kind[kind] = stats.by_kind.get(kind, 0) + 1
        if len(text) < _MIN_CHARS:
            stats.skipped_empty += 1
            if verbose:
                print(f"  skip (empty {kind}): {asset.label}")
            continue
        if quality is not None:
            reviewed = quality.review(text, asset)
            if reviewed is None:
                stats.dropped_quality += 1
                if verbose:
                    print(f"  drop (quality): {asset.label}")
                continue
            text = reviewed
        if kind in ("image", "table"):
            stats.ocr_used += 1
        elif kind in ("audio", "video"):
            stats.asr_used += 1
        n += 1
        label = asset.label or asset.id
        doc_id = _safe_id(label, n)
        passages = chunk_text(text, chunk_chars)
        multi = len(passages) > 1
        for pi, passage in enumerate(passages):
            cid = f"{doc_id}_p{pi:02d}" if multi else doc_id
            tag = f" · passage {pi + 1}/{len(passages)}" if multi else ""
            header = f"[source: {label} · kind: {kind}{tag}]\n\n"
            (out / f"{cid}.txt").write_text(header + passage, encoding="utf-8")
            parquet_rows.append({"chunk_id": cid, "filename": f"{cid}.txt",
                                 "text": header + passage, "ts": 0.0})
            manifest.write(json.dumps({
                "id": cid, "source": label, "kind": kind, "passage": pi,
                "chars": len(passage), "path": asset.uri,
            }, ensure_ascii=False) + "\n")
            stats.written += 1
        stats.by_kind[kind + "_chunks"] = stats.by_kind.get(kind + "_chunks", 0) + len(passages)
        if verbose:
            print(f"  {kind:6} {len(text):>7} chars → {len(passages):>3} passage(s)  {label}")

    manifest.close()
    # Also emit a single columnar corpus.parquet for fast bulk-load (DuckDB read_parquet /
    # one-file ingest at scale). Best-effort: skipped cleanly if no parquet backend is present.
    if parquet_rows:
        try:
            from .parquet_corpus import PARQUET_NAME, parquet_available, write_corpus_parquet
            if parquet_available():
                write_corpus_parquet(parquet_rows, out / PARQUET_NAME)
                stats.by_kind["parquet_rows"] = len(parquet_rows)
        except Exception:
            pass   # the .txt corpus is always written; parquet is an optional accelerator
    return stats
