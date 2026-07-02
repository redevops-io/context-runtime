"""ExtractorPlugins — the parse half of ingestion (SPEC §4.8).

An extractor turns a RawAsset (a file path, or raw bytes, or an already-textual
record) into normalized text. MultimodalExtractor is the reference implementation and
wraps the existing extract_text() backends (pypdf/docx/html/rapidocr/whisper). It works
on RawAssets that carry a `uri` path, raw `data` bytes, or inline `text`.

PaddleOCRExtractor lives in paddle_ocr.py (optional [ocr] extra) and adds PP-Structure
table recognition — the piece that helps dense lab-value tables survive ingestion.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..types import PluginInfo, RawAsset
from .multimodal import Extractors, extract_text

# best-effort suffix inference so raw bytes route to the right extract_text backend
_MIME_SUFFIX = {
    "application/pdf": ".pdf", "text/html": ".html", "text/plain": ".txt",
    "image/png": ".png", "image/jpeg": ".jpg", "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a", "audio/wav": ".wav", "video/mp4": ".mp4",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


def _suffix_for(asset: RawAsset) -> str:
    for cand in (asset.label, asset.uri):
        if cand and Path(cand).suffix:
            return Path(cand).suffix.lower()
    return _MIME_SUFFIX.get((asset.mime or "").lower(), ".bin")


class MultimodalExtractor:
    """Reference extractor over the multimodal backends. Reuses one lazy Extractors
    instance so OCR/ASR models load at most once across a whole ingest run."""

    def __init__(self, ex: Extractors | None = None):
        self._ex = ex or Extractors()

    def supports(self, asset: RawAsset) -> bool:
        return True  # inline text, a path, or bytes all route through extract_text

    def extract(self, asset: RawAsset) -> tuple[str, str]:
        if asset.text is not None:
            return asset.text.strip(), "text"
        if asset.uri and os.path.exists(asset.uri):
            return extract_text(asset.uri, self._ex)
        if asset.data is not None:
            with tempfile.NamedTemporaryFile(suffix=_suffix_for(asset), delete=True) as fh:
                fh.write(asset.data)
                fh.flush()
                return extract_text(fh.name, self._ex)
        return "", "empty"

    def info(self) -> PluginInfo:
        return PluginInfo(name="multimodal_extractor", kind="extractor", version="0.1",
                          capabilities=frozenset({"pdf", "docx", "html", "image", "audio", "video", "text"}))
