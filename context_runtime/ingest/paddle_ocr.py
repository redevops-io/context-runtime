"""PaddleOCRExtractor — OCR + table recognition via PaddleOCR (Apache-2.0).

    pip install "context_runtime[ocr]"

Two capabilities beyond the default rapidocr backend:
  * PP-OCR      — multilingual text OCR (lang via CR_OCR_LANG, default "en").
  * PP-Structure — layout + TABLE recognition. Dense lab-value tables (the honest
    retrieval-quality gap on the medical corpus) survive as pipe tables instead of a
    flattened word-soup, so a query like "ферритин" can match the row it lives in.

Everything is lazy-imported and guarded: absent the [ocr] extra, `available` is False
and the ingest pipeline simply uses another extractor. PaddleOCR's API has drifted
across releases, so each entry point is tried defensively and degrades to plain OCR.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from ..types import PluginInfo, RawAsset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def _html_table_to_text(html: str) -> str:
    """Flatten a PP-Structure table (HTML) into a markdown-ish pipe table — searchable
    and keeps each lab row on one line. Regex-only (no bs4 dependency in the [ocr] extra)."""
    rows = re.findall(r"<tr>(.*?)</tr>", html, flags=re.S | re.I)
    lines = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


class PaddleOCRExtractor:
    def __init__(self, lang: str | None = None, use_tables: bool = True):
        self.lang = lang or os.getenv("CR_OCR_LANG", "en")
        self.use_tables = use_tables
        self._ocr = None
        self._structure = None
        self._tried = False

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        except Exception:
            self._ocr = None
        if self.use_tables:
            try:
                from paddleocr import PPStructure
                self._structure = PPStructure(show_log=False, lang="en")
            except Exception:
                self._structure = None

    @property
    def available(self) -> bool:
        self._load()
        return self._ocr is not None

    def supports(self, asset: RawAsset) -> bool:
        ext = Path(asset.label or asset.uri or "").suffix.lower()
        return ext in _IMAGE_EXTS or (asset.mime or "").startswith("image/")

    def _path_of(self, asset: RawAsset) -> tuple[str, bool]:
        if asset.uri and os.path.exists(asset.uri):
            return asset.uri, False
        suffix = Path(asset.label or "img.png").suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(asset.data or b"")
        tmp.close()
        return tmp.name, True

    def _ocr_text(self, path: str) -> str:
        try:
            result = self._ocr.ocr(path, cls=True)
        except Exception:
            return ""
        lines = []
        for page in result or []:
            for entry in page or []:
                try:
                    lines.append(entry[1][0])  # [box, (text, conf)]
                except Exception:
                    continue
        return "\n".join(lines)

    def _table_text(self, path: str) -> str:
        if not self._structure:
            return ""
        try:
            regions = self._structure(path)
        except Exception:
            return ""
        out = []
        for r in regions or []:
            if r.get("type") == "table":
                html = (r.get("res") or {}).get("html", "")
                if html:
                    out.append(_html_table_to_text(html))
        return "\n\n".join(out)

    def extract(self, asset: RawAsset) -> tuple[str, str]:
        self._load()
        if self._ocr is None:
            return "", "empty"
        path, tmp = self._path_of(asset)
        try:
            tables = self._table_text(path) if self.use_tables else ""
            text = self._ocr_text(path)
            merged = "\n\n".join(p for p in (tables, text) if p.strip())
            return merged.strip(), ("table" if tables else "image")
        finally:
            if tmp:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def info(self) -> PluginInfo:
        return PluginInfo(name="paddleocr_extractor", kind="extractor", version="0.1",
                          capabilities=frozenset({"image", "ocr", "table", "multilingual"}))
