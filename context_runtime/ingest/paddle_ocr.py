"""PaddleOCRExtractor — OCR + table recognition via PaddleOCR (Apache-2.0).

    pip install "context_runtime[ocr]"

Two capabilities beyond the default rapidocr backend:
  * PP-OCR       — multilingual text OCR (lang via CR_OCR_LANG, default "en").
  * PP-Structure — layout + TABLE recognition. Dense lab-value tables (the honest
    retrieval-quality gap on the medical corpus) survive as pipe tables instead of a
    flattened word-soup, so a query like "ферритин" can match the row it lives in.

Provenance: PaddleOCR is Baidu's; the code + weights are Apache-2.0, but weights are
fetched from Baidu Cloud (bcebos.com) on first use — pre-stage the model dir for
air-gapped deploys. Inference is fully local (no API calls).

Version/robustness notes: PaddleOCR's API drifted between 2.x (``PaddleOCR(use_angle_cls
=...).ocr(path)`` / ``PPStructure``) and 3.x (``PaddleOCR(...).predict(path)`` returning
``rec_texts`` / ``PPStructureV3``). Both are tried. paddle 3.x's oneDNN kernels crash on
some CPUs (ConvertPirAttribute2RuntimeAttribute), so oneDNN is DISABLED by default;
set CR_OCR_MKLDNN=1 to re-enable where it works. Everything is lazy + guarded: absent
the [ocr] extra, ``available`` is False and the pipeline uses another extractor.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from ..types import PluginInfo, RawAsset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def _mkldnn() -> bool:
    return os.getenv("CR_OCR_MKLDNN", "0") == "1"


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
        self._predict = False           # True → 3.x predict() API, False → 2.x ocr()
        self._tried = False

    def _load(self):
        if self._tried:
            return
        self._tried = True
        try:
            from paddleocr import PaddleOCR
        except Exception:
            return
        # constructor kwargs drifted across versions — try newest first, degrade.
        for kwargs in (
            {"lang": self.lang, "enable_mkldnn": _mkldnn()},                 # 3.x
            {"lang": self.lang, "use_angle_cls": True, "show_log": False},   # 2.x
            {"lang": self.lang},
        ):
            try:
                self._ocr = PaddleOCR(**kwargs)
                break
            except Exception:
                self._ocr = None
        if self._ocr is None:
            return
        self._predict = hasattr(self._ocr, "predict")
        if self.use_tables:
            self._load_structure()

    def _load_structure(self):
        try:
            from paddleocr import PPStructureV3
            self._structure = PPStructureV3(enable_mkldnn=_mkldnn())
        except Exception:
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
            if self._predict:  # 3.x
                res = self._ocr.predict(path)
                lines: list[str] = []
                for r in res or []:
                    lines.extend((r.get("rec_texts") if hasattr(r, "get") else None) or [])
                return "\n".join(lines)
            result = self._ocr.ocr(path, cls=True)  # 2.x
            lines = []
            for page in result or []:
                for entry in page or []:
                    try:
                        lines.append(entry[1][0])  # [box, (text, conf)]
                    except Exception:
                        continue
            return "\n".join(lines)
        except Exception:
            return ""

    def _table_text(self, path: str) -> str:
        if not self._structure:
            return ""
        try:
            regions = (self._structure.predict(path)
                       if hasattr(self._structure, "predict") else self._structure(path))
        except Exception:
            return ""
        out = []
        for r in regions or []:
            html = ""
            if hasattr(r, "get"):
                res = r.get("res") or r
                html = (res.get("html") if hasattr(res, "get") else "") or ""
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
