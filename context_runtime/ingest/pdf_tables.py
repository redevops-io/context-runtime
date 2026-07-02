"""PdfTableExtractor — text-layer PDF extractor that PRESERVES tables (SPEC §4.8).

    pip install "context_runtime[pdf-tables]"

Financial filings (10-K/10-Q), scientific papers, and reports keep their key facts in
TABLES. Naive text extraction (pypdf) flattens a table into a word-soup where "Capital
expenditures 1,577 1,373 1,577" loses the row/column structure — so a query like "FY2018
capex" can't reliably match the figure. This extractor uses pdfplumber to pull page text
AND detected tables, rendering each table as a pipe table inline so the row stays intact
and retrievable. For scanned/image PDFs use PaddleOCRExtractor instead (this needs a text
layer). Emitted kind is "pdf_table".
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..types import PluginInfo, RawAsset


def _table_to_pipes(table: list[list]) -> str:
    lines = []
    for row in table or []:
        cells = ["" if c is None else " ".join(str(c).split()) for c in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


class PdfTableExtractor:
    def __init__(self, with_tables: bool = True):
        self.with_tables = with_tables

    def supports(self, asset: RawAsset) -> bool:
        ext = Path(asset.label or asset.uri or "").suffix.lower()
        return ext == ".pdf" or (asset.mime or "") == "application/pdf"

    def _path_of(self, asset: RawAsset) -> tuple[str, bool]:
        if asset.uri and os.path.exists(asset.uri):
            return asset.uri, False
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(asset.data or b"")
        tmp.close()
        return tmp.name, True

    def extract(self, asset: RawAsset) -> tuple[str, str]:
        try:
            import pdfplumber
        except Exception:
            return "", "empty"
        path, tmp = self._path_of(asset)
        try:
            parts: list[str] = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    chunk = text
                    if self.with_tables:
                        try:
                            tables = page.extract_tables() or []
                        except Exception:
                            tables = []
                        rendered = "\n\n".join(t for t in (_table_to_pipes(tb) for tb in tables) if t)
                        if rendered:
                            chunk = (chunk + "\n\n[TABLE]\n" + rendered).strip()
                    if chunk.strip():
                        parts.append(chunk)
            return ("\n\n".join(parts).strip(), "pdf_table")
        except Exception:
            return "", "error"
        finally:
            if tmp:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def info(self) -> PluginInfo:
        return PluginInfo(name="pdf_table_extractor", kind="extractor", version="0.1",
                          capabilities=frozenset({"pdf", "tables", "text_layer"}))
