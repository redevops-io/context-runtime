"""Source connectors — the pull half of ingestion (SPEC §4.8).

A SourcePlugin yields RawAssets from somewhere (a local folder, a dlt connector, an
API). It does not parse; an ExtractorPlugin does that next. Keeping connectors behind
a plugin protocol means the same runtime ingests a folder of PDFs or a Postgres table
or a Notion workspace without the ingestion path knowing the difference.
"""
from __future__ import annotations

from .local import LocalFolderSource

__all__ = ["LocalFolderSource"]

try:  # optional connector suite — pip install "context_runtime[connectors]"
    from .dlt_source import DltSource  # noqa: F401

    __all__.append("DltSource")
except Exception:  # pragma: no cover - dlt absent
    pass
