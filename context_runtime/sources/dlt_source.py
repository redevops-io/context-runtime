"""DltSource — a SourcePlugin backed by dlt (https://dlthub.com), Apache-2.0.

dlt is a *library* (not a platform), so a connector is just a Python iterable of
records — which maps cleanly onto RawAsset. This adapter wraps any dlt source/resource
(or any iterable of dict records) and turns each record into a RawAsset ready for the
same extract → quality → chunk → index pipeline the local folder uses.

    pip install "context_runtime[connectors]"

Usage:
    from dlt.sources.filesystem import filesystem, read_csv
    src = DltSource(filesystem(bucket_url="file:///data", file_glob="*.csv") | read_csv(),
                    text_fields=["title", "body"], id_field="id")
    for asset in src.read(): ...

Note: connectors run Python-side only. The Go runtime consumes the normalized output
(passages via /index), it does not re-implement connectors.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any

import dlt  # noqa: F401  — presence marks the [connectors] extra as installed

from ..types import PluginInfo, RawAsset


class DltSource:
    def __init__(self, resource: Any, *, text_fields: list[str] | None = None,
                 id_field: str | None = None, label_field: str | None = None,
                 name: str = "dlt"):
        """`resource` is a dlt resource/source (or any iterable of dict records).
        text_fields: which record fields form the text (joined); default = whole record
        as JSON. id_field / label_field: which fields provide the id / provenance label.
        """
        self._resource = resource
        self.text_fields = text_fields
        self.id_field = id_field
        self.label_field = label_field
        self.name = name

    def _iter_records(self) -> Iterable[Any]:
        r = self._resource
        return r() if callable(r) else r

    def _text_of(self, rec: Any) -> str:
        if isinstance(rec, dict) and self.text_fields:
            parts = [str(rec[f]) for f in self.text_fields if rec.get(f) not in (None, "")]
            return "\n\n".join(parts)
        if isinstance(rec, dict):
            return json.dumps(rec, ensure_ascii=False, default=str)
        return str(rec)

    def read(self) -> Iterator[RawAsset]:
        for i, rec in enumerate(self._iter_records()):
            rid = str(rec.get(self.id_field)) if (isinstance(rec, dict) and self.id_field
                                                  and rec.get(self.id_field) is not None) \
                else f"{self.name}-{i:06d}"
            label = str(rec.get(self.label_field)) if (isinstance(rec, dict) and self.label_field
                                                       and rec.get(self.label_field)) else rid
            meta = {"source": self.name}
            if isinstance(rec, dict):
                meta["record_keys"] = sorted(rec.keys())
            yield RawAsset(id=rid, uri=None, text=self._text_of(rec), label=label,
                           mime="application/json", meta=meta)

    def info(self) -> PluginInfo:
        return PluginInfo(name=f"{self.name}_dlt_source", kind="source", version="0.1",
                          capabilities=frozenset({"records", "dlt", "incremental"}))
