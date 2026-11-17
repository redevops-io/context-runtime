"""Narrow, restart-surviving storage for the control plane's ``/memory/*`` routes.

The fact semantics are exactly ``adapters/store_temporal.py``'s ``TemporalStore`` — this module
adds nothing but process-lifetime caching and append-only JSON-line persistence, the same
"persists under home" idiom the control plane already uses for learned policies (see ``_build()``
in ``app.py``). One store per namespace, so independent external consumers (e.g. a DeerFlow
deployment) don't share facts by accident.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from ..adapters.store_temporal import TemporalFact, TemporalStore
from ..platform_services import ensure_private_dir, locked_append, state_dir

_lock = Lock()
_stores: dict[str, TemporalStore] = {}


def _memory_dir() -> Path:
    return ensure_private_dir(Path(state_dir()) / "memory")


def _fact_path(namespace: str) -> Path:
    safe = "".join(c for c in namespace if c.isalnum() or c in "-_") or "default"
    return _memory_dir() / f"{safe}.jsonl"


def _load(namespace: str) -> TemporalStore:
    facts: list[TemporalFact] = []
    path = _fact_path(namespace)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                facts.append(TemporalFact(**d))
    return TemporalStore(facts)


def get_store(namespace: str) -> TemporalStore:
    """One ``TemporalStore`` per namespace, process-lifetime cached, lazily loaded from disk."""
    with _lock:
        store = _stores.get(namespace)
        if store is None:
            store = _load(namespace)
            _stores[namespace] = store
        return store


def append_fact(namespace: str, fact: TemporalFact) -> None:
    """Append-only: one JSON line per fact, never rewrites prior lines. Uses the platform's
    ``locked_append`` (not a bare ``open(path, "a")``) so concurrent threads/processes never
    lose or interleave a line — see its docstring for why a bare append isn't enough."""
    line = json.dumps({
        "subject": fact.subject, "predicate": fact.predicate, "obj": fact.obj,
        "valid_from": fact.valid_from, "valid_to": fact.valid_to,
        "recorded_at": fact.recorded_at, "meta": fact.meta,
    }) + "\n"
    locked_append(_fact_path(namespace), line)


def reset_for_tests() -> None:
    """Drop the in-process cache (tests only — does not touch disk)."""
    with _lock:
        _stores.clear()
