"""Stable JSON (de)serialization for the persisted seams (SPEC §8).

Persisted types (Plan, ExecutionGraph, Trace, Plan-Cache entries) MUST round-trip,
including unknown forward fields, and reject a higher MAJOR ``spec_version``. We meet
that by recursively walking dataclass fields and parking unknown keys in each type's
``extra`` bag, which is merged back on dump.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, get_args, get_origin

from . import types as T

SPEC_MAJOR = int(T.SPEC_VERSION.split(".")[0])


def dump(obj: Any) -> Any:
    """Dataclass → JSON-safe structure (tuples → lists, ``extra`` merged up)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            if f.name == "extra":
                continue
            out[f.name] = dump(getattr(obj, f.name))
        extra = getattr(obj, "extra", None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                out.setdefault(k, v)
        return out
    if isinstance(obj, (list, tuple)):
        return [dump(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.items()}
    return obj


def dumps(obj: Any, **kw: Any) -> str:
    return json.dumps(dump(obj), **kw)


def _field_type_map(cls: type) -> dict[str, Any]:
    return {f.name: f.type for f in dataclasses.fields(cls)}


def load(cls: type, data: Any) -> Any:
    """JSON structure → dataclass ``cls``, preserving unknown keys in ``extra``.

    Raises ``ValueError`` on a higher MAJOR spec_version (SPEC §8).
    """
    if not (dataclasses.is_dataclass(cls) and isinstance(data, dict)):
        return data

    sv = data.get("spec_version")
    if isinstance(sv, str) and int(sv.split(".")[0]) > SPEC_MAJOR:
        raise ValueError(f"spec_version {sv} has a higher major than supported {T.SPEC_VERSION}")

    known = {f.name for f in dataclasses.fields(cls)}
    has_extra = "extra" in known
    types = _field_type_map(cls)
    kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    for key, val in data.items():
        if key == "extra":
            extra.update(val or {})
            continue
        if key not in known:
            extra[key] = val           # unknown forward field — preserve verbatim
            continue
        kwargs[key] = _coerce(types[key], val)

    if has_extra and extra:
        kwargs["extra"] = extra
    return cls(**kwargs)  # type: ignore[arg-type]


def _coerce(anno: Any, val: Any) -> Any:
    """Best-effort coercion of a JSON value to a field's declared type."""
    if val is None:
        return None
    # resolve string annotations against the types module
    if isinstance(anno, str):
        anno = getattr(T, anno, None)
        if anno is None:
            return val

    origin = get_origin(anno)
    args = get_args(anno)

    if dataclasses.is_dataclass(anno) and isinstance(val, dict):
        return load(anno, val)

    if origin in (tuple,):
        if not isinstance(val, (list, tuple)):
            return val
        if len(args) == 2 and args[1] is Ellipsis:     # tuple[X, ...]
            return tuple(_coerce(args[0], x) for x in val)
        if args:                                       # tuple[A, B]
            return tuple(_coerce(a, x) for a, x in zip(args, val))
        return tuple(val)

    if origin in (list,):
        inner = args[0] if args else Any
        return [_coerce(inner, x) for x in val]

    # Optional[X] / Union — try the first dataclass arg, else passthrough
    if args:
        for a in args:
            if dataclasses.is_dataclass(a) and isinstance(val, dict):
                return load(a, val)
    return val


def loads(cls: type, s: str) -> Any:
    return load(cls, json.loads(s))
