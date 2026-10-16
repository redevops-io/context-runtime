"""Request validation helpers for the control plane (module proxy paths, tenant binding)."""

from __future__ import annotations

import hmac
import os
import re
from urllib.parse import quote, unquote, urlsplit

# One path segment: unreserved + a few sub-delims. No '%' (no second decoding upstream),
# no slashes, no backslashes, no whitespace/control characters.
_SEGMENT = re.compile(r"^[A-Za-z0-9._~!$&'()*+,;=:@-]+$")
_MAX_DECODE_ROUNDS = 5


class UnsafePath(ValueError):
    """The requested path is not an acceptable relative API path."""


def safe_api_path(path: str) -> str:
    """Return a normalised relative path for ``/api/<path>`` or raise ``UnsafePath``.

    The path is percent-decoded repeatedly; if any decoding round yields a dot segment, a
    backslash, a NUL/control character or a slash that was encoded, it is rejected.
    Every remaining segment must match a conservative character allowlist.
    """
    if path is None:
        raise UnsafePath("empty path")
    current = path
    for _ in range(_MAX_DECODE_ROUNDS):
        _check_round(current)
        decoded = unquote(current)
        if decoded == current:
            break
        current = decoded
    else:
        raise UnsafePath("too many encoding layers")
    segments = path.split("/")
    if segments and segments[-1] == "":
        segments = segments[:-1]            # a trailing slash is fine
    out = []
    for seg in segments:
        if not _SEGMENT.match(seg) or seg in (".", ".."):
            raise UnsafePath("invalid path segment")
        out.append(seg)
    return "/".join(out) + ("/" if path.endswith("/") and out else "")


def _check_round(s: str) -> None:
    if "\\" in s or any(ord(c) < 0x20 or ord(c) == 0x7F for c in s):
        raise UnsafePath("invalid character")
    if any(seg.strip() in (".", "..") for seg in s.split("/")):
        raise UnsafePath("dot segment")
    low = s.lower()
    if "%2f" in low or "%5c" in low or "%00" in low:
        raise UnsafePath("encoded separator")


def upstream_api_url(base: str, path: str) -> str:
    """Build ``<base>/api/<path>`` from a validated path and assert it stays under ``/api/``."""
    rel = safe_api_path(path)
    url = base.rstrip("/") + "/api/" + "/".join(quote(s, safe="") for s in rel.split("/"))
    prefix = urlsplit(base.rstrip("/")).path + "/api/"
    got = urlsplit(url).path
    if not got.startswith(prefix) or ".." in got.split("/"):
        raise UnsafePath("path escapes /api/")
    return url


# ───────────────────────────── tenant binding ─────────────────────────────

def parse_tenant_keys(spec: str) -> dict[str, frozenset[str]]:
    """Parse ``CR_TENANT_KEYS``: ``key1=tenantA,tenantB;key2=*`` -> {key: allowed tenants}.

    ``*`` binds a credential to every tenant. Malformed entries are ignored.
    """
    out: dict[str, frozenset[str]] = {}
    for entry in (spec or "").split(";"):
        key, sep, tenants = entry.partition("=")
        key = key.strip()
        if not key or not sep:
            continue
        out[key] = frozenset(t.strip() for t in tenants.split(",") if t.strip())
    return out


def allowed_tenants(credential: str | None, bindings: dict[str, frozenset[str]],
                    all_tenants: list[str], default: str, anonymous_any: bool) -> set[str]:
    """Tenants a caller may use. Unbound credentials get only the default tenant, unless the
    deployment explicitly enables anonymous multi-tenant mode (``CR_ALLOW_ANONYMOUS_TENANTS``)."""
    if credential:
        for key, tenants in bindings.items():
            if hmac.compare_digest(key.encode(), credential.encode()):
                return set(all_tenants) if "*" in tenants else set(tenants) & set(all_tenants)
    if anonymous_any:
        return set(all_tenants)
    return {default}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
