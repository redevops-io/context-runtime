"""Request validation helpers for the control plane (module proxy paths, tenant binding)."""

from __future__ import annotations

import enum
import hashlib
import hmac
import os
import re
from pathlib import Path
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
    """Tenants a caller may use. A credential that is not bound to any tenant gets only the
    default tenant, unless the deployment explicitly enables multi-tenant mode for every
    authenticated caller (``CR_ALLOW_ANONYMOUS_TENANTS``). Callers without any credential never
    reach this function on a protected route: the access gate refuses them first."""
    if credential:
        match: frozenset[str] | None = None
        for key, tenants in bindings.items():            # no early exit: constant work per key
            if digest_equal(key, credential) and match is None:
                match = tenants
        if match is not None:
            return set(all_tenants) if "*" in match else set(match) & set(all_tenants)
    if anonymous_any:
        return set(all_tenants)
    return {default}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ───────────────────────────── credentials and access classes ─────────────────────────────

def digest_equal(a: str, b: str) -> bool:
    """Constant-time comparison of two secrets. Both sides are hashed first so the comparison
    always runs over equal-length input and the length of the configured secret is not exposed."""
    return hmac.compare_digest(hashlib.sha256(a.encode("utf-8")).digest(),
                               hashlib.sha256(b.encode("utf-8")).digest())


def admin_key() -> str | None:
    """The control-plane key (``CONTEXT_RUNTIME_API_KEY``, else ``AGENTIC_OS_API_KEY``).
    An empty or whitespace-only value counts as not configured."""
    for name in ("CONTEXT_RUNTIME_API_KEY", "AGENTIC_OS_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def credential_from_headers(headers) -> str | None:
    """The caller's credential: ``X-API-Key`` or an ``Authorization: Bearer`` token."""
    key = (headers.get("x-api-key") or "").strip()
    if not key:
        auth = headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    return key or None


class Access(enum.Enum):
    """Who may call a route.

    PUBLIC: no credential (only the minimal health route).
    READER: the control-plane key or a tenant key from ``CR_TENANT_KEYS`` (tenant-scoped reads).
    ADMIN: the control-plane key only (control plane, mutation, ingestion, learning).
    """
    PUBLIC = "public"
    READER = "reader"
    ADMIN = "admin"


NOT_CONFIGURED = "server authentication is not configured"


def authorize(access: Access, credential: str | None) -> tuple[int, str] | None:
    """Return ``None`` when the call is allowed, else ``(status, detail)``.

    Fail closed: when the deployment has not configured the credential a protected route needs,
    the answer is 503 for every caller. With the credential configured, a missing or wrong one is 401.
    """
    if access is Access.PUBLIC:
        return None
    key = admin_key()
    tenant_keys = parse_tenant_keys(os.environ.get("CR_TENANT_KEYS", "")) if access is Access.READER else {}
    if key is None and not tenant_keys:
        return 503, NOT_CONFIGURED
    if access is Access.ADMIN and key is None:
        return 503, NOT_CONFIGURED
    if credential:
        ok = False
        if key is not None and digest_equal(key, credential):
            ok = True
        for tk in tenant_keys:                           # no early exit: constant work per key
            if digest_equal(tk, credential):
                ok = True
        if ok:
            return None
    return 401, "invalid or missing credential"


# ───────────────────────────── ingestion: corpus root ─────────────────────────────

class CorpusError(Exception):
    """An ingestion request that must be refused. ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


CORPUS_ROOT_ENV = "CONTEXT_RUNTIME_CORPUS_ROOT"
_CORPUS_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UNKNOWN_CORPUS = "unknown corpus"


def corpus_segments(ref: str) -> list[str]:
    """Validate a corpus reference (an identifier or a relative path under the corpus root) and
    return its segments. Refused: empty, absolute, drive-qualified, ``.``/``..``, hidden, backslash,
    percent-encoded or control characters, and anything outside ``[A-Za-z0-9._-]`` per segment."""
    if not isinstance(ref, str) or not ref or len(ref) > 512:
        raise CorpusError(400, "invalid corpus reference")
    if "\\" in ref or "%" in ref or any(ord(c) < 0x20 or ord(c) == 0x7F for c in ref):
        raise CorpusError(400, "invalid corpus reference")
    segments = ref.split("/")
    if segments and segments[-1] == "":
        segments = segments[:-1]                         # one trailing slash is fine
    if not segments or len(segments) > 8:
        raise CorpusError(400, "invalid corpus reference")
    for seg in segments:
        if not _CORPUS_SEGMENT.match(seg) or seg.endswith("."):
            raise CorpusError(400, "invalid corpus reference")
    return segments


def resolve_corpus_dir(corpus: str | None, path: str | None, root: str | None = None) -> Path:
    """Resolve an ingestion request to a real directory inside the configured corpus root.

    ``corpus`` is the allowlisted identifier (a directory directly under the root); ``path`` is
    accepted for older clients only as a path *relative to the root*. No root configured means
    ingestion is disabled (503). A directory that resolves outside the root (``..``, a symlinked
    directory) or that contains a symlink pointing outside itself is refused with the same answer
    as a missing one.
    """
    root = (root if root is not None else os.environ.get(CORPUS_ROOT_ENV, "")).strip()
    if not root:
        raise CorpusError(503, f"ingestion is disabled: {CORPUS_ROOT_ENV} is not configured")
    try:
        root_real = Path(root).resolve(strict=True)
    except (OSError, RuntimeError):
        raise CorpusError(503, f"ingestion is disabled: {CORPUS_ROOT_ENV} is not a directory") from None
    if not root_real.is_dir():
        raise CorpusError(503, f"ingestion is disabled: {CORPUS_ROOT_ENV} is not a directory")
    ref = corpus if corpus is not None else path
    if ref is None:
        raise CorpusError(400, "a corpus identifier is required")
    segments = corpus_segments(ref)
    if corpus is not None and len(segments) != 1:
        raise CorpusError(400, "invalid corpus reference")
    try:
        target = root_real.joinpath(*segments).resolve(strict=True)
    except (OSError, RuntimeError):
        raise CorpusError(404, _UNKNOWN_CORPUS) from None
    if target == root_real or not target.is_relative_to(root_real) or not target.is_dir():
        raise CorpusError(404, _UNKNOWN_CORPUS)
    _assert_no_symlink_escape(target)
    return target


def _assert_no_symlink_escape(directory: Path) -> None:
    """Refuse a corpus directory that holds a symlink (file or directory) whose target is outside it."""
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
        for name in [*dirnames, *filenames]:
            entry = Path(dirpath, name)
            if entry.is_symlink():
                try:
                    real = entry.resolve(strict=True)
                except (OSError, RuntimeError):
                    raise CorpusError(404, _UNKNOWN_CORPUS) from None
                if not real.is_relative_to(directory):
                    raise CorpusError(404, _UNKNOWN_CORPUS)
