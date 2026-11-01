"""Portable file-name rules shared by anything that turns an identifier into a path component.

A name that is fine on Linux can be unusable on Windows (device names such as ``con`` or ``nul`` are
reserved, with or without an extension) or on another file system (component length limits). These
helpers apply the strictest common rule so an identifier maps to the same, valid file name everywhere.
"""

from __future__ import annotations

import hashlib

# Device names Windows reserves in every directory, regardless of extension or case. The superscript
# digits are included because Windows treats COM¹/COM²/COM³ and LPT¹/LPT²/LPT³ as reserved too; COM0 and LPT0
# are listed as reserved in current Microsoft documentation.
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{c}" for c in "0123456789¹²³"}
    | {f"lpt{c}" for c in "0123456789¹²³"}
)

# Marker for the two escapes below. Callers that build names from ``[a-z0-9_-]`` never produce it, so an
# escaped name cannot equal any ordinary name: the mapping is injective (up to a 32-bit digest on truncation).
ESCAPE = "~"

# 255 is the usual per-component limit (NTFS, APFS, ext4); stay well below it so a suffix
# such as ``.jsonl`` or ``.tmp`` still fits.
MAX_COMPONENT = 128


def is_windows_reserved(name: str) -> bool:
    """True when ``name`` (the part before the first dot) is a Windows device name."""
    stem = name.split(".", 1)[0].rstrip(" ").lower()
    return stem in _WINDOWS_RESERVED


def cap_component(name: str, limit: int = MAX_COMPONENT) -> str:
    """Shorten ``name`` to at most ``limit`` characters, keeping it unique by appending ``~`` and a short
    digest of the full name when it had to be truncated."""
    if len(name) <= limit:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"{name[: limit - 9]}{ESCAPE}{digest}"


def portable_component(name: str, limit: int = MAX_COMPONENT) -> str:
    """Make an already character-filtered ``name`` safe as a single path component everywhere:
    reserved Windows device names get a trailing ``~`` (``con`` becomes ``con~``, ``nul.txt`` becomes
    ``nul~.txt``), and over-long names are shortened. Distinct inputs stay distinct: ``con`` and ``con_`` map to
    ``con~`` and ``con_``."""
    if is_windows_reserved(name):
        stem, dot, rest = name.partition(".")
        name = stem + ESCAPE + dot + rest
    return cap_component(name, limit)
