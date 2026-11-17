"""Every text-mode file access in the package, scripts and tests names its encoding.

Without ``encoding=`` Python uses the locale code page (cp1252 on Windows), so the same corpus file
decodes differently from one operating system to the next. This scans the source with ``ast`` (which also
sees calls that a type-based linter cannot attribute to ``pathlib``; it covers the common call forms, not every
possible spelling, e.g. ``gzip.open(p, 'rt')`` or ``os.fdopen``) and fails on any ``open()``,
``Path.open()``, ``read_text()``, ``write_text()`` or text-mode ``NamedTemporaryFile`` without it.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ("context_runtime", "scripts", "tests", "examples", "eval", "benchmarks")
# Receivers whose ``.open`` is not ``Path.open``/``io.open``.
NOT_FILES = {"os", "gzip", "bz2", "zipfile", "tarfile", "Image", "webbrowser", "codecs", "pdfplumber", "fitz",
             "Image", "wave"}


def _binary(mode: ast.expr | None) -> bool:
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def _mode(call: ast.Call, positional: int) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    return call.args[positional] if len(call.args) > positional else None


def violations(source: str, name: str = "<src>") -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if any(k.arg == "encoding" or k.arg is None for k in node.keywords):
            continue
        fn = node.func
        if any(k.arg in ("text", "universal_newlines") and isinstance(k.value, ast.Constant) and k.value.value is True
               for k in node.keywords):
            out.append(f"{name}:{node.lineno}: text-mode subprocess without encoding=")
        elif isinstance(fn, ast.Attribute) and fn.attr in ("read_text", "write_text"):
            out.append(f"{name}:{node.lineno}: {fn.attr}() without encoding=")
        elif isinstance(fn, ast.Attribute) and fn.attr == "open":
            recv = fn.value
            if isinstance(recv, ast.Name) and recv.id in NOT_FILES:
                continue
            if "opener" in ast.unparse(recv):   # urllib opener.open(...)
                continue
            if not _binary(_mode(node, 0)):
                out.append(f"{name}:{node.lineno}: .open() in text mode without encoding=")
        elif isinstance(fn, ast.Name) and fn.id == "open":
            if not _binary(_mode(node, 1)):
                out.append(f"{name}:{node.lineno}: open() in text mode without encoding=")
        elif isinstance(fn, ast.Attribute) and fn.attr == "NamedTemporaryFile":
            mode = _mode(node, 0)
            if mode is not None and not _binary(mode):
                out.append(f"{name}:{node.lineno}: NamedTemporaryFile text mode without encoding=")
    return out


def test_checker_flags_the_unsafe_forms():
    bad = (
        "open(p)\nopen(p, 'w')\nPath(p).read_text()\nq.write_text('x')\nq.open('a')\n"
        "tempfile.NamedTemporaryFile('w')\nPath(p).read_text(errors='ignore')\nsubprocess.run(c, text=True)\n"
    )
    assert len(violations(bad)) == 8
    good = (
        "open(p, 'rb')\nopen(p, encoding='utf-8')\nPath(p).read_text(encoding='utf-8')\nq.open('wb')\n"
        "os.open(p, 0)\nopener.open(r)\nfh.read()\nsubprocess.run(c, text=True, encoding='utf-8')\n"
    )
    assert violations(good) == []


def test_no_text_mode_file_access_without_an_explicit_encoding():
    found: list[str] = []
    for top in SCANNED:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.extend(violations(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))))
    assert not found, "text-mode file access without encoding=:\n" + "\n".join(found)
