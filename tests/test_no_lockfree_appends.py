"""No new lock-free append writer in the repository's Python code.

Lines appended with a bare ``open(path, "a")`` (or ``io.open``, ``codecs.open``, ``gzip.open``, ``os.fdopen``,
``os.open`` with ``O_APPEND``, ``logging.FileHandler``, ``logging.basicConfig(filename=...)``) can be lost on Windows
when writers overlap: the C runtime appends as "seek to the end, then write". Appenders must go through
``context_runtime.platform_services.locked_append``. A new bare append fails this test unless it is added to
``ALLOWED`` below, with the number of occurrences expected in that function and a reason a reviewer accepts.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ("context_runtime", "scripts", "eval", "benchmarks", "examples")

#: (file relative to the repo root, enclosing function) -> (expected number of bare appends, reason).
#: More occurrences than listed fail the test; fewer only warn, so an entry can be cleaned up later at leisure.
ALLOWED = {
    ("context_runtime/platform_services.py", "locked_append"):
        (1, "the helper itself: appends inside the in-process and file locks"),
    ("context_runtime/control_plane/context.py", "_append"):
        (1, "append semantics covered by dedicated tests in that module"),
}

_MODE = re.compile(r"^[rwxabt+]{1,4}$")            # a string that looks like a file mode
_FILE_HANDLERS = {"FileHandler", "WatchedFileHandler", "RotatingFileHandler", "TimedRotatingFileHandler"}
#: callable name -> (positional index of the mode, may a non-literal mode be flagged?)
_BUILTIN_STYLE = {"open": (1, True), "fdopen": (1, True), "codecs.open": (1, False), "io.open": (1, False),
                  "gzip.open": (1, False), "bz2.open": (1, False), "lzma.open": (1, False),
                  "tarfile.open": (1, False)}


def _dotted(fn: ast.expr) -> str:
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        base = _dotted(fn.value)
        return f"{base}.{fn.attr}" if base else fn.attr
    return ""


def _mode_arg(call: ast.Call, positional: int):
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    return call.args[positional] if len(call.args) > positional else None


def _is_append_literal(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(_MODE.match(node.value)) \
        and "a" in node.value


def _find(source: str):
    """Yield (lineno, enclosing function, description) for constructs that may append without a lock."""
    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def enclosing(node: ast.AST) -> str:
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return "<module>"

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "O_APPEND" or isinstance(node, ast.Attribute) \
                and node.attr == "O_APPEND":
            yield node.lineno, enclosing(node), "O_APPEND"
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        last = name.rsplit(".", 1)[-1]
        if name in _BUILTIN_STYLE or (last == "fdopen" and name.endswith("os.fdopen")):
            key = name if name in _BUILTIN_STYLE else "fdopen"
            positional, flag_nonliteral = _BUILTIN_STYLE[key]
            mode = _mode_arg(node, positional)
            if _is_append_literal(mode):
                yield node.lineno, enclosing(node), f"{name}(..., {mode.value!r})"
            elif mode is not None and not isinstance(mode, ast.Constant) and flag_nonliteral:
                yield node.lineno, enclosing(node), f"{name}() with a non-literal mode (cannot prove it is not append)"
        elif last == "open" and name not in ("os.open",):
            # Path(...).open(mode) and other `.open(...)`: only a literal that looks like an append mode is flagged
            mode = _mode_arg(node, 0)
            if _is_append_literal(mode):
                yield node.lineno, enclosing(node), f"{name}({mode.value!r})"
        elif last in _FILE_HANDLERS:
            mode = _mode_arg(node, 1)
            if not (isinstance(mode, ast.Constant) and mode.value in ("w", "x")):
                yield node.lineno, enclosing(node), f"{last} (appends by default)"
        elif name.endswith("basicConfig"):
            if any(k.arg == "filename" for k in node.keywords):
                fm = next((k.value for k in node.keywords if k.arg == "filemode"), None)
                if not (isinstance(fm, ast.Constant) and fm.value in ("w", "x")):
                    yield node.lineno, enclosing(node), "logging.basicConfig(filename=...) (appends by default)"


def occurrences(source: str, rel: str = "<src>") -> dict[tuple[str, str], list[str]]:
    """Bare appends per (file, function), one entry per source line."""
    out: dict[tuple[str, str], list[str]] = {}
    seen: set[int] = set()
    for lineno, func, what in _find(source):
        if lineno in seen:
            continue
        seen.add(lineno)
        out.setdefault((rel, func), []).append(f"{rel}:{lineno} in {func}(): {what}")
    return out


def violations(found: dict[tuple[str, str], list[str]]) -> list[str]:
    out = []
    for key, lines in found.items():
        allowed = ALLOWED.get(key, (0, ""))[0]
        if len(lines) > allowed:
            out.extend(lines if allowed == 0 else lines[allowed:] or lines)
    return out


# ───────────────────────── the detector ─────────────────────────

def _v(src: str, rel: str = "<src>") -> list[str]:
    return violations(occurrences(src, rel))


def test_detector_flags_every_append_form():
    for src in (
        "open(p, 'a')", "open(p, mode='ab')", "open(p, 'a+', encoding='utf-8')", "q.open('a', encoding='utf-8')",
        "Path(p).open(mode='ab')", "io.open(p, 'a')", "codecs.open(p, 'a', 'utf-8')", "gzip.open(p, 'ab')",
        "gzip.open(p, 'at')", "bz2.open(p, 'ab')", "tarfile.open(p, 'a')", "os.fdopen(fd, 'a')",
        "os.open(p, os.O_WRONLY | os.O_APPEND)", "from os import O_APPEND\nflags = O_APPEND",
        "logging.FileHandler('x.log')", "handlers.RotatingFileHandler('x.log')",
        "logging.basicConfig(filename='x.log')", "open(p, m)", "os.fdopen(fd, m)",
    ):
        assert _v(src), src


def test_detector_ignores_reads_writes_and_unrelated_open_methods():
    for src in (
        "open(p)", "open(p, 'r')", "open(p, 'wb')", "open(p, mode='r+')", "q.open('w')", "io.open(p, 'rb')",
        "gzip.open(p, 'rt')", "os.fdopen(fd, 'w')", "os.open(p, os.O_RDONLY)",
        "logging.FileHandler('x', 'w')", "logging.basicConfig(filename='x.log', filemode='w')",
        "logging.basicConfig(level=10)", "webbrowser.open('https://example.com/a')", "zf.open('data.json')",
        "Image.open('a.png')", "pdfplumber.open(p)", "opener.open(req, timeout=5)", "tarfile.open(p, 'r:gz')",
        "codecs.open(p, 'r', 'utf-8')", "x.open(path_variable)", "gzip.open(p, m)", "io.open(p, m)",
    ):
        assert _v(src) == [], src


def test_the_allow_list_counts_occurrences_per_function():
    rel = "context_runtime/platform_services.py"
    one = "def locked_append(p):\n    open(p, 'ab')\n"
    assert _v(one, rel) == []
    two = "def locked_append(p):\n    open(p, 'ab')\n    open(p, 'ab')\n"      # a second bare append is caught
    assert len(_v(two, rel)) == 1
    assert _v("def other(p):\n    open(p, 'ab')\n", rel) == ["context_runtime/platform_services.py:2 in other(): open(..., 'ab')"]
    assert _v("def locked_append(p):\n    pass\n", rel) == []                  # fewer than listed is not a failure


# ───────────────────────── the repository ─────────────────────────

def _scan() -> dict[tuple[str, str], list[str]]:
    found: dict[tuple[str, str], list[str]] = {}
    for top in SCANNED:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.update(occurrences(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix()))
    return found


def test_no_new_lock_free_append_in_the_repository():
    bad = violations(_scan())
    assert not bad, ("lock-free append writers (use context_runtime.platform_services.locked_append, or add an "
                     "ALLOWED entry with a count and a reason):\n" + "\n".join(bad))


def test_allow_list_entries_that_no_longer_match_only_warn():
    found = _scan()
    for key, (count, _reason) in ALLOWED.items():
        have = len(found.get(key, []))
        if have < count:
            warnings.warn(f"allow-list entry {key} expects {count} bare append(s) but finds {have}; it can be "
                          "removed or lowered", stacklevel=1)
