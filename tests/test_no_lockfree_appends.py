"""No new lock-free append writer in the package.

Lines appended with a bare ``open(path, "a")`` / ``os.open(..., O_APPEND)`` / ``logging.FileHandler`` can be lost on
Windows when writers overlap (the C runtime appends as seek-then-write). Appenders must go through
``context_runtime.platform_services.locked_append``. A new bare append fails this test unless it is added to
``ALLOWED`` below with a reason a reviewer accepts.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANNED = ("context_runtime", "scripts")

#: (file relative to the repo root, enclosing function) -> reason
ALLOWED = {
    ("context_runtime/platform_services.py", "locked_append"):
        "the helper itself: appends inside file_lock",
    ("context_runtime/control_plane/context.py", "_append"):
        "approvals queue: handled separately from the append-writer change; listed here so it is not forgotten",
}

_FILE_HANDLERS = {"FileHandler", "WatchedFileHandler", "RotatingFileHandler", "TimedRotatingFileHandler"}


def _mode_of(call: ast.Call, positional: int):
    for kw in call.keywords:
        if kw.arg == "mode":
            return kw.value
    return call.args[positional] if len(call.args) > positional else None


def _append_calls(source: str):
    """Yield (lineno, enclosing function, description) for every construct that may append without a lock."""
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
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
        if name == "open" and not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                                   and fn.value.id == "os"):
            positional = 1 if isinstance(fn, ast.Name) else 0       # open(path, mode) vs path.open(mode)
            mode = _mode_of(node, positional)
            if mode is None:
                continue
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                if "a" in mode.value:
                    yield node.lineno, enclosing(node), f"open(..., {mode.value!r})"
            elif isinstance(fn, ast.Name):      # builtin open(path, mode); `.open(...)` of other objects is not checked
                yield node.lineno, enclosing(node), "open() with a non-literal mode (cannot prove it is not append)"
        elif name == "open" and "O_APPEND" in ast.unparse(node):
            yield node.lineno, enclosing(node), "os.open with O_APPEND"
        elif name in _FILE_HANDLERS:
            mode = _mode_of(node, 1)
            if not (isinstance(mode, ast.Constant) and mode.value == "w"):
                yield node.lineno, enclosing(node), f"{name} (appends by default)"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "O_APPEND":
            yield node.lineno, enclosing(node), "O_APPEND"


def violations(source: str, rel: str = "<src>") -> list[str]:
    out, seen = [], set()
    for lineno, func, what in _append_calls(source):
        key = (rel, func)
        if key in ALLOWED or lineno in seen:
            continue
        seen.add(lineno)
        out.append(f"{rel}:{lineno} in {func}(): {what}")
    return out


def test_detector_flags_the_unsafe_forms_and_ignores_reads_and_writes():
    bad = (
        "open(p, 'a')\nopen(p, mode='ab')\nq.open('a', encoding='utf-8')\nos.open(p, os.O_WRONLY | os.O_APPEND)\n"
        "logging.FileHandler('x.log')\nopen(p, m)\npdfplumber.open(p)\n"
    )
    assert len(violations(bad)) == 6
    good = "open(p)\nopen(p, 'r')\nopen(p, 'wb')\nq.open('w')\nlogging.FileHandler('x', 'w')\nopen(p, mode='r+')\n"
    assert violations(good) == []


def test_the_allow_list_is_per_function():
    src = "def locked_append(p):\n    open(p, 'ab')\n\ndef other(p):\n    open(p, 'ab')\n"
    assert violations(src, "context_runtime/platform_services.py") == [
        "context_runtime/platform_services.py:5 in other(): open(..., 'ab')"]


def test_no_new_lock_free_append_in_the_package():
    found: list[str] = []
    for top in SCANNED:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.extend(violations(path.read_text(encoding="utf-8"), path.relative_to(ROOT).as_posix()))
    assert not found, ("lock-free append writers (use context_runtime.platform_services.locked_append, or add an "
                       "ALLOWED entry with a reason):\n" + "\n".join(found))


def test_every_allow_list_entry_still_matches_something():
    seen = set()
    for rel, _func in ALLOWED:
        text = (ROOT / rel).read_text(encoding="utf-8")
        seen |= {(rel, f) for _l, f, _w in _append_calls(text)}
    assert seen == set(ALLOWED), f"stale allow-list entries: {set(ALLOWED) - seen}"
