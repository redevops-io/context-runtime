"""The README's published API tour must run exactly as written, offline.

The test extracts the first python block under "## 30-second tour" and executes it verbatim in a fresh namespace
(no edits, no extra setup), so the README cannot drift from the API again.
"""
from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def _python_block(heading: str) -> str:
    text = README.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## )", text, re.S | re.M)
    assert m, f"README has no '## {heading}' section"
    blocks = re.findall(r"```python\n(.*?)```", m.group(1), re.S)
    assert blocks, f"no python block under '## {heading}'"
    return blocks[0]


def test_readme_30_second_tour_runs_verbatim():
    code = _python_block("30-second tour")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(code, "README.md (30-second tour)", "exec"), {"__name__": "__readme_tour__"})
    printed = [line for line in out.getvalue().splitlines() if line.strip()]
    assert len(printed) >= 3, printed  # RUN, EXPLAIN and SIMULATE each print a line
