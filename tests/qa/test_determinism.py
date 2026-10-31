"""Closure output is identical across hash seeds, locales and processes."""
from __future__ import annotations

import pytest
from qa_subproc import run_py

SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from qa_fixtures import random_dag
from context_runtime.qa import resolve
g = random_dag(11, 60)
c = resolve(g, ["t059", "t040", "t013"])
print(c.digest())
print(",".join(n.key for n in c.nodes))
print(";".join(f"{{t}}:{{','.join(c.exclusive_keys(t))}}" for t in c.test_ids))
"""


@pytest.mark.parametrize("seed", ["0", "1", "4242", "random"])
@pytest.mark.parametrize("locale", ["C", "C.UTF-8", "tr_TR.UTF-8"])
def test_closure_is_identical_across_hash_seeds_and_locales(seed, locale):
    from pathlib import Path

    code = SCRIPT.format(tests=str(Path(__file__).parent))
    reference = run_py(code, seed="0", locale="C")
    assert reference.count("\n") == 3
    assert run_py(code, seed=seed, locale=locale) == reference
