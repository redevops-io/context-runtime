"""ContextOS — a database query planner for LLM context.

    from contextos import ContextRuntime
    rt = ContextRuntime.default(docs)
    result = rt.run("Explain why deployment X failed")
    print(result.answer, result.trace)

The core abstraction is ``run``, not ``ask``. See SPEC.md for the contracts.
"""
from __future__ import annotations

from .runtime.runtime import ContextRuntime
from .types import (
    Constraints,
    Explanation,
    Goal,
    Plan,
    RunResult,
    Simulation,
    SourceRef,
)

__version__ = "0.1.0"
__all__ = [
    "ContextRuntime",
    "Goal",
    "Constraints",
    "SourceRef",
    "Plan",
    "RunResult",
    "Explanation",
    "Simulation",
]
