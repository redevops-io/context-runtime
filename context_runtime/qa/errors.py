"""Typed refusals of QA planning.

Planning never degrades silently: an input it cannot plan for fully is refused with one of these, carrying a
stable ``code`` and the full list of ``problems`` found (not just the first).
"""
from __future__ import annotations

from dataclasses import dataclass


class Code:
    """Stable codes. Strings, so a consumer in any language can compare them."""

    # closure
    UNKNOWN_REFERENCE = "UNKNOWN_REFERENCE"
    UNKNOWN_TARGET = "UNKNOWN_TARGET"
    PIN_MISMATCH = "PIN_MISMATCH"
    CYCLE = "CYCLE"
    DUPLICATE_NODE = "DUPLICATE_NODE"
    EMPTY_SELECTION = "EMPTY_SELECTION"
    INVALID_GRAPH = "INVALID_GRAPH"
    # planning
    MISSING_ESTIMATE = "MISSING_ESTIMATE"
    MISSING_PIN = "MISSING_PIN"
    UNDECLARED_RESOURCE = "UNDECLARED_RESOURCE"
    BUDGET_INFEASIBLE = "BUDGET_INFEASIBLE"
    UNSUPPORTED_COVERAGE_CONTRACT = "UNSUPPORTED_COVERAGE_CONTRACT"
    PLAN_INVALID = "PLAN_INVALID"
    INVALID_REQUEST = "INVALID_REQUEST"


@dataclass(frozen=True)
class Problem:
    """One finding: a stable code, the subject it is about, and a human-readable detail."""

    code: str
    subject: str
    detail: str

    def to_dict(self) -> dict:
        return {"code": self.code, "subject": self.subject, "detail": self.detail}

    def __str__(self) -> str:
        return f"{self.code}: {self.subject}: {self.detail}"


class QAPlanningError(Exception):
    """Base of every typed refusal. ``code`` is the primary reason; ``problems`` lists every finding."""

    def __init__(self, code: str, message: str, problems: tuple[Problem, ...] = ()):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.problems = tuple(problems)


class ClosureError(QAPlanningError):
    """The declared graph cannot be resolved: unknown reference, pin mismatch, cycle, duplicate node."""

    def __init__(self, code: str, message: str, problems: tuple[Problem, ...] = (),
                 cycle: tuple[str, ...] = ()):
        super().__init__(code, message, problems)
        self.cycle = tuple(cycle)
        """For a ``CYCLE``: the node keys of the cycle, in order, starting at the smallest key."""


class PlanningRefusal(QAPlanningError):
    """A plan cannot be built for the request as stated."""


class InfeasibleBudget(PlanningRefusal):
    """The budget cannot cover every test in the closure. The planner refuses; it never drops a test."""

    def __init__(self, message: str, problems: tuple[Problem, ...] = (), binding: str = "",
                 candidates: tuple = ()):
        super().__init__(Code.BUDGET_INFEASIBLE, message, problems)
        self.binding = binding
        """The constraint that made it infeasible (for example ``wall_time_ms`` or ``capacity:memory_mb``)."""
        self.candidates = tuple(candidates)
        """The candidate lane counts that were tried, with their estimated makespans."""


class UnsupportedCoverage(PlanningRefusal):
    def __init__(self, value: str):
        super().__init__(Code.UNSUPPORTED_COVERAGE_CONTRACT,
                         f"coverage_contract {value} is not supported; only FULL is accepted, and no partial "
                         "coverage is planned")
