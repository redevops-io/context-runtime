"""QA planning: declared-graph closure, and (in later modules) cost-based selection that emits a ``QAPlan``.

The output of planning is data: a canonical ``qa/v1`` ``QAPlan`` built with the ``runtime-contracts`` classes. This
package never drives a browser, never edits or even reads assertions, and never imports a worker or runner;
executing the plan is the job of whatever consumes the data. It requires the ``qa`` extra
(``runtime-contracts``); importing it without that package fails with the install hint, it does not degrade.
"""
from __future__ import annotations

from . import _contracts  # noqa: F401  (hard import: fails loudly, with the install hint, when the package is absent)
from .closure import (
    Claim, Closure, ClosureNode, EffectiveClaim, Node, NodeKind, QAGraph, Reason, Ref, graph_from_specs,
    node_key, resolve,
)
from .cost import CostModel, Estimate, bottom_levels, critical_path
from .errors import (
    ClosureError, Code, InfeasibleBudget, PlanningRefusal, Problem, QAPlanningError, UnsupportedCoverage,
)
from .planner import (
    PLANNER_ID, PLANNER_VERSION, Candidate, CostBasedQAPlanner, PlanBudget, PlanRequest, PlanResult, QAPlanner,
    RunCaps, ScheduledItem, request_digest, require_valid_plan, verify_plan,
)

__all__ = [
    "Claim", "Closure", "ClosureNode", "EffectiveClaim", "Node", "NodeKind", "QAGraph", "Reason", "Ref",
    "graph_from_specs", "node_key", "resolve",
    "CostModel", "Estimate", "bottom_levels", "critical_path",
    "ClosureError", "Code", "InfeasibleBudget", "PlanningRefusal", "Problem", "QAPlanningError", "UnsupportedCoverage",
    "PLANNER_ID", "PLANNER_VERSION", "Candidate", "CostBasedQAPlanner", "PlanBudget", "PlanRequest", "PlanResult",
    "QAPlanner", "RunCaps", "ScheduledItem", "request_digest", "require_valid_plan", "verify_plan",
]
