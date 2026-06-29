"""Execution Graph IR — the Planner/Scheduler boundary (SPEC §5).

v0.1 compiles a chosen Candidate into a linear graph, but the IR already carries the
branch/loop/approval/rollback node kinds so v0.4 slots in without a schema change.
``validate`` enforces the §5 validity rules.
"""
from __future__ import annotations

from ..types import Candidate, ExecutionGraph, GraphEdge, GraphNode

# StepSpec.type → NodeKind (identity here; explicit so the mapping is auditable)
_STEP_TO_NODE = {
    "retrieve": "retrieve", "rerank": "rerank", "compress": "compress",
    "route": "route", "reason": "reason", "delegate": "delegate", "verify": "verify",
}


def build(plan_id: str, candidate: Candidate) -> ExecutionGraph:
    """Compile a chosen candidate into a linear Execution Graph."""
    nodes: list[GraphNode] = []
    for step in candidate.steps:
        nodes.append(GraphNode(
            kind=_STEP_TO_NODE[step.type],  # type: ignore[arg-type]
            params=dict(step.params),
            plugin=step.plugin,
            budget_tokens=step.params.get("target_tokens"),
        ))
    edges = tuple(
        GraphEdge(src=nodes[i].id, dst=nodes[i + 1].id, kind="then")
        for i in range(len(nodes) - 1)
    )
    g = ExecutionGraph(nodes=tuple(nodes), edges=edges, plan_id=plan_id)
    validate(g)
    return g


def validate(g: ExecutionGraph) -> None:
    """SPEC §5 validity: ids unique, edges reference real nodes, acyclic except guarded loops."""
    ids = [n.id for n in g.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate node ids")
    idset = set(ids)
    for e in g.edges:
        if e.src not in idset or e.dst not in idset:
            raise ValueError(f"edge references unknown node: {e.src}->{e.dst}")
        if e.kind == "on_condition" and e.condition is None:
            raise ValueError("on_condition edge missing condition guard")
    # loop nodes must carry a bounded iteration count
    for n in g.nodes:
        if n.kind == "loop" and "max_iters" not in n.params:
            raise ValueError(f"loop node {n.id} missing max_iters guard")
        if n.kind == "rollback" and "compensates" not in n.params:
            raise ValueError(f"rollback node {n.id} must name nodes it compensates")
    _check_acyclic(g)


def _check_acyclic(g: ExecutionGraph) -> None:
    adj: dict[str, list[str]] = {n.id: [] for n in g.nodes}
    loop_ids = {n.id for n in g.nodes if n.kind == "loop"}
    for e in g.edges:
        # back-edges through a guarded loop node are allowed
        if e.dst in loop_ids and e.kind == "on_condition":
            continue
        adj[e.src].append(e.dst)
    color: dict[str, int] = {}

    def dfs(u: str) -> None:
        color[u] = 1
        for v in adj[u]:
            if color.get(v) == 1:
                raise ValueError("execution graph has an unguarded cycle")
            if color.get(v, 0) == 0:
                dfs(v)
        color[u] = 2

    for n in g.nodes:
        if color.get(n.id, 0) == 0:
            dfs(n.id)
