"""Scheduler — decides *when/where* (SPEC §4.6, §5.1).

The Planner decides *what* (the Execution Graph); the Scheduler turns it into a
physical Schedule (ordered parallel waves + concurrency + retry). v0.1 is a trivial
topological sort that a real Dagster run would consume. Cost-aware scheduling (reorder
for latency, budget-aware concurrency) is v2 — it slots in behind this contract.
"""
from __future__ import annotations

from ..types import Constraints, ExecutionGraph, PluginInfo, Schedule


class TopoScheduler:
    """``DagsterScheduler`` default: Kahn topological waves."""

    def __init__(self, max_concurrency: int = 4, default_retries: int = 1):
        self.max_concurrency = max_concurrency
        self.default_retries = default_retries

    def schedule(self, graph: ExecutionGraph, constraints: Constraints) -> Schedule:
        indeg: dict[str, int] = {n.id: 0 for n in graph.nodes}
        adj: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
        for e in graph.edges:
            if e.kind == "on_condition":   # loop back-edges don't count toward ordering
                continue
            adj[e.src].append(e.dst)
            indeg[e.dst] += 1

        waves: list[tuple[str, ...]] = []
        frontier = [nid for nid, d in indeg.items() if d == 0]
        seen = set(frontier)
        while frontier:
            waves.append(tuple(frontier))
            nxt: list[str] = []
            for nid in frontier:
                for v in adj[nid]:
                    indeg[v] -= 1
                    if indeg[v] == 0 and v not in seen:
                        nxt.append(v)
                        seen.add(v)
            frontier = nxt

        retry = {n.id: self.default_retries for n in graph.nodes if n.kind in ("reason", "delegate", "verify")}
        return Schedule(waves=tuple(waves), max_concurrency=self.max_concurrency, retry=retry)

    def info(self) -> PluginInfo:
        return PluginInfo(name="topo_scheduler", kind="scheduler")
