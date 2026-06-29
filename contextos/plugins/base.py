"""Plugin contracts — the only interfaces the runtime depends on (SPEC §4).

The runtime imports none of ``openai``, ``anthropic``, ``duckdb``, nor the literal
``"BM25"``. It calls these Protocols. v0.1 ships in-tree implementations; out-of-tree
registration is v1.0.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import (
    BuiltContext,
    Candidate,
    CostModelStatistics,
    ExecutionGraph,
    Goal,
    Hit,
    Intent,
    ModelCapabilities,
    ModelRequest,
    ModelResult,
    Plan,
    PlanScore,
    PluginInfo,
    ReasonRequest,
    Retrieval,
    Schedule,
    Trace,
    Verdict,
)
from ..types import Constraints


@runtime_checkable
class Plugin(Protocol):
    def info(self) -> PluginInfo: ...


# ──────────────────────────── §4.2 planner trio ────────────────────────────


@runtime_checkable
class IntentAnalyzer(Protocol):
    def analyze(self, goal: Goal) -> Intent: ...


@runtime_checkable
class CandidateGenerator(Protocol):
    def generate(self, intent: Intent, goal: Goal) -> list[Candidate]: ...
    def prune(self, candidates: list[Candidate], goal: Goal) -> list[Candidate]: ...


@runtime_checkable
class CostOptimizer(Protocol):
    def score(self, candidate: Candidate, goal: Goal) -> PlanScore: ...
    def select(self, scored: list[tuple[Candidate, PlanScore]], goal: Goal) -> Plan: ...


# ──────────────────────────── §3.1 cost estimator ────────────────────────────


@runtime_checkable
class CostEstimator(Protocol):
    def estimate(self, candidate: Candidate, goal: Goal) -> PlanScore: ...
    def statistics(self, bucket: str | None = None) -> CostModelStatistics: ...
    def observe(self, plan: Plan, trace: Trace) -> None: ...


# ──────────────────────────── §4.3 model ────────────────────────────


@runtime_checkable
class ModelPlugin(Protocol):
    def complete(self, req: ModelRequest) -> ModelResult: ...
    def capabilities(self, model: str) -> ModelCapabilities: ...
    def count_tokens(self, text: str, model: str) -> int: ...
    def info(self) -> PluginInfo: ...


# ──────────────────────────── §4.4 reasoner ────────────────────────────


@runtime_checkable
class ReasonerPlugin(Protocol):
    def reason(self, req: ReasonRequest) -> ModelResult: ...
    def info(self) -> PluginInfo: ...


# ──────────────────────────── §4.5 retriever / store ────────────────────────────


@runtime_checkable
class RetrieverPlugin(Protocol):
    def search(self, query: str, k: int, method: Retrieval) -> list[Hit]: ...
    def info(self) -> PluginInfo: ...


@runtime_checkable
class StorePlugin(Protocol):
    def index(self, path: str) -> dict: ...
    def info(self) -> PluginInfo: ...


# ──────────────────────────── §4.6 scheduler ────────────────────────────


@runtime_checkable
class SchedulerPlugin(Protocol):
    def schedule(self, graph: ExecutionGraph, constraints: Constraints) -> Schedule: ...
    def info(self) -> PluginInfo: ...


# ──────────────────────────── §4.7 remaining ────────────────────────────


@runtime_checkable
class CompressionPlugin(Protocol):
    def compress(self, text: str, target_tokens: int): ...


@runtime_checkable
class VerifierPlugin(Protocol):
    def verify(self, result: ModelResult, plan: Plan, ctx: BuiltContext) -> Verdict: ...
