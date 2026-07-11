"""Plugin contracts — the only interfaces the runtime depends on (SPEC §4).

The runtime imports none of ``openai``, ``anthropic``, ``duckdb``, nor the literal
``"BM25"``. It calls these Protocols. v0.1 ships in-tree implementations; out-of-tree
registration is v1.0.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..types import (
    BuiltContext,
    Candidate,
    Constraints,
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
    RawAsset,
    ReasonRequest,
    Retrieval,
    Schedule,
    Trace,
    Verdict,
)


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
    # ``context`` (the intent bucket) is optional: static optimizers ignore it; an online
    # (Gen-4) optimizer uses it as the contextual-bandit key. Kept keyword-defaulted for
    # backward compatibility with callers that pass only (scored, goal).
    def select(self, scored: list[tuple[Candidate, PlanScore]], goal: Goal, context: str = "") -> Plan: ...


# ──────────────────────────── governance seam (enterprise open-core) ────────────────────────────
# The optimizer optionally consults two injected providers. Both are OPTIONAL and default to None,
# so the OSS engine runs standalone; the commercial layer (context-runtime-v3) supplies concrete
# implementations that adapt its PolicyEngine / TrustLedger to these Protocols. The engine never
# imports the enterprise types — it depends only on the interface, exactly like every other plugin.


@runtime_checkable
class PolicyProvider(Protocol):
    """Policy-Constrained Planning (Whitepaper v3): filter the feasible execution space BEFORE cost
    ranking. Return ``None`` if the candidate is policy-feasible, else a short human-readable reason
    (surfaced in ``Plan.rejected`` → EXPLAIN's "rejected alternatives + why"). The score is provided
    so cost/budget policies can bind to the real estimate. A policy-violating plan is infeasible
    regardless of its estimated quality or cost — authorization is a first-class planning constraint."""

    def feasible(self, candidate: Candidate, goal: Goal, score: PlanScore) -> str | None: ...


@runtime_checkable
class TrustProvider(Protocol):
    """Trust-Aware Execution (Whitepaper v3, Generation 5): a [0, 1] trust score for a candidate in
    the context of this goal. The optimizer uses it to break ties between equally cost-ranked feasible
    plans (and, in the full Gen-5 build, folds it into the objective). Unobserved ⇒ a neutral prior."""

    def score(self, candidate: Candidate, goal: Goal) -> float: ...


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


# ──────────────────────────── §4.8 ingestion (source → quality → extract) ────────────────────────────


@runtime_checkable
class SourcePlugin(Protocol):
    """Where raw data comes from: a local folder, a dlt connector, an API. The
    connector half of ingestion — it pulls, it does not parse. See sources/."""

    def read(self) -> Iterable[RawAsset]: ...
    def info(self) -> PluginInfo: ...


@runtime_checkable
class ExtractorPlugin(Protocol):
    """Turns a RawAsset into normalized text. The parse half of ingestion (PDF→text,
    OCR, ASR, table recognition). Returns (text, kind); text may be empty when the
    required backend is absent. See ingest/."""

    def supports(self, asset: RawAsset) -> bool: ...
    def extract(self, asset: RawAsset) -> tuple[str, str]: ...
    def info(self) -> PluginInfo: ...


@runtime_checkable
class QualityPlugin(Protocol):
    """Optional pre-index gate: clean, normalize or reject extracted text before it is
    chunked and indexed (dedup, boilerplate stripping, LLM/sidekick-driven review).
    Returns cleaned text, or None to drop the asset. See ingest/quality.py."""

    def review(self, text: str, asset: RawAsset) -> str | None: ...
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


# ──────────────────────────── tools (new seam) ────────────────────────────


@runtime_checkable
class ToolPlugin(Protocol):
    """How a plan reaches an external system (SIEM, BI core, firewall). See tools/."""

    def spec(self): ...                 # -> ToolSpec
    def run(self, args: dict): ...      # -> ToolResult


@runtime_checkable
class TraceExporter(Protocol):
    """How a finalized Trace leaves the process (Langfuse, OTel, JSONL). See observability/."""

    def export(self, trace) -> None: ...
