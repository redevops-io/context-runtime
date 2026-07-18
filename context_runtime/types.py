"""The Context Runtime type catalog — SPEC.md §2–§7 made concrete.

Pure data only: dataclasses, no behavior, no imports of plugins or runtime. Everything
that crosses a process boundary or is persisted (Plan, ExecutionGraph, Trace) carries
``spec_version`` and an ``extra`` bag so unknown forward fields round-trip (SPEC §8).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

SPEC_VERSION = "0.1"


def new_id(kind: str) -> str:
    """Stable opaque id: ``<kind>_<hex>`` (SPEC §-conventions)."""
    return f"{kind}_{uuid.uuid4().hex[:12]}"


# ──────────────────────────── §2.1 request-side ────────────────────────────

Sensitivity = Literal["public", "internal", "restricted"]


@dataclass(frozen=True)
class Constraints:
    """Hard ceilings (feasibility) + soft requirements (SPEC §2.1)."""

    max_cost_usd: float | None = None
    max_latency_seconds: float | None = None
    max_tokens: int | None = None
    require_citations: bool = False
    require_verification: bool = False
    sensitivity: Sensitivity = "public"
    weight_overrides: dict[str, float] = field(default_factory=dict)


SourceKind = Literal["docs", "code", "logs", "metrics", "api", "graph", "memory"]


@dataclass(frozen=True)
class SourceRef:
    name: str
    kind: SourceKind = "docs"
    uri: str | None = None
    version: str | None = None  # content fingerprint → Plan-Cache key (SPEC §7)


@dataclass
class RawAsset:
    """One unit of data pulled from a SourcePlugin, before extraction (SPEC §4.8).

    A source yields RawAssets (a file on disk, a row from a dlt connector, an API
    record). An ExtractorPlugin turns each into normalized text; a QualityPlugin may
    clean or drop it before it is chunked and indexed. Exactly one of ``data`` (raw
    bytes needing extraction) or ``text`` (already-textual payload) is typically set.
    """

    id: str                                   # stable id within the source
    uri: str | None = None                    # where it came from (path/url/table)
    data: bytes | None = None                 # raw bytes (PDF/image/audio…) to extract
    text: str | None = None                   # already-textual payload (row/api record)
    mime: str | None = None                   # hint, e.g. "application/pdf"
    label: str | None = None                  # human provenance label
    meta: dict = field(default_factory=dict)  # arbitrary source metadata


@dataclass(frozen=True)
class Goal:
    text: str
    sources: tuple[SourceRef, ...] = ()
    constraints: Constraints = field(default_factory=Constraints)
    conversation_id: str | None = None


# ──────────────────────────── §2.2 intent ────────────────────────────

IntentBucket = Literal[
    "exact_lookup", "conceptual", "incident", "code_reasoning",
    "synthesis", "high_risk", "sensitive", "multi_hop", "temporal", "unknown",
]


@dataclass(frozen=True)
class Intent:
    bucket: IntentBucket
    entities: tuple[str, ...] = ()
    risk: Literal["low", "medium", "high"] = "low"
    normalized: str = ""           # deterministic canonical form → cache key
    confidence: float = 0.0
    # v4: the knowledge representation this request is about — the decision engine's first
    # axis, chosen before any retrieval method. Candidate generation is constrained to the
    # methods that specialize this representation (see planner/representations.py).
    representation: "KnowledgeRepresentation" = "document"


# ──────────────────────────── §2.3 candidate / plan ────────────────────────────

StepType = Literal["retrieve", "rerank", "compress", "route", "reason", "delegate", "verify"]


@dataclass(frozen=True)
class StepSpec:
    type: StepType
    params: dict[str, Any] = field(default_factory=dict)
    plugin: str | None = None


@dataclass(frozen=True)
class Candidate:
    steps: tuple[StepSpec, ...]
    model_tier: str = "local"


@dataclass(frozen=True)
class PlanScore:
    """The soft objective (SPEC §3). Estimates normalized to [0,1] before weighting."""

    expected_accuracy: float = 0.0
    cache_hit_probability: float = 0.0
    verification_confidence: float = 0.0
    cost_usd: float = 0.0          # raw $; normalized at scoring time
    latency_seconds: float = 0.0   # raw s; normalized at scoring time
    risk: float = 0.0
    hallucination_probability: float = 0.0
    context_loss: float = 0.0
    total: float = 0.0             # the weighted PlanScore — what's maximized
    feasible: bool = True


@dataclass(frozen=True)
class Plan:
    intent: Intent
    chosen: Candidate
    score: PlanScore
    rejected: tuple[tuple[Candidate, str], ...] = ()
    cache: Literal["hit", "miss", "bypass"] = "miss"
    id: str = field(default_factory=lambda: new_id("plan"))
    spec_version: str = SPEC_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────── §4.4/4.5 retrieval + model ────────────────────────────

Retrieval = Literal[
    "vector", "bm25", "hybrid", "graph", "community", "image", "colpali", "video",
    # structured-store methods (the `analytical` representation): relational + the NoSQL/search
    # backends a deployment may plug in where SQL is not applicable.
    "sql", "mongo", "elastic", "api", "logs", "file", "code", "temporal",
]

# The knowledge *representation* a retrieval method operates over. The planner's first
# decision is which representation best answers the request (a document, a graph of
# relationships, a bi-temporal fact history, an analytical/OLAP cube, code, or media);
# the concrete `Retrieval` method is a specialization within the chosen representation.
# Retrieval is therefore one family of knowledge access, not the whole of it.
KnowledgeRepresentation = Literal[
    "document", "graph", "temporal", "analytical", "community", "code", "multimodal",
]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    filename: str
    text: str
    score: float = 0.0
    created_at: str | None = None
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    max_context_tokens: int = 8192
    prompt_cache: bool = False
    tool_calling: bool = False
    structured_outputs: bool = False
    vision: bool = False


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[dict[str, str], ...]
    capability: str = "draft"
    max_tokens: int = 1024
    system: str | None = None
    tools: tuple[dict, ...] | None = None
    thinking: bool | None = None   # None = model default; True/False toggles a reasoning model's think mode
    temperature: float | None = None   # None = adapter default; the +sc arm sets >0 so its K samples diverge


@dataclass(frozen=True)
class ModelResult:
    text: str
    model: str
    tier: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    est_cost_usd: float = 0.0
    cache_hit: bool = False
    models_used: tuple[str, ...] = ()   # rolls up Reasoner sub-calls (SPEC §4.4)


# Reasoning/generation strategies. `single_shot` is the legacy default (one cite-based call). The
# generation-strategy layer (CR_GENSTRATEGY) adds the answer-plane arms the bandit selects per intent:
#   terse (extractive, no-think) · reason (think+CoT) · decompose (multi-hop) · mapreduce (aggregation).
ReasoningStrategy = Literal[
    "single_shot", "plan_worker_critic", "debate", "tool_loop",
    "terse", "reason", "decompose", "mapreduce",
]


@dataclass(frozen=True)
class ReasonRequest:
    context: "BuiltContext"
    strategy: ReasoningStrategy = "single_shot"
    capability: str = "synthesis"
    constraints: Constraints = field(default_factory=Constraints)


# ──────────────────────────── §4.6 schedule ────────────────────────────


@dataclass(frozen=True)
class Schedule:
    waves: tuple[tuple[str, ...], ...]      # node-ids grouped into ordered parallel waves
    max_concurrency: int = 4
    retry: dict[str, int] = field(default_factory=dict)


# ──────────────────────────── §5 execution graph IR ────────────────────────────

NodeKind = Literal[
    "retrieve", "rerank", "compress", "route", "reason", "delegate",
    "verify", "branch", "loop", "approval", "merge", "rollback",
]


@dataclass(frozen=True)
class GraphNode:
    kind: NodeKind
    params: dict[str, Any] = field(default_factory=dict)
    plugin: str | None = None
    budget_tokens: int | None = None
    id: str = field(default_factory=lambda: new_id("node"))


EdgeKind = Literal["then", "on_success", "on_failure", "on_condition", "parallel"]


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    kind: EdgeKind = "then"
    condition: str | None = None


@dataclass(frozen=True)
class ExecutionGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    plan_id: str = ""
    id: str = field(default_factory=lambda: new_id("xg"))
    spec_version: str = SPEC_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────── §6 trace ────────────────────────────

SpanKind = Literal[
    "intent", "candidate", "optimize", "schedule", "retrieve", "reason",
    "compress", "verify", "delegate", "cache",
]


@dataclass(frozen=True)
class Span:
    name: str
    kind: SpanKind
    start: str
    end: str
    parent_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("span"))


@dataclass(frozen=True)
class Trace:
    plan_id: str
    goal_text: str
    spans: tuple[Span, ...] = ()
    actual_cost_usd: float = 0.0
    actual_latency_seconds: float = 0.0
    actual_tokens: int = 0
    citations: tuple[str, ...] = ()
    verification_passed: bool | None = None
    cache: Literal["hit", "miss", "bypass"] = "miss"
    id: str = field(default_factory=lambda: new_id("trace"))
    spec_version: str = SPEC_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


# ──────────────────────────── §3.1 cost-model statistics ────────────────────────────


@dataclass(frozen=True)
class FieldStatistics:
    field: str
    mean_absolute_error: float = 0.0
    calibration: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    sample_count: int = 0
    last_updated: str | None = None


@dataclass(frozen=True)
class CostModelStatistics:
    estimator_version: str
    fields: tuple[FieldStatistics, ...] = ()
    bucket: str | None = None


# ──────────────────────────── §6.2 simulate / §6.1 explain ────────────────────────────


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


@dataclass(frozen=True)
class Simulation:
    expected_cost_usd: Interval
    expected_latency_seconds: Interval
    expected_tokens: Interval
    expected_confidence: float
    expected_models: tuple[str, ...]
    expected_retrieval: tuple[Retrieval, ...]
    plan_id: str
    based_on_samples: int


@dataclass(frozen=True)
class Explanation:
    intent: Intent
    candidates: tuple[tuple[Candidate, PlanScore], ...]
    chosen: Plan
    retrieved_sources: tuple[str, ...]
    rejected_sources: tuple[tuple[str, str], ...]
    token_budget: dict[str, int]
    estimated: PlanScore
    statistics: CostModelStatistics
    plan_cache: Literal["hit", "miss", "bypass"] = "miss"
    analyze: Trace | None = None


# ──────────────────────────── assembled context / result ────────────────────────────


@dataclass(frozen=True)
class Compressed:
    text: str
    tokens: int
    derived_from: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    refresh_after: str | None = None


@dataclass(frozen=True)
class Verdict:
    passed: bool
    confidence: float = 0.0
    findings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuiltContext:
    """A Plan turned into assembled context + the graph to execute (SPEC §9)."""

    plan: Plan
    graph: ExecutionGraph
    hits: tuple[Hit, ...] = ()
    assembled_text: str = ""
    token_budget: dict[str, int] = field(default_factory=dict)


@dataclass
class RunResult:
    answer: str
    plan: Plan
    trace: Trace
    citations: tuple[str, ...] = ()
    cost_usd: float = 0.0
    verdict: Verdict | None = None


@dataclass(frozen=True)
class PluginInfo:
    name: str
    kind: Literal[
        "model", "reasoner", "store", "retriever", "scheduler", "knowledge",
        "compression", "verifier", "router", "policy", "planner",
        "source", "extractor", "quality",
    ]
    version: str = "0.1"
    capabilities: frozenset[str] = frozenset()
