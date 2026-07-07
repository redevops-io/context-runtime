"""ContextRuntime — the orchestrator (SPEC §9).

Wires the plugins and implements the lifecycle:
    Goal → Intent → Plan → Execution Graph → Schedule → Retrieve/Compress/Reason/Verify
         → Response + Trace + (cost-model observe)

``run`` = plan → build_context → execute → verify, composed. The granular methods are
exposed for inspection and reuse. ``plan``/``explain``/``simulate`` never execute.
"""
from __future__ import annotations

from dataclasses import replace

from ..compression.structural import StructuralCompressor
from ..costmodel.estimators import HeuristicEstimator
from ..execution import graph as graphmod
from ..observability import traces
from ..optimizer.knapsack import KnapsackOptimizer
from ..plancache.cache import NullPlanCache, build_key
from ..planner.candidates import RuleCandidateGenerator
from ..planner.intent import RuleIntentAnalyzer
from ..reasoner.single_shot import SingleShotReasoner
from ..scheduler.schedule import TopoScheduler
from ..types import (
    BuiltContext,
    Candidate,
    Constraints,
    Explanation,
    Goal,
    Interval,
    Plan,
    PlanScore,
    ReasonRequest,
    RunResult,
    Simulation,
    SourceRef,
)
from ..verification.verifier import CitationVerifier
from .config import Config

_TIER_MODELS = ("local", "cheap", "premium")


class ContextRuntime:
    def __init__(self, *, models, retriever, estimator=None, config: Config | None = None,
                 intent=None, candidates=None, optimizer=None, compressor=None,
                 scheduler=None, verifier=None, plan_cache=None, exporter=None):
        self.config = config or Config()
        # models: a dict tier->ModelPlugin, or a single ModelPlugin used for all tiers
        if not isinstance(models, dict):
            models = {t: models for t in _TIER_MODELS}
        self.models = models
        self.retriever = retriever
        self.estimator = estimator or HeuristicEstimator(stats_path=self.config.stats_path)
        self.intent = intent or RuleIntentAnalyzer()
        self.candidates = candidates or RuleCandidateGenerator(
            default_top_k=self.config.top_k, final_k=self.config.final_k,
            target_tokens=self.config.target_tokens,
        )
        self.optimizer = optimizer or KnapsackOptimizer(self.estimator)
        self.compressor = compressor or StructuralCompressor()
        self.scheduler = scheduler or TopoScheduler()
        self.verifier = verifier or CitationVerifier()
        self.plan_cache = plan_cache or NullPlanCache()
        self.exporter = exporter        # optional TraceExporter (Langfuse/OTel/JSONL)

    # ──────────────────────────── construction ────────────────────────────

    @classmethod
    def default(cls, docs: list[dict] | None = None, **kw) -> "ContextRuntime":
        """Fully-offline runtime: stub models per tier + in-memory store."""
        from ..adapters.model_stub import StubModel
        from ..adapters.store_inmemory import InMemoryStore

        models = {t: StubModel(tier=t) for t in _TIER_MODELS}
        return cls(models=models, retriever=InMemoryStore(docs or []), **kw)

    @classmethod
    def from_config(cls, path: str) -> "ContextRuntime":
        cfg = Config.from_yaml(path)
        # model plugin
        if cfg.model == "litellm":
            from ..adapters.model_litellm import LiteLLMModel
            models = LiteLLMModel(default_tier=cfg.default_tier)
        else:
            from ..adapters.model_stub import StubModel
            models = {t: StubModel(tier=t) for t in _TIER_MODELS}
        # store plugin
        if cfg.store == "redevops_rag":
            from ..adapters.store_redevops import RedevopsRagRetriever
            retriever = RedevopsRagRetriever()
        else:
            from ..adapters.store_inmemory import InMemoryStore
            retriever = InMemoryStore([])
        return cls(models=models, retriever=retriever, config=cfg)

    # ──────────────────────────── planning (no execution) ────────────────────────────

    def _coerce_goal(self, goal, sources, constraints) -> Goal:
        if isinstance(goal, Goal):
            return goal
        srcs = tuple(s if isinstance(s, SourceRef) else SourceRef(name=str(s)) for s in (sources or ()))
        cons = constraints or Constraints()
        if isinstance(cons, dict):
            cons = Constraints(**cons)
        return Goal(text=str(goal), sources=srcs, constraints=cons)

    def _make_plan(self, goal: Goal):
        intent = self.intent.analyze(goal)
        cands = self.candidates.prune(self.candidates.generate(intent, goal), goal)
        scored: list[tuple[Candidate, PlanScore]] = [(c, self.optimizer.score(c, goal)) for c in cands]
        # pass the intent bucket as the selection context — an online (Gen-4) optimizer keys its
        # contextual bandit on it; the static optimizer ignores it.
        plan = self.optimizer.select(scored, goal, context=intent.bucket)
        plan = replace(plan, intent=intent)   # attach the real intent
        return plan, scored, intent

    def plan(self, goal, *, sources=None, constraints=None) -> Plan:
        g = self._coerce_goal(goal, sources, constraints)
        # consult the (v0.1 null) plan cache
        plan, _, intent = self._make_plan(g)
        key = build_key(intent, g)
        cached = self.plan_cache.get(key)
        if cached is not None:
            return replace(cached, cache="hit")
        self.plan_cache.put(key, plan)
        return plan

    # ──────────────────────────── context assembly ────────────────────────────

    def build_context(self, plan: Plan, goal: Goal | None = None) -> BuiltContext:
        steps = {s.type: s.params for s in plan.chosen.steps}
        method = steps.get("retrieve", {}).get("method", "hybrid")
        top_k = steps.get("retrieve", {}).get("top_k", self.config.top_k)
        final_k = steps.get("rerank", {}).get("final_k", self.config.final_k)
        target_tokens = steps.get("compress", {}).get("target_tokens", self.config.target_tokens)

        query = plan.intent.normalized or goal.text if goal else plan.intent.normalized
        hits = self.retriever.search(query, k=top_k, method=method)
        if "rerank" in steps:
            hits = hits[:final_k]

        packed = self.compressor.assemble(hits, target_tokens=target_tokens)
        kept = sum(max(1, len(h.text) // 4) for h in hits)
        budget = {"keep": kept, "compress": packed.tokens, "drop": max(0, kept - packed.tokens)}

        g = graphmod.build(plan.id, plan.chosen)
        return BuiltContext(
            plan=plan, graph=g, hits=tuple(hits),
            assembled_text=packed.text, token_budget=budget,
        )

    # ──────────────────────────── execution ────────────────────────────

    def execute(self, ctx: BuiltContext, goal: Goal | None = None) -> RunResult:
        tb = traces.TraceBuilder(ctx.plan.id, goal.text if goal else ctx.plan.intent.normalized)

        # schedule (decides when/where; recorded for inspection)
        t0 = traces.now()
        sched = self.scheduler.schedule(ctx.graph, (goal.constraints if goal else Constraints()))
        tb.span("schedule", "schedule", {"waves": len(sched.waves)}, t0, traces.now())

        # retrieve span (work done in build_context; record the outcome)
        tb.span("retrieve", "retrieve", {"hits": len(ctx.hits), "tokens": ctx.token_budget.get("compress", 0)},
                t0, traces.now())

        # reason (Reasoner → Router → Model)
        tier = ctx.plan.chosen.model_tier
        model = self.models.get(tier) or next(iter(self.models.values()))
        reasoner = SingleShotReasoner(model)
        t1 = traces.now()
        result = reasoner.reason(ReasonRequest(context=ctx, capability="synthesis",
                                               constraints=(goal.constraints if goal else Constraints())))
        tb.span("reason", "reason", {"model": result.model, "tier": result.tier,
                                     "tokens": result.prompt_tokens + result.completion_tokens,
                                     "cost_usd": result.est_cost_usd}, t1, traces.now())
        tb.add_cost(result.est_cost_usd, result.prompt_tokens + result.completion_tokens)

        # verify (part of execution when the plan requires it)
        verdict = None
        if any(s.type == "verify" for s in ctx.plan.chosen.steps):
            t2 = traces.now()
            verdict = self.verifier.verify(result, ctx.plan, ctx)
            tb.span("verify", "verify", {"passed": verdict.passed, "confidence": verdict.confidence,
                                         "findings": list(verdict.findings)}, t2, traces.now())
            tb.set_verified(verdict.passed)

        citations = tuple(h.chunk_id for h in ctx.hits)
        tb.set_citations(citations)
        trace = tb.finalize()

        # close the loop: record estimate-vs-actual into the cost-model statistics
        self.estimator.observe(ctx.plan, trace)
        if self.config.trace_dir:
            traces.save_trace(trace, self.config.trace_dir)
        if self.exporter is not None:   # ship to Langfuse / OTel / JSONL
            try:
                self.exporter.export(trace)
            except Exception:
                pass

        return RunResult(
            answer=result.text, plan=ctx.plan, trace=trace, citations=citations,
            cost_usd=trace.actual_cost_usd, verdict=verdict,
        )

    def verify(self, result: RunResult) -> RunResult:
        """Idempotent: verification already ran inside execute when required."""
        return result

    # ──────────────────────────── the core command ────────────────────────────

    def run(self, goal, *, sources=None, constraints=None) -> RunResult:
        g = self._coerce_goal(goal, sources, constraints)
        plan = self.plan(g)
        ctx = self.build_context(plan, g)
        return self.verify(self.execute(ctx, g))

    # ──────────────────────────── explain / simulate (no execution) ────────────────────────────

    def explain(self, goal, *, analyze: bool = False, sources=None, constraints=None) -> Explanation:
        g = self._coerce_goal(goal, sources, constraints)
        plan, scored, intent = self._make_plan(g)
        chosen = plan.chosen
        rejected_sources = tuple(
            (s.name, "not selected by chosen retrieval method") for s in g.sources
        )
        retrieved = tuple(s.name for s in g.sources) or ("(default store)",)
        token_budget = {"target_tokens": self.config.target_tokens, "final_k": self.config.final_k}

        analyze_trace = None
        if analyze:
            ctx = self.build_context(plan, g)
            analyze_trace = self.execute(ctx, g).trace

        return Explanation(
            intent=intent,
            candidates=tuple(scored),
            chosen=plan,
            retrieved_sources=retrieved,
            rejected_sources=rejected_sources,
            token_budget=token_budget,
            estimated=chosen and plan.score,
            statistics=self.estimator.statistics(intent.bucket),
            plan_cache=plan.cache,
            analyze=analyze_trace,
        )

    def simulate(self, goal, *, sources=None, constraints=None) -> Simulation:
        g = self._coerce_goal(goal, sources, constraints)
        plan, _, intent = self._make_plan(g)
        s = plan.score
        samples = getattr(getattr(self.estimator, "stats", None), "samples", lambda: 0)()

        def interval(field_name: str, point: float) -> Interval:
            stats = getattr(self.estimator, "stats", None)
            if stats is not None and samples > 0:
                _, low, high = stats.interval(field_name)
                # recentre on this plan's estimate, keep the learned half-width
                half = max((high - low) / 2, 0.0)
                return Interval(point=round(point, 4), low=round(max(0.0, point - half), 4),
                                high=round(point + half, 4))
            # cold start: honest wide ±50%
            return Interval(point=round(point, 4), low=round(point * 0.5, 4), high=round(point * 1.5, 4))

        method = next((st.params.get("method") for st in plan.chosen.steps if st.type == "retrieve"), "hybrid")
        return Simulation(
            expected_cost_usd=interval("cost_usd", s.cost_usd),
            expected_latency_seconds=interval("latency_seconds", s.latency_seconds),
            expected_tokens=interval("expected_accuracy", float(self.config.target_tokens)),
            expected_confidence=s.expected_accuracy,
            expected_models=(plan.chosen.model_tier,),
            expected_retrieval=(method,),
            plan_id=plan.id,
            based_on_samples=samples,
        )

    # ──────────────────────────── helpers ────────────────────────────

    def index(self, path: str) -> dict:
        return self.retriever.index(path)
