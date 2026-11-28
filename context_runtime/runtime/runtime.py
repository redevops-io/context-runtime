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
from ..learning.reward import DefaultReward
from ..optimizer.knapsack import KnapsackOptimizer
from ..plancache.cache import NullPlanCache, SnapshotPlanCache, build_key
from ..planner.candidates import RuleCandidateGenerator
from ..planner import representations
from ..planner.intent import RuleIntentAnalyzer
from ..plugins.base import NoEligiblePlan
from ..types import Limitation
from ..reasoner.single_shot import SingleShotReasoner
from ..reasoner.strategies import reasoner_for
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
from ..verification.narration import facts_from_hits, serve_verified
from ..verification.verifier import CitationVerifier
from .config import Config

_TIER_MODELS = ("local", "cheap", "premium")


class ContextRuntime:
    def __init__(self, *, models, retriever, estimator=None, config: Config | None = None,
                 intent=None, candidates=None, optimizer=None, compressor=None,
                 scheduler=None, verifier=None, plan_cache=None, exporter=None,
                 learning: bool = False, reward=None, freshness_policy=None):
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
        # learning=True puts the online BanditOptimizer on the DEFAULT serving path (the audit's gap:
        # the machinery existed but the runtime defaulted to the static knapsack optimizer and never
        # fed reward back). execute() then closes the loop via the shared RewardContract below.
        self.learning = learning
        if optimizer is not None:
            self.optimizer = optimizer
        elif learning:
            from ..optimizer.online import BanditOptimizer
            self.optimizer = BanditOptimizer(self.estimator, context_of=lambda g: "default")
        else:
            self.optimizer = KnapsackOptimizer(self.estimator)
        self.reward = reward or DefaultReward()
        self.compressor = compressor or StructuralCompressor()
        self.scheduler = scheduler or TopoScheduler()
        self.verifier = verifier or CitationVerifier()
        # Deterministic-replay plan cache by default (keyed on source_fingerprint = pinned evidence
        # identity). Under online learning the optimizer must re-select every call, so caching is off
        # (NullPlanCache) unless the caller injects one explicitly.
        self.plan_cache = plan_cache or (NullPlanCache() if learning else SnapshotPlanCache())
        self.exporter = exporter        # optional TraceExporter (Langfuse/OTel/JSONL)
        # Evidence-sourced freshness on the DEFAULT serving path (v0.2.x freshness-from-evidence).
        # None / disabled → freshness stays 1.0 and execute() is byte-for-byte the legacy path; set a
        # FreshnessPolicy to derive freshness from retrieved evidence and gate REFRESH at serve time.
        self.freshness_policy = freshness_policy

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
        cands = self._exact_feasible(cands, goal)
        scored: list[tuple[Candidate, PlanScore]] = [(c, self.optimizer.score(c, goal)) for c in cands]
        # pass the intent bucket as the selection context — an online (Gen-4) optimizer keys its
        # contextual bandit on it; the static optimizer ignores it.
        plan = self.optimizer.select(scored, goal, context=intent.bucket)
        plan = replace(plan, intent=intent)   # attach the real intent
        return plan, scored, intent

    def _method_capabilities(self, method: str) -> frozenset:
        """Capabilities the bound provider declares for a retrieval method. A router answers per method; a plain retriever declares the
        methods it serves plus any capability tokens in its ``info().capabilities``."""
        per_method = getattr(self.retriever, "method_capabilities", None)
        if per_method is not None:
            return frozenset(per_method(method))
        info = getattr(self.retriever, "info", None)
        caps = frozenset(info().capabilities) if info is not None else frozenset()
        return frozenset({"exact"}) & caps if method in caps else frozenset()

    def _exact_feasible(self, cands, goal):
        """Feasibility BEFORE cost: with require_exact / require_complete only plans whose retrieval method is served by a provider that
        declares "exact" may be scored. If none exists the request fails with a typed NoEligiblePlan - never a top-k fallback."""
        c = goal.constraints
        if not (c.require_exact or c.require_complete):
            return cands
        methods = [x.steps[0].params.get("method", "hybrid") for x in cands]
        ok = [x for x, m in zip(cands, methods) if "exact" in self._method_capabilities(m)]
        if not ok and cands:
            # The generator restricts candidates to the intent's representation. A caller that REQUIRES exactness has said the answer must
            # be computed, so widen to every bound exact method (same plan shape, no rerank of an exact result) before giving up.
            from ..planner import representations
            from ..types import StepSpec
            for m in representations.methods_for("analytical"):
                if "exact" in self._method_capabilities(m):
                    base = cands[0]
                    steps = tuple(st for st in base.steps if st.type != "rerank")
                    steps = (StepSpec("retrieve", {**steps[0].params, "method": m}),) + steps[1:]
                    ok.append(replace(base, steps=steps))
        if not ok:
            raise NoEligiblePlan("no bound provider declares the 'exact' capability for any candidate retrieval method", sorted(set(methods)))
        return ok

    def plan(self, goal, *, sources=None, constraints=None) -> Plan:
        g = self._coerce_goal(goal, sources, constraints)
        # the fresh selection runs first (with a policy it raises PolicyDenied when nothing is permitted), then the cache
        plan, _, intent = self._make_plan(g)
        key = build_key(intent, g)
        cached = self.plan_cache.get(key)
        if cached is not None and self._may_replay(cached, plan, g):
            return replace(cached, cache="hit")
        self.plan_cache.put(key, plan)
        return plan

    def _may_replay(self, cached: Plan, fresh: Plan, goal: Goal) -> bool:
        """Whether a cached plan may be executed NOW. The cache key covers the goal, its sources, constraints, tenant and
        permissions - not what policy decides. Policy can change while a process runs, and two identities can share a key,
        so a replayed plan is asked again: policy must still permit it, and the optimizer must not rank it infeasible when
        it has a feasible plan to offer. Anything else falls back to the fresh selection, which then replaces it."""
        score = self.optimizer.score(cached.chosen, goal)
        policy = getattr(self.optimizer, "policy", None)
        if policy is not None and policy.feasible(cached.chosen, goal, score) is not None:
            return False
        return score.feasible or not fresh.score.feasible

    # ──────────────────────────── context assembly ────────────────────────────

    def build_context(self, plan: Plan, goal: Goal | None = None) -> BuiltContext:
        steps = {s.type: s.params for s in plan.chosen.steps}
        method = steps.get("retrieve", {}).get("method", "hybrid")
        top_k = steps.get("retrieve", {}).get("top_k", self.config.top_k)
        final_k = steps.get("rerank", {}).get("final_k", self.config.final_k)
        target_tokens = steps.get("compress", {}).get("target_tokens", self.config.target_tokens)

        natural = goal is not None and method in getattr(self.retriever, "natural_query_methods", ())
        query = goal.text if natural else (plan.intent.normalized or goal.text if goal else plan.intent.normalized)
        hits = self.retriever.search(query, k=top_k, method=method)
        if "rerank" in steps:
            hits = hits[:final_k]

        packed = self.compressor.assemble(hits, target_tokens=target_tokens)
        kept = sum(max(1, len(h.text) // 4) for h in hits)
        budget = {"keep": kept, "compress": packed.tokens, "drop": max(0, kept - packed.tokens)}

        limitation = None
        if goal is not None and (goal.constraints.require_exact or goal.constraints.require_complete):
            if not hits:
                limitation = Limitation("empty_exact_result", "the exact query returned no rows, so no value can be asserted", (method,))
            elif goal.constraints.require_complete and any((h.meta or {}).get("at_row_cap") for h in hits):
                limitation = Limitation("incomplete_evidence", "the exact result reached the provider's row cap and may be truncated", (method,))

        g = graphmod.build(plan.id, plan.chosen)
        return BuiltContext(
            plan=plan, graph=g, hits=tuple(hits),
            assembled_text=packed.text, token_budget=budget, limitation=limitation,
            question=(goal.text if goal is not None and (self.config.natural_question or goal.constraints.require_exact
                                                          or goal.constraints.require_complete) else None),
        )

    # ──────────────────────────── execution ────────────────────────────

    def execute(self, ctx: BuiltContext, goal: Goal | None = None, *, as_of=None) -> RunResult:
        tb = traces.TraceBuilder(ctx.plan.id, goal.text if goal else ctx.plan.intent.normalized)
        if ctx.limitation is not None:      # typed limitation: no model call, no prose answer
            return RunResult(answer="", plan=ctx.plan, trace=tb.finalize(), limitation=ctx.limitation, freshness=ctx.plan.score.freshness)

        # schedule (decides when/where; recorded for inspection)
        t0 = traces.now()
        sched = self.scheduler.schedule(ctx.graph, (goal.constraints if goal else Constraints()))
        tb.span("schedule", "schedule", {"waves": len(sched.waves)}, t0, traces.now())

        # retrieve span (work done in build_context; record the outcome)
        tb.span("retrieve", "retrieve", {"hits": len(ctx.hits), "tokens": ctx.token_budget.get("compress", 0)},
                t0, traces.now())

        # freshness gate (v0.2.x freshness-from-evidence). Derive freshness from the retrieved
        # evidence's source timestamps, record it on the plan score, and — if a policy is configured
        # and the evidence is staler than its bar — REFRESH instead of serving. Disabled (no policy)
        # ⇒ this whole block is skipped and the path is byte-for-byte the legacy one.
        if self.freshness_policy is not None and getattr(self.freshness_policy, "enabled", False):
            from .. import freshness as _fresh
            from ..abstention import AbstentionGate
            fr = _fresh.score_hits(ctx.hits, as_of=as_of, policy=self.freshness_policy)
            ctx = replace(ctx, plan=replace(ctx.plan, score=replace(ctx.plan.score, freshness=fr)))
            gate = AbstentionGate(min_confidence=self.freshness_policy.min_confidence,
                                  min_freshness=self.freshness_policy.min_freshness)
            verdict = gate.evaluate(ctx.plan.score, goal)
            if verdict.refresh:
                tb.span("freshness", "freshness", {"freshness": round(fr, 4), "action": "refresh"},
                        t0, traces.now())
                return RunResult(
                    answer="", plan=ctx.plan, trace=tb.finalize(), citations=(),
                    freshness=fr, refresh=True, refresh_reason=verdict.reason,
                )

        # reason (Reasoner -> Router -> Model). Unified dispatch over both reasoning lines, keyed by the
        # plan's chosen strategy (SPEC 4.4):
        #   single_shot                            -> SingleShotReasoner (legacy path, byte-for-byte)
        #   plan_worker_critic / debate / tool_loop -> concrete multi-shot reasoners (reasoner_for)
        #   any generation-strategy arm            -> StrategyReasoner (self-consistency / refine / verify)
        # The two strategy namespaces are disjoint, so one classification routes unambiguously.
        tier = ctx.plan.chosen.model_tier
        model = self.models.get(tier) or next(iter(self.models.values()))
        _rstep = next((s for s in ctx.plan.chosen.steps if s.type == "reason"), None)
        strategy = (_rstep.params.get("strategy") if _rstep else None) or "single_shot"
        if strategy == "single_shot":
            reasoner = SingleShotReasoner(model)
        elif strategy in ("plan_worker_critic", "debate", "tool_loop"):
            reasoner = reasoner_for(strategy, model)
        else:
            from ..reasoner.strategy_reasoner import StrategyReasoner
            reasoner = StrategyReasoner(model, strategy,
                                        verify=bool(_rstep and _rstep.params.get("verify")),
                                        samples=int(_rstep.params.get("self_consistency") or 0) if _rstep else 0)
        t1 = traces.now()
        result = reasoner.reason(ReasonRequest(context=ctx, capability="synthesis",
                                               constraints=(goal.constraints if goal else Constraints())))
        narration = None
        facts = facts_from_hits(ctx.hits) if self.config.verify_narration else []
        if facts:   # a provider attached facts: the narration may not omit or change them (serve the structured rendering otherwise)
            structured = "\n".join(h.text for h in ctx.hits if (h.meta or {}).get("facts"))
            served, narration = serve_verified(result.text, facts, structured, question=ctx.question or (goal.text if goal else ""))
            if served != result.text:
                result = replace(result, text=served)
            tb.span("narration", "verify", {"status": narration.status, "findings": list(narration.findings)}, t1, traces.now())
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
        # and — when learning is on — fold a measured reward back into the online optimizer, so the
        # DEFAULT serving path learns which (retrieval × tier × strategy) arm wins per bucket.
        if self.learning and hasattr(self.optimizer, "learn_from_plan"):
            self.optimizer.learn_from_plan(ctx.plan, self.reward.reward(trace, verdict))
        if self.config.trace_dir:
            traces.save_trace(trace, self.config.trace_dir)
        if self.exporter is not None:   # ship to Langfuse / OTel / JSONL
            try:
                self.exporter.export(trace)
            except Exception:
                pass

        return RunResult(
            answer=result.text, plan=ctx.plan, trace=trace, citations=citations,
            cost_usd=trace.actual_cost_usd, verdict=verdict, freshness=ctx.plan.score.freshness, narration=narration,
        )

    def verify(self, result: RunResult) -> RunResult:
        """Idempotent: verification already ran inside execute when required."""
        return result

    # ──────────────────────────── the core command ────────────────────────────

    def _typed_resolver(self):
        r = self.retriever
        r = getattr(r, "typed_resolver", None) or r
        return r if hasattr(r, "resolve") and hasattr(r, "execute") else None

    def _fast_path(self, g: Goal) -> "RunResult | None":
        """Deterministic fast path: only for a request that REQUIRES exactness, is analytical by the existing intent classification, and
        resolves to typed calls that cover everything it asks for. Anything else (and any error) takes the normal path."""
        if not (self.config.fast_path and g.constraints.require_exact) or getattr(self.optimizer, "policy", None) is not None:
            return None
        resolver = self._typed_resolver()
        if resolver is None:
            return None
        intent = self.intent.analyze(g)
        if representations.classify(intent.bucket, g.text) != "analytical":
            return None
        try:
            resolution = resolver.resolve(g.text)
            if resolution is None or not resolution.covered:
                return None
            tb = traces.TraceBuilder("fast-path", g.text)
            t0 = traces.now()
            hits = resolver.execute(resolution)
            if not hits:
                return None
            tb.span("fast_path", "retrieve", {"operations": [c.op.name for c in resolution.calls], "model_calls": 0}, t0, traces.now())
        except Exception:
            return None
        citations = tuple(h.chunk_id for h in hits)
        tb.set_citations(citations)
        return RunResult(answer="\n".join(h.text for h in hits), plan=None, trace=tb.finalize(), citations=citations, path="fast")

    def run(self, goal, *, sources=None, constraints=None, as_of=None) -> RunResult:
        g = self._coerce_goal(goal, sources, constraints)
        fast = self._fast_path(g)
        if fast is not None:
            return fast
        try:
            plan = self.plan(g)
        except NoEligiblePlan as e:
            tb = traces.TraceBuilder("no-plan", g.text)
            return RunResult(answer="", plan=None, trace=tb.finalize(), limitation=Limitation("no_eligible_plan", e.reason, e.considered))
        ctx = self.build_context(plan, g)
        return self.verify(self.execute(ctx, g, as_of=as_of))

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
