"""Candidate Generator — "what plans are even possible?" (SPEC §4.2, stage two).

Enumerates possible plans (retrieval method × model tier × reasoning strategy ×
verification), then rule-prunes the impossible/forbidden before the optimizer scores
the survivors.
"""
from __future__ import annotations

from ..reasoner import strategies as genstrat
from ..types import Candidate, Goal, Intent, PluginInfo, StepSpec
from . import representations, rules

# below this intent confidence the generator WIDENS beyond the chosen representation so the
# bandit can explore other representations and learn which one wins for this context.
EXPLORE_CONFIDENCE = 0.5


class RuleCandidateGenerator:
    def __init__(self, default_top_k: int = 50, final_k: int = 8, target_tokens: int = 3000):
        self.default_top_k = default_top_k
        self.final_k = final_k
        self.target_tokens = target_tokens

    def _methods_for_intent(self, intent: Intent, bucket_methods: tuple) -> tuple:
        """v4: constrain to the chosen knowledge representation instead of the flat bucket table.

        Representation-first, with two safety valves: a document (`hybrid`) fallback so a missing or
        infeasible representation engine degrades gracefully (the cost model prunes the infeasible
        one), and confidence-gated widening so uncertain intents still explore across representations
        (that exploration is how the bandit LEARNS representation selection)."""
        rep = getattr(intent, "representation", "document")
        # the bucket's own methods that already belong to the chosen representation, in bucket order…
        primary = tuple(m for m in bucket_methods if representations.representation_for(m) == rep)
        if not primary:                                   # …else the representation's own methods
            primary = representations.methods_for(rep)
        fallback = () if rep == "document" else ("hybrid",)
        widen = bucket_methods if intent.confidence < EXPLORE_CONFIDENCE else ()
        methods = tuple(dict.fromkeys(primary + fallback + widen))   # de-dupe, representation-first
        return methods or bucket_methods

    def generate(self, intent: Intent, goal: Goal) -> list[Candidate]:
        bucket_methods, strategy, want_verify = rules.BUCKET_DEFAULTS[intent.bucket]
        methods = self._methods_for_intent(intent, bucket_methods)
        tiers = rules.BUCKET_TIERS[intent.bucket]
        # Effort-up vs model-up (B): also offer the ladder at the premium tier, so the bandit weighs
        # "more effort, same model" against "bigger model". Restricted/sensitive stays local.
        if genstrat.effort_vs_model() and "premium" not in tiers and intent.bucket != "sensitive":
            tiers = tiers + ("premium",)
        c = goal.constraints
        # citations are checked by the verify step, so requiring them implies verify
        require_verify = want_verify or c.require_verification or c.require_citations

        # Generation-strategy layer (CR_GENSTRATEGY): the `reason` step becomes a bandit arm too — one
        # candidate per (method × tier × generation strategy), the strategies seeded per intent bucket.
        # Off → the single legacy strategy from BUCKET_DEFAULTS, so plans are byte-identical to before.
        gen_strats = genstrat.strategies_for(intent.bucket) if genstrat.enabled() else (strategy,)

        # Verification Optimizer: correctness-sensitive classes ALSO get a self-checked variant of each
        # strategy (a distinct arm) so the bandit can learn whether the self-check earns its cost here.
        verify_opts = (False, True) if genstrat.offers_verify(intent.bucket) else (False,)
        # Self-consistency arm (A): a +sc variant samples K reasoning traces → consensus. Off → 0.
        sc_k = genstrat.self_consistency_k() if genstrat.offers_self_consistency(intent.bucket) else 0

        out: list[Candidate] = []
        for method in methods:
            for tier in tiers:
                for gstrat in gen_strats:
                    # sc only on thinking strategies — sampling a terse no-think answer buys nothing.
                    sc_opts = (0, sc_k) if (sc_k >= 2 and genstrat.enabled() and genstrat.get(gstrat).thinking) else (0,)
                    for do_verify in verify_opts:
                        for sc in sc_opts:
                            steps: list[StepSpec] = [
                                StepSpec("retrieve", {"method": method, "top_k": self.default_top_k}),
                            ]
                            if method in ("hybrid", "vector", "code"):
                                steps.append(StepSpec("rerank", {"final_k": self.final_k}))
                            steps.append(StepSpec("compress", {"target_tokens": self.target_tokens}))
                            steps.append(StepSpec("route", {"tier": tier}))
                            reason_params = {"strategy": gstrat, "capability": "synthesis"}
                            if genstrat.enabled():
                                gs = genstrat.get(gstrat)
                                reason_params.update(thinking=gs.thinking, max_tokens=gs.max_tokens)
                                if do_verify:
                                    reason_params["verify"] = True
                                if sc >= 2:
                                    reason_params["self_consistency"] = sc
                            steps.append(StepSpec("reason", reason_params))
                            if require_verify:
                                steps.append(StepSpec("verify", {"method": "citation"}))
                            out.append(Candidate(steps=tuple(steps), model_tier=tier))
        return out

    def prune(self, candidates: list[Candidate], goal: Goal) -> list[Candidate]:
        c = goal.constraints
        kept: list[Candidate] = []
        for cand in candidates:
            # sensitive/restricted data MUST stay local (hard rule, not a score penalty)
            if c.sensitivity == "restricted" and cand.model_tier != "local":
                continue
            # require_citations implies a verify step must exist
            if c.require_citations and not any(s.type == "verify" for s in cand.steps):
                continue
            kept.append(cand)
        return kept or candidates[:1]   # never prune to empty

    def info(self) -> PluginInfo:
        return PluginInfo(name="rule_candidates", kind="planner", capabilities=frozenset({"candidates"}))
