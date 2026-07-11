"""Candidate Generator — "what plans are even possible?" (SPEC §4.2, stage two).

Enumerates possible plans (retrieval method × model tier × reasoning strategy ×
verification), then rule-prunes the impossible/forbidden before the optimizer scores
the survivors.
"""
from __future__ import annotations

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
        c = goal.constraints
        # citations are checked by the verify step, so requiring them implies verify
        require_verify = want_verify or c.require_verification or c.require_citations

        out: list[Candidate] = []
        for method in methods:
            for tier in tiers:
                steps: list[StepSpec] = [
                    StepSpec("retrieve", {"method": method, "top_k": self.default_top_k}),
                ]
                if method in ("hybrid", "vector", "code"):
                    steps.append(StepSpec("rerank", {"final_k": self.final_k}))
                steps.append(StepSpec("compress", {"target_tokens": self.target_tokens}))
                steps.append(StepSpec("route", {"tier": tier}))
                steps.append(StepSpec("reason", {"strategy": strategy, "capability": "synthesis"}))
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
