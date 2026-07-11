"""Registry-backed candidate generator — the same plan shapes as RuleCandidateGenerator, but the
retrieval methods and model tiers are drawn from a CapabilityRegistry instead of the hard-coded rule
tables. Adding a capability (a new retrieval method, a new tier) becomes a ``registry.register()`` call
that immediately widens the planner's search space — with no edit to rules.py and nothing added to the
model's context window (Whitepaper v3: "capabilities live in the planner, not the prompt").

The reasoning strategy and the verify requirement remain per-intent planner policy (from rules); this
generator governs which *capabilities* are composed, which is the part that scales.
"""
from __future__ import annotations

from ..registry import CapabilityRegistry
from ..types import Candidate, Goal, Intent, PluginInfo, StepSpec
from . import rules


class RegistryCandidateGenerator:
    def __init__(
        self,
        registry: CapabilityRegistry | None = None,
        default_top_k: int = 50,
        final_k: int = 8,
        target_tokens: int = 3000,
    ):
        self.registry = registry or CapabilityRegistry.from_rules()
        self.default_top_k = default_top_k
        self.final_k = final_k
        self.target_tokens = target_tokens

    def generate(self, intent: Intent, goal: Goal) -> list[Candidate]:
        _methods, strategy, want_verify = rules.BUCKET_DEFAULTS.get(
            intent.bucket, rules.BUCKET_DEFAULTS["unknown"]
        )
        methods = self.registry.values_for(intent.bucket, "retrieval") or ["hybrid"]
        tiers = self.registry.values_for(intent.bucket, "model_tier") or ["local"]

        c = goal.constraints
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
            if c.sensitivity == "restricted" and cand.model_tier != "local":
                continue
            if c.require_citations and not any(s.type == "verify" for s in cand.steps):
                continue
            kept.append(cand)
        return kept or candidates[:1]

    def info(self) -> PluginInfo:
        return PluginInfo(name="registry_candidates", kind="planner", capabilities=frozenset({"candidates"}))
