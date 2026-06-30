"""Intent Analyzer — "what does the user want?" (SPEC §4.2, the first planner stage).

v0.1 is a rule-table classifier (cheap, fast, deterministic, cacheable). Later it may
become the cheapest model tier; the contract is unchanged.
"""
from __future__ import annotations

from ..types import Goal, Intent, PluginInfo
from . import rules


class RuleIntentAnalyzer:
    def analyze(self, goal: Goal) -> Intent:
        bucket, risk = rules.classify(goal.text)
        # an explicit restricted source upgrades the bucket to sensitive
        if any(s.kind == "memory" or "restrict" in (s.name or "") for s in goal.sources):
            pass
        if goal.constraints.sensitivity == "restricted":
            bucket, risk = "sensitive", "high"
        entities = rules.extract_entities(goal.text)
        # confidence: a non-unknown bucket or a matched entity is a stronger signal
        confidence = 0.8 if bucket != "unknown" else 0.3
        if entities:
            confidence = min(1.0, confidence + 0.1)
        return Intent(
            bucket=bucket,
            entities=entities,
            risk=risk,  # type: ignore[arg-type]
            normalized=rules.normalize(goal.text),
            confidence=confidence,
        )

    def info(self) -> PluginInfo:
        return PluginInfo(name="rule_intent", kind="planner", capabilities=frozenset({"intent"}))
