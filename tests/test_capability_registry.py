"""Capability registry + registry-backed candidate generation.

Two claims: (1) the default registry (lifted from the rule tables) is behavior-equivalent to the
hand-written RuleCandidateGenerator, and (2) a new capability registered out of band immediately
widens the planner's candidate set — with no change to rules.py and nothing added to any prompt.
"""
from __future__ import annotations

from context_runtime.planner.candidates import RuleCandidateGenerator
from context_runtime.planner.candidates_registry import RegistryCandidateGenerator
from context_runtime.registry import Capability, CapabilityRegistry
from context_runtime.types import Goal, Intent


def _sig(cands):
    """Order-independent signature of a candidate set: (model_tier, retrieval method) pairs."""
    out = set()
    for c in cands:
        method = next((s.params.get("method") for s in c.steps if s.type == "retrieve"), None)
        out.add((c.model_tier, method))
    return out


_BUCKETS = [
    "exact_lookup", "conceptual", "incident", "code_reasoning",
    "synthesis", "high_risk", "sensitive", "multi_hop", "unknown",
]


def test_default_registry_matches_rule_generator_across_buckets():
    rule = RuleCandidateGenerator()
    reg = RegistryCandidateGenerator()  # defaults to CapabilityRegistry.from_rules()
    goal = Goal(text="q")
    for bucket in _BUCKETS:
        intent = Intent(bucket=bucket)
        assert _sig(reg.generate(intent, goal)) == _sig(rule.generate(intent, goal)), bucket


def test_from_rules_registers_expected_capabilities():
    reg = CapabilityRegistry.from_rules()
    methods = {c.value for c in reg.list("retrieval")}
    tiers = {c.value for c in reg.list("model_tier")}
    assert {"bm25", "vector", "hybrid", "code", "graph"} <= methods
    assert {"local", "cheap", "premium"} == tiers
    # graph serves multi_hop; bm25 does not
    assert reg.get("retrieval:graph").serves("multi_hop")
    assert not reg.get("retrieval:bm25").serves("multi_hop")


def test_registering_a_new_capability_widens_the_candidate_set():
    reg = CapabilityRegistry.from_rules()
    gen = RegistryCandidateGenerator(reg)
    goal, intent = Goal(text="q"), Intent(bucket="multi_hop")
    before = _sig(gen.generate(intent, goal))
    assert ("cheap", "temporal") not in before

    # Register a brand-new retrieval capability serving multi_hop — no edit to rules.py, no prompt.
    reg.register(Capability(
        key="retrieval:temporal", kind="retrieval", value="temporal",
        buckets=frozenset({"multi_hop"}), cost_prior=0.5, quality_prior=0.88, local=True,
    ))
    after = _sig(gen.generate(intent, goal))
    assert ("cheap", "temporal") in after and ("premium", "temporal") in after
    assert before < after  # strictly more candidates, everything prior still present


def test_for_intent_orders_by_quality_prior():
    reg = CapabilityRegistry.from_rules()
    tiers = reg.values_for("multi_hop", "model_tier")   # cheap + premium serve multi_hop
    assert tiers == ["premium", "cheap"]                # best quality first
