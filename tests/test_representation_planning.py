"""v4 knowledge-aware planning — intent → representation → constrained candidates → learning.

Proves the wiring the whitepaper-v4 claims: the decision engine classifies a knowledge
representation, candidate generation is constrained to that representation's methods (with a
document fallback + confidence-gated exploration), and the choice is recorded on the plan/event.
"""
from __future__ import annotations

from context_runtime.planner import representations
from context_runtime.planner.candidates import RuleCandidateGenerator
from context_runtime.planner.intent import RuleIntentAnalyzer
from context_runtime.learning.events import OutcomeEvent
from context_runtime.types import Goal, Intent, Candidate, StepSpec, PlanScore, Plan


def _methods(cands):
    out = []
    for c in cands:
        for s in c.steps:
            if s.type == "retrieve":
                out.append(s.params["method"])
    return set(out)


def _plan_for(text: str):
    intent = RuleIntentAnalyzer().analyze(Goal(text=text))
    cands = RuleCandidateGenerator().generate(intent, Goal(text=text))
    return intent, cands


# ─── classify head ───────────────────────────────────────────────────────────
def test_classify_maps_intent_to_representation():
    assert representations.classify("multi_hop", "how does X relate to Y") == "graph"
    assert representations.classify("temporal", "what was the plan as of March") == "temporal"
    assert representations.classify("code_reasoning", "refactor this function") == "code"
    assert representations.classify("conceptual", "what is idempotency") == "document"
    # HINTS reach representations no bucket produces on its own:
    assert representations.classify("incident", "how many incidents last week") == "analytical"
    assert representations.classify("conceptual", "what's in this screenshot") == "multimodal"


def test_analyzer_sets_representation():
    assert RuleIntentAnalyzer().analyze(Goal(text="how does A depend on B")).representation == "graph"
    assert RuleIntentAnalyzer().analyze(Goal(text="explain caching")).representation == "document"


# ─── candidates constrained to the representation ────────────────────────────
def test_graph_query_constrains_to_graph_plus_document_fallback():
    intent, cands = _plan_for("what is the dependency chain between service A and service B")
    assert intent.representation == "graph"
    methods = _methods(cands)
    assert "graph" in methods and "hybrid" in methods          # representation-first + fallback
    assert "vector" not in methods and "bm25" not in methods    # not the flat document table


def test_analytical_hint_generates_olap_candidates():
    intent, cands = _plan_for("how many deploys happened per week over the last month")
    assert intent.representation == "analytical"
    methods = _methods(cands)
    assert methods & {"sql", "logs", "api"}                     # OLAP candidates now reachable
    assert "hybrid" in methods                                  # with a document fallback


def test_plain_document_query_unchanged():
    intent, cands = _plan_for("what is eventual consistency")   # conceptual, no graph/temporal cue
    assert intent.representation == "document"
    assert _methods(cands) <= {"vector", "bm25", "hybrid", "file"}


def test_bandit_arms_span_representations_for_learning():
    # multi_hop routes to graph but keeps the document fallback -> the bandit's arms span TWO
    # representations, so choosing among them IS representation selection (what it learns online).
    _, cands = _plan_for("trace how the outage propagated across the payment and auth services")
    reps = {representations.representation_for(m) for m in _methods(cands)}
    assert {"graph", "document"} <= reps


# ─── representation recorded on the outcome event ────────────────────────────
def test_representation_recorded_on_outcome_event():
    intent = Intent(bucket="multi_hop", representation="graph")
    plan = Plan(intent=intent,
                chosen=Candidate(steps=(StepSpec("retrieve", {"method": "graph"}),)),
                score=PlanScore(),
                extra={"bandit": {"context": "multi_hop", "arm": "graph:cheap"}})
    ev = OutcomeEvent.from_plan(plan, reward=1.0)
    assert ev.representation == "graph" and ev.context == "multi_hop"
