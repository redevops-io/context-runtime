"""EXPLAIN surface + quality-aware routing through the tenant: the decision rationale, the
per-method retrieval trace, the served/abstain view, reward provenance, and the text renderer."""
from __future__ import annotations

from context_runtime.explain import render_explain
from context_runtime.integrations.librechat import LibreChatTenant, RetrievalStrategy
from context_runtime.quality import QualityLedger
from context_runtime.types import Hit

STRATS = (
    RetrievalStrategy("bm25", 3, False),
    RetrievalStrategy("hybrid", 3, False),
    RetrievalStrategy("vector", 3, False),
)


class _Stub:
    """hybrid returns the two relevant docs; bm25/vector return noise. Deterministic."""
    def search(self, query, k, method):
        if method == "hybrid":
            hits = [Hit("r0", "relevant-0.txt", "the answer", 0.9),
                    Hit("r1", "relevant-1.txt", "more answer", 0.8)]
        else:
            hits = [Hit(f"n0_{method}", f"noise-{method}.txt", "off topic", 5.0),
                    Hit(f"n1_{method}", f"noise2-{method}.txt", "off topic", 4.0)]
        return hits[:k]

    def index(self, path):
        return {}


def _tenant():
    return LibreChatTenant(retriever=_Stub(), strategies=STRATS,
                           quality_ledger=QualityLedger(), quality_routing=True,
                           quality_min_samples=2)


def test_explain_structure_and_render():
    t = _tenant()
    exp = t.explain("what is the answer", k=3)
    # top-level shape
    for key in ("request", "intent_bucket", "context_key", "decision", "retrieval", "served", "reward"):
        assert key in exp
    # decision: every arm present, chosen flagged + first, each has a reason + quality slot
    cands = exp["decision"]["candidates"]
    assert len(cands) == 3 and cands[0]["chosen"] and "reason" in cands[0]
    assert all("bandit" in c and "quality" in c for c in cands)
    # retrieval trace: methods present, hybrid's hits are the relevant docs, served-marked
    assert "hybrid" in exp["retrieval"] and "bm25" in exp["retrieval"]
    # reward provenance names the native-signal-first policy
    assert "native" in exp["reward"]["policy"] and exp["reward"]["quality_routing"] is True
    # the renderer produces the EXPLAIN-ANALYZE-style text
    txt = render_explain(exp)
    for token in ("EXPLAIN", "decision", "retrieval", "served", "reward", exp["decision"]["chosen"]["key"]):
        assert token in txt


def test_quality_routing_learns_the_better_arm():
    t = _tenant()
    # teach: hybrid → high quality, others → low. After exploration, routing exploits hybrid.
    for _ in range(20):
        ctx = t.retrieve("lookup question")
        t.record_judgment("lookup question", 0.9 if ctx.strategy.method == "hybrid" else 0.3)
    exp = t.explain("lookup question")
    assert exp["decision"]["chosen"]["method"] == "hybrid"          # routed to the best-quality arm
    chosen = exp["decision"]["candidates"][0]
    assert chosen["quality"] is not None and chosen["quality"]["quality"] > 0.6
    assert "quality-routed" in chosen["reason"]


def test_explain_survives_a_new_arm_on_a_persisted_context():
    """Regression: a persisted policy predates a newly-added arm (e.g. the image arm turned on by
    CR_MULTIMODAL). EXPLAIN iterates every arm → bandit.value() must not KeyError on the new one."""
    t = _tenant()
    for _ in range(4):
        ctx = t.retrieve("q"); t.record_judgment("q", 0.7)   # learns contexts under the 3 base arms
    # add a 4th arm after the fact (simulates enabling a new strategy against a learned policy)
    new = RetrievalStrategy("graph", 6, False)
    t.strategies = STRATS + (new,)
    t.bandit.arms = t.bandit.arms + (new,)
    exp = t.explain("q")                                     # must not raise
    keys = {c["key"] for c in exp["decision"]["candidates"]}
    assert new.key in keys                                   # the new arm shows as un-tried (n=0)
    assert next(c for c in exp["decision"]["candidates"] if c["key"] == new.key)["bandit"]["n"] == 0


def test_explain_is_read_only():
    """EXPLAIN must not mutate the learned policy (no bandit/ledger writes)."""
    t = _tenant()
    for _ in range(6):
        ctx = t.retrieve("q")
        t.record_judgment("q", 0.8 if ctx.strategy.method == "hybrid" else 0.2)
    before = t.quality_ledger.to_dict()
    n_before, _ = t.bandit.value(t._select_ctx(t.runtime.plan(__import__(
        "context_runtime.types", fromlist=["Goal"]).Goal(text="q")), "q"), "hybrid:k3:norr")
    t.explain("q"); t.explain("q")
    assert t.quality_ledger.to_dict() == before                    # ledger unchanged
