"""outreach-engine tenant — play selection, approval-gated send, learning, and fleet registration."""
from __future__ import annotations

from context_runtime.integrations.outreach_engine import (
    DEFAULT_PLAYS, OutreachEngineTenant, OutreachPlay, outreach_bucket,
)


def test_bucketing_by_strongest_signal():
    assert outreach_bucket("they just raised a Series B") == "funded"
    assert outreach_bucket("hiring an ML platform engineer") == "hiring"
    assert outreach_bucket("their RAG has reranking + latency pain") == "tech_pain"
    assert outreach_bucket("new Head of AI joins as of March") == "leadership"
    assert outreach_bucket("random company") == "cold"


def test_choose_drafts_opener_and_send_is_approval_gated():
    t = OutreachEngineTenant()
    play = t.choose("their RAG has reranking pain", bucket="tech_pain")
    assert isinstance(play, OutreachPlay)
    # default-deny approver → the send pauses for human sign-off (never auto-sends)
    pending = t.send_sequence("their RAG has reranking pain")
    assert pending["status"] == "pending_approval" and pending["action"] == "send_sequence"
    assert "preview" in pending
    # with approval granted, it sends
    t2 = OutreachEngineTenant(approver=lambda action: True)
    t2.choose("their RAG has reranking pain", bucket="tech_pain")
    assert t2.send_sequence("their RAG has reranking pain")["status"] == "sent"


def test_artifact_play_produces_a_teardown_opener():
    t = OutreachEngineTenant()
    # force an artifact play and check the opener is the EXPLAIN teardown (the wedge)
    t.bandit = type(t.bandit)((OutreachPlay("tech_pain", "email", "artifact"),))
    t.choose("acme RAG pain", bucket="tech_pain")
    opener = t._pending[t._key("acme RAG pain")][3]
    assert "EXPLAIN" in opener and "/planner" in opener


def test_learns_to_invest_effort_on_high_signal_accounts():
    t = OutreachEngineTenant(epsilon=0.1)
    # teach: artifact converts on tech_pain (value high); template on tech_pain under-converts.
    def value(play):
        depth = {"template": 1.0, "company": 2.6, "artifact": 4.2}[play.depth]
        match = 1.0 if play.signal == "tech_pain" else 0.55
        return depth * match * 1.2
    for i in range(200):
        p = t.choose(f"tech_pain account {i}", bucket="tech_pain")
        t.record_outcome(f"tech_pain account {i}", value(p))
    best = t.policy().get("tech_pain", "")
    # the learned best play for tech-pain accounts is an artifact (teardown) play, not a template
    assert "artifact" in best


def test_fleet_registration_resolves_the_outreach_spec():
    from context_runtime.control_plane.fleet import spec_for
    from context_runtime.control_plane.registry import Module
    m = Module(name="outreach-engine", repo="redevops-io/outreach-engine",
               pain="pilot outreach & pipeline", agents=("sourcer", "personalizer", "sequencer"),
               approval_required=("send_sequence",))
    spec = spec_for(m)
    assert spec.core == "Twenty CRM" and "send_sequence" in spec.actions
