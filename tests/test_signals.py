"""Signal connectors + ICP scoring/templates — offline (injected fetchers, no network)."""
from __future__ import annotations

from context_runtime.integrations.outreach_icp import rank_accounts, score_signal, teardown_email
from context_runtime.integrations.signals import (
    AccountSignal, collect_signals, github_signals, hn_hiring_signals, hn_signals, manual_signals,
)


def test_github_signals_normalize_orgs_and_artifact():
    fake = {"items": [
        {"full_name": "acme/rag", "html_url": "https://github.com/acme/rag",
         "owner": {"login": "acme", "type": "Organization"}, "stargazers_count": 900,
         "description": "hybrid RAG"},
    ]}
    sig = github_signals(fetch=lambda url, headers=None: fake)
    assert len(sig) == 1
    s = sig[0]
    assert s.account == "acme" and s.signal == "tech_pain" and s.source == "github"
    assert s.artifact == "https://github.com/acme/rag" and s.meta["stars"] == 900


def test_hn_connectors_filter_and_shape():
    hits = {"hits": [
        {"author": "bob", "objectID": "10", "comment_text": "<p>our RAG <b>reranking</b> hurts precision</p>"},
    ]}
    pain = hn_signals(fetch=lambda url, headers=None: hits)
    assert pain and pain[0].signal == "tech_pain" and "reranking" in pain[0].evidence
    # hiring filter keeps ML/RAG posts, drops the rest
    hiring_hits = {"hits": [
        {"author": "x", "objectID": "11", "comment_text": "Nimbus | ML Platform Eng | remote | RAG infra"},
        {"author": "y", "objectID": "12", "comment_text": "Bakery | Cashier | onsite"},
    ]}
    hiring = hn_hiring_signals(fetch=lambda url, headers=None: hiring_hits)
    assert len(hiring) == 1 and hiring[0].signal == "hiring" and "Nimbus" in hiring[0].account


def test_connector_failure_degrades_to_empty():
    def boom(url, headers=None):
        raise RuntimeError("network down")
    assert github_signals(fetch=boom) == [] and hn_signals(fetch=boom) == []


def test_collect_dedups_and_prefers_artifact():
    a = AccountSignal("acme", "tech_pain", "hn", "chatter", artifact=None)
    b = AccountSignal("Acme", "tech_pain", "github", "repo", artifact="https://github.com/acme/rag")
    merged = collect_signals([a], [b])
    assert len(merged) == 1 and merged[0].artifact == "https://github.com/acme/rag"


def test_icp_scoring_and_ranking():
    strong = AccountSignal("acme", "tech_pain", "github", "RAG", artifact="https://x", meta={"stars": 3000})
    weak = AccountSignal("who", "cold", "hn", "meh")
    assert score_signal(strong) > score_signal(weak)
    ranked = rank_accounts([weak, strong])
    assert ranked[0][0].account == "acme"          # the strong signal ranks first


def test_teardown_email_leads_with_signal_and_artifact():
    s = AccountSignal("acme", "tech_pain", "github", "RAG", artifact="https://github.com/acme/rag")
    msg = teardown_email(s)
    assert "acme" in msg["subject"] and "EXPLAIN" in msg["body"]
    assert "github.com/acme/rag" in msg["body"] and msg["cta"] == "pilot scoping call"
