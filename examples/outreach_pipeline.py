#!/usr/bin/env python3
"""outreach-engine — the end-to-end pipeline you actually run.

    signals (GitHub/HN/manual) → ICP score/rank → outreach tenant picks the play per account
    → draft the EXPLAIN teardown → approval-gated sequence (nothing sends without your sign-off)

Runs OFFLINE on canned demo signals by default (so it's reproducible); pass ``--live`` to pull real
signals from the free GitHub + Hacker News APIs (set GITHUB_TOKEN for a higher rate limit).

    PYTHONPATH=. python examples/outreach_pipeline.py            # offline demo
    PYTHONPATH=. python examples/outreach_pipeline.py --live     # real signals
    PYTHONPATH=. python examples/outreach_pipeline.py --live --contacts  # + resolve who to email
    PYTHONPATH=. python examples/outreach_pipeline.py --live --approve   # actually mark as sent

With ``--contacts`` the pipeline resolves the *person* to reach per account: GitHub maintainers
(free) for repo-backed signals, then the Hunter/Apollo/Clay waterfall (set HUNTER_API_KEY etc.) to
backfill missing emails. Without any key it still shows the resolved names + profiles.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context_runtime.integrations.outreach_contacts import find_contacts
from context_runtime.integrations.outreach_engine import OutreachEngineTenant
from context_runtime.integrations.outreach_icp import rank_accounts, teardown_email
from context_runtime.integrations.signals import (
    AccountSignal, collect_signals, github_signals, hn_hiring_signals, hn_signals, manual_signals,
)

# ── canned signals so the pipeline runs offline / in CI ──
DEMO_SIGNALS = [
    AccountSignal("acme-ai", "tech_pain", "github",
                  "builds RAG in acme-ai/support-rag (1.2k★): hybrid search + reranking",
                  "https://github.com/acme-ai/support-rag", "https://github.com/acme-ai/support-rag",
                  {"stars": 1200, "org": True}),
    AccountSignal("nimbus-labs", "hiring", "hn_hiring",
                  "hiring for ML/RAG: “Nimbus Labs | Senior ML Platform Eng | remote | RAG + LLM infra”",
                  "https://news.ycombinator.com/item?id=1"),
    AccountSignal("Vector Financial", "funding", "manual",
                  "raised a $14M Series A; building an internal RAG assistant over filings",
                  "https://example.com/vf-series-a"),
    AccountSignal("hn-user-42", "tech_pain", "hn",
                  "discussing RAG pain: “our retrieval recall is fine but precision tanks after reranking”",
                  "https://news.ycombinator.com/item?id=2"),
]


def gather(live: bool) -> list[AccountSignal]:
    if not live:
        return DEMO_SIGNALS
    print("… pulling live signals from GitHub + Hacker News (free APIs)\n")
    return collect_signals(
        github_signals(limit=8),
        hn_signals(limit=6),
        hn_hiring_signals(limit=6),
        manual_signals([]),   # add your funding/leadership rows here
    )


def _warm(tenant: OutreachEngineTenant) -> None:
    """Train the tenant on past outcomes so choose() reflects a LEARNED policy (spend the artifact
    teardown on high-signal accounts). Uses the same simulated outcome model as examples/outreach_engine."""
    from examples.outreach_engine import BUCKETS, _rng, pilot_value
    rng = _rng(1)
    for i in range(400):
        bucket = BUCKETS[int(rng() * len(BUCKETS)) % len(BUCKETS)]
        acct = f"warm {bucket} {i}"
        play = tenant.choose(acct, bucket=bucket)
        tenant.record_outcome(acct, pilot_value(play, bucket, rng()))


def _contacts_line(s: AccountSignal) -> str:
    """Resolve who to email at this account (free GitHub maintainers, then the waterfall)."""
    contacts = find_contacts(s, limit=3)
    if not contacts:
        return "    contact: — (no repo maintainers / set HUNTER_API_KEY for the waterfall)"
    parts = []
    for c in contacts[:3]:
        who = c.name + (f" <{c.email}>" if c.email else " (no public email)")
        parts.append(f"{who} [{c.source}·{c.confidence:.2f}]")
    return "    contact: " + "; ".join(parts)


def main() -> int:
    live = "--live" in sys.argv
    approve = "--approve" in sys.argv
    contacts = "--contacts" in sys.argv
    tenant = OutreachEngineTenant(approver=(lambda action: approve))
    _warm(tenant)   # so the displayed plays reflect what the runtime has learned

    signals = gather(live)
    ranked = rank_accounts(signals, top=10)
    print("=" * 74)
    print(f"OUTREACH PIPELINE — {len(signals)} signals → top {len(ranked)} accounts by ICP score")
    print("=" * 74)
    for s, score in ranked:
        play = tenant.choose(s.as_query(), bucket=s.signal)            # runtime picks the play
        msg = teardown_email(s)                                        # the proof-of-effort opener
        send = tenant.send_sequence(s.as_query())                     # approval-gated
        gate = "✓ SENT" if send["status"] == "sent" else "⏸ awaiting your approval"
        print(f"\n▶ {s.account:<20} score {score:<5} · signal={s.signal} · play={play.key}")
        print(f"    source: {s.source} · artifact: {s.artifact or '—'}")
        if contacts:
            print(_contacts_line(s))
        print(f"    {gate}   subject: {msg['subject']}")
        print("    " + msg["body"].strip().replace("\n", "\n    ")[:520])
    print("\n" + "-" * 74)
    print("Nothing was sent" + ("" if approve else " (no --approve): every sequence is human-gated.")
          + "  Wire outreach-engine's send_sequence to Twenty CRM / your sender to go live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
