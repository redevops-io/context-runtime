"""Incident-review example — the §10 vertical slice, end to end, offline.

    python examples/incident_review.py

Shows run() vs explain() vs simulate() over a tiny in-memory corpus, with zero
external deps (stub model + in-memory store). Swap the runtime for
``ContextRuntime.from_config("contextos.yaml")`` with model=litellm, store=redevops_rag
to run the *same plan* against real models and a real index.
"""
from __future__ import annotations

from contextos import ContextRuntime, SourceRef

CORPUS = [
    {"chunk_id": "deploy-x.md::0", "filename": "deploy-x.md",
     "text": "Deployment X failed: the readiness probe timed out because the Cloudflare "
             "origin certificate had expired. cert-manager had not rotated it. Rollback to "
             "the previous release restored service within 4 minutes.", "created_at": None},
    {"chunk_id": "runbook.md::0", "filename": "runbook.md",
     "text": "On-call runbook: for a failed deploy, check pod status, then ingress, then "
             "certificate expiry in the cert-manager logs.", "created_at": None},
    {"chunk_id": "arch.md::0", "filename": "arch.md",
     "text": "Production APIs must run behind Cloudflare. Origin certificates rotate every "
             "90 days via cert-manager; a missed rotation breaks TLS at the edge.", "created_at": None},
]

GOAL = "Explain why deployment X failed"


def main() -> None:
    rt = ContextRuntime.default(CORPUS)
    sources = [SourceRef("docs", "docs"), SourceRef("git", "code"), SourceRef("grafana", "metrics")]
    cons = {"max_cost_usd": 2.0, "max_latency_seconds": 90, "require_citations": True}

    print("# SIMULATE (forecast, no execution)")
    sim = rt.simulate(GOAL, sources=sources, constraints=cons)
    print(f"  expected cost ${sim.expected_cost_usd.point} "
          f"[{sim.expected_cost_usd.low}, {sim.expected_cost_usd.high}] "
          f"· models {sim.expected_models} · retrieval {sim.expected_retrieval} "
          f"· based on {sim.based_on_samples} samples\n")

    print("# EXPLAIN (the chosen plan)")
    ex = rt.explain(GOAL, sources=sources, constraints=cons)
    print(f"  intent={ex.intent.bucket} risk={ex.intent.risk} "
          f"candidates={len(ex.candidates)}")
    print(f"  chosen tier={ex.chosen.chosen.model_tier} "
          f"steps={[s.type for s in ex.chosen.chosen.steps]}")
    print(f"  score={ex.chosen.score.total} feasible={ex.chosen.score.feasible}\n")

    print("# RUN (plan → build_context → execute → verify)")
    res = rt.run(GOAL, sources=sources, constraints=cons)
    print(f"  answer: {res.answer}")
    print(f"  cost=${res.cost_usd:.5f} citations={len(res.citations)} "
          f"verified={res.verdict.passed if res.verdict else None}")
    print(f"  trace spans: {[s.kind for s in res.trace.spans]}")


if __name__ == "__main__":
    main()
