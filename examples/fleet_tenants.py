"""The migrated fleet — every business module learning its cheapest source policy.

Each agentic module (plus net-new tenants from the use-cases doc) is now a Context Runtime
tenant with a goal and a metric. This runs the whole catalog offline: for each module
a hidden decisive source exists per question kind, and every tenant learns the cheapest
bundle that meets its goal — proving the old hand-wired fleet collapses into one
data-driven pattern.

    python examples/fleet_tenants.py
"""
from __future__ import annotations

from context_runtime import ContextRuntime
from context_runtime.integrations.modules import CATALOG, ModuleTenant, question_kind, reward

# one representative question per module + its hidden decisive source
PROBES = {
    "billing": ("why is the ledger out of balance?", "ledger"),
    "support": ("why can't the customer log in?", "tickets"),
    "control_tower": ("why did revenue fall last quarter?", "warehouse"),
    "compliance": ("which controls are failing?", "scan_results"),
    "books": ("why won't the month-end close?", "ledger"),
    "crm": ("which deals are most likely to close?", "deals"),
    "market_radar": ("did a competitor change pricing?", "pricing"),
    "growth_engine": ("which channel drove signups?", "attribution"),
    "outreach": ("which prospects should we reach out to first?", "leads"),
    "social": ("what content is trending for us?", "trends"),
    "lifecycle": ("why did deliverability drop?", "segments"),
    "privacy": ("can we fulfill this DSAR in time?", "requests"),
    "edge_sentinel": ("is this IP malicious?", "threat_intel"),
    "guide": ("how do I set up permissions for my app?", "docs"),
    "growth_assistant": ("which channel should a first-time founder try first?", "playbooks"),
    "incident": ("why did the deploy fail?", "logs"),
    "research": ("how does mitochondrial dysfunction relate to Parkinson?", "citations"),
    "finance": ("should we invest given the latest filing?", "filings"),
    "personal": ("what should I prioritize today?", "tasks"),
}


def run(rounds: int = 60) -> None:
    rt = ContextRuntime.default([])
    rng = [0x51EE]
    print(f"{'module':<15}{'core':<18}{'learned (tuned)':>16}{'always-full':>14}  policy")
    print("-" * 92)
    for name, spec in CATALOG.items():
        tenant = ModuleTenant(spec, runtime=rt, epsilon=0.1)
        q, latent = PROBES[name]
        tuned, full = [], []
        for _ in range(rounds):
            r = tenant.handle(q)
            success = latent in r.bundle.sources
            tuned.append(tenant.record_outcome(q, success))
            from context_runtime.integrations.modules import SourceBundle
            fb = SourceBundle(tuple(spec.sources))
            full.append(reward(latent in fb.sources, fb, len(spec.sources)))
        kind = question_kind(q)
        learned = tenant.policy().get(f"{name}:{kind}", "?")
        print(f"{name:<15}{spec.core:<18}{sum(tuned[-12:])/12:>16.3f}{sum(full[-12:])/12:>14.3f}  {learned}")

    print("\nEvery module: one goal, one metric, a learned cheapest-sufficient source policy.")
    print("16 hand-wired fleet controllers → one data-driven Context Runtime tenant pattern.")


if __name__ == "__main__":
    run()
