"""edge-sentinel × ContextOS — SOC triage tenant (tool-using, approval-gated).

Shows the cybersecurity use case end to end, offline: ContextOS plans which sources to
pull for each alert, runs them as ToolPlugins (CrowdSec/threat-intel/EDR — simulated
when no live LAPI), assembles the evidence, recommends an approval-gated block, and
learns the cheapest source bundle that still reaches the right verdict.

    python examples/soc_triage.py
"""
from __future__ import annotations

from contextos.integrations.edge_sentinel import (
    DEFAULT_BUNDLES, SOCTriageTenant, _soc_bandit, reward_triage, soc_bucket,
)

# Alert stream. Hidden truth: each SOC bucket has ONE decisive source — the tenant must
# learn the cheapest bundle that includes it (network→crowdsec, malware→threat-intel,
# host→edr). A real SOC has exactly this structure; the bandit discovers it from outcomes.
ALERTS = [
    ("Is 185.220.101.4 brute-forcing ssh?", "crowdsec"),             # network_anomaly
    ("Unusual port scan traffic from a single IP", "crowdsec"),       # network_anomaly
    ("Is this ransomware on the finance share?", "threat_intel"),     # threat_hunt
    ("Is CVE-2024-3094 relevant to us?", "threat_intel"),            # threat_hunt
    ("Suspicious powershell process spawned by winword", "edr"),      # behavioral
    ("Persistence via a scheduled task on the host", "edr"),          # behavioral
]


def _correct(bundle, latent_source) -> bool:
    """The verdict is correct iff the chosen bundle includes the source that holds the
    decisive evidence for this alert kind."""
    return latent_source in bundle.sources


def run(rounds: int = 72) -> None:
    soc = SOCTriageTenant(bandit=_soc_bandit(0.1),
                          approver=lambda a: False)  # deny blocks by default (no human here)
    learned, baseline_full = [], []

    print("First few triages (sources chosen by the bandit, learning the minimal sufficient set):\n")
    for i in range(rounds):
        q, latent = ALERTS[i % len(ALERTS)]
        r = soc.triage(q)
        correct = _correct(r.bundle, latent)
        reward = soc.record_outcome(q, confirmed_malicious=correct, analyst_correct=correct)
        learned.append(reward)
        # baseline: always pull the full bundle (correct but most expensive)
        full = DEFAULT_BUNDLES[-1]
        baseline_full.append(reward_triage(_correct(full, latent), full))
        if i < 6:
            print(f"  [{r.soc_bucket:<16}] {q[:46]:<46} sources={r.bundle.key:<28} "
                  f"{'✓' if correct else '✗'} action={r.recommended_action}")

    w = 18
    print(f"\nreward = correct-verdict − source-cost (higher = right AND cheaper)\n")
    print(f"  ContextOS (learned bundles): {sum(learned[-w:]) / w:.3f}")
    print(f"  baseline (always full set):  {sum(baseline_full[-w:]) / w:.3f}")
    print("\n── learned source policy per SOC bucket ──")
    for bucket, key in soc.policy().items():
        print(f"  {bucket:<16} → {key}")

    # the approval gate: a block is denied unless an approver allows it
    blocked = soc.act("185.220.101.4")
    print(f"\n── remediation (approval-gated) ──\n  block_ip → ok={blocked.ok}: {blocked.text}")
    print(f"  audit log: {soc.registry.audit[-1]}")


if __name__ == "__main__":
    run()
