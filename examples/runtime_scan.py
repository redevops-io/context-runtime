"""runtime-scan × Context Runtime — deployment-DAST tenant, offline end-to-end.

Shows edge-sentinel's *runtime* security plane (complement of the supply_chain/Trivy plane):
Context Runtime plans which DAST bundle to run against a running deployment, runs it via the
deploy_scan tools (nmap/nuclei/ZAP — simulated when the binaries aren't installed), and learns
the cheapest scan bundle that still catches the exposure. Passive scans auto-approve; the ZAP
*active* scan is approval-gated. Every target is scope-checked against SCAN_ALLOWLIST first.

    python examples/runtime_scan.py
"""
from __future__ import annotations

import os
import time

# Scope: only these (owned) hosts may be scanned. Fail-closed — unset ⇒ nothing scans.
os.environ.setdefault(
    "SCAN_ALLOWLIST",
    "192.168.40.105,vibexgen.io,*.vibexgen.io,telegrambot.ai,*.telegrambot.ai,learnerbot.ai,*.learnerbot.ai",
)

from context_runtime.integrations.runtime_scan import (  # noqa: E402
    DEFAULT_BUNDLES, RuntimeScanTenant, ScanBundle, _scan_bandit, mount_deploy_scan,
    passive_only_approver, reward_scan, scan_bucket,
)
from context_runtime.tools.base import ApprovalPolicy, ToolRegistry  # noqa: E402

# A stream of OWNED deployment targets. Hidden truth: each target class has ONE decisive tool
# (tls→nmap, web_cve→nuclei, injection→zap_active). The tenant must learn the cheapest bundle
# that includes it — e.g. escalate to the (gated) active scan ONLY for injection-shaped targets.
TARGETS = [
    "192.168.40.105",                         # tls        (bare host)
    "learnerbot.ai",                          # tls
    "https://vibexgen.io/",                   # web_cve
    "https://telegrambot.ai/",                # web_cve
    "https://app.telegrambot.ai/login",       # injection  (needs active)
    "https://auth.vibexgen.io/api",           # injection
]

BASELINE = ScanBundle(("nmap", "nuclei", "zap_baseline"))   # "thorough passive" — misses injection


def _caught(result) -> bool:
    """Grounded in the actual scan output: a real exposure = a high/critical finding."""
    return any(f.severity in ("high", "critical") for f in result.report.findings)


def _bundle_cost(b: ScanBundle) -> float:
    from context_runtime.integrations.runtime_scan import _bundle_cost as bc
    return bc(b)


def main() -> None:
    tenant = RuntimeScanTenant(bandit=_scan_bandit(epsilon=0.2))
    print("scanner binaries available:", tenant.scanner.available(), "(sim fallback otherwise)\n")

    print("── one-shot: classify + scan each target ──")
    for t in TARGETS:
        r = tenant.scan(t)
        print(f"  {t:42s} class={r.bucket:9s} bundle={r.bundle.key:24s} "
              f"max_sev={r.max_severity:8s} findings={len(r.report.findings)}")
        tenant.record_outcome(t, _caught(r))
    print()

    # ── train the bandit, then compare learned policy vs the thorough-passive baseline ──
    print("── learning: cheapest bundle that still catches the exposure ──")
    tenant = RuntimeScanTenant(bandit=_scan_bandit(epsilon=0.2))
    for _ in range(60):
        for t in TARGETS:
            r = tenant.scan(t)
            tenant.record_outcome(t, _caught(r))

    def evaluate(select_bundle):
        caught = cost = 0.0
        for t in TARGETS:
            bucket = scan_bucket(t)
            bundle = select_bundle(bucket)
            # sim truth: exposure caught iff bundle holds the class's decisive tool
            decisive = {"tls": "nmap", "web_cve": "nuclei", "injection": "zap_active"}[bucket]
            ok = decisive in bundle.tools
            caught += 1.0 if ok else 0.0
            cost += _bundle_cost(bundle)
        return caught / len(TARGETS), cost / len(TARGETS)

    learned_policy = tenant.policy()
    learned_catch, learned_cost = evaluate(lambda b: _bundle_by_key(learned_policy[b]))
    base_catch, base_cost = evaluate(lambda b: BASELINE)

    print("  learned policy (per target class):")
    for bucket, key in sorted(learned_policy.items()):
        print(f"    {bucket:9s} → {key}")
    print(f"\n  catch-rate   learned={learned_catch:.2f}   baseline(thorough-passive)={base_catch:.2f}")
    print(f"  avg cost     learned={learned_cost:.2f}    baseline={base_cost:.2f}")
    print(f"  → CR escalates to the gated ACTIVE scan only where a passive bundle can't see the bug,\n"
          f"    and stays cheap elsewhere (delta catch +{learned_catch - base_catch:.2f}).\n")

    _mcp_demo()


def _bundle_by_key(key: str) -> ScanBundle:
    for b in DEFAULT_BUNDLES:
        if b.key == key:
            return b
    return BASELINE


def _mcp_demo() -> None:
    """Prove the deploy_scan MCP server + approval gate end-to-end (async start/poll/results)."""
    print("── deploy_scan MCP server: async scan + approval gate ──")
    reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=passive_only_approver))
    client, names = mount_deploy_scan(reg)
    try:
        print("  mounted MCP tools:", names)

        # passive scan → auto-approved by passive_only_approver
        start = reg.run("scan.scan_start", {"target": "https://vibexgen.io/", "profile": "web-baseline"})
        job = (start.data or {}).get("job_id")
        print(f"  scan_start(web-baseline) → {start.text[:80]}")
        for _ in range(20):
            st = reg.run("scan.scan_status", {"job_id": job})
            if (st.data or {}).get("state") in ("done", "error"):
                break
            time.sleep(0.3)
        res = reg.run("scan.scan_results", {"job_id": job})
        print(f"  scan_results → summary={ (res.data or {}).get('summary') }")

        # active scan without a human approver → BLOCKED by the ApprovalPolicy
        blocked = reg.run("scan.scan_start", {"target": "https://app.telegrambot.ai/login", "profile": "web-active"})
        print(f"  scan_start(web-active, no approver) → {'BLOCKED: ' + (blocked.error or '') if not blocked.ok else blocked.text}")

        # out-of-scope target → refused before any scanner runs
        oos = reg.run("scan.scan_start", {"target": "https://example.com/", "profile": "recon"})
        print(f"  scan_start(example.com, off-scope) → isError={not oos.ok or 'out of scope' in (oos.text or '')}: {oos.text[:70]}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
