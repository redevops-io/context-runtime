"""edge-sentinel SOC tenant: tools assemble evidence, bandit learns cheap sufficient bundles, blocks are gated."""
from __future__ import annotations

import inspect

import pytest

from context_runtime.integrations.edge_sentinel import (
    DEFAULT_BUNDLES, SOCTriageTenant, _soc_bandit, reward_triage, soc_bucket,
)


def test_soc_bucketing():
    assert soc_bucket("Is this ransomware on the endpoint?") == "threat_hunt"
    assert soc_bucket("Is 1.2.3.4 brute-forcing ssh?") == "network_anomaly"
    assert soc_bucket("suspicious powershell process persistence") == "behavioral"


def test_triage_pulls_tool_evidence():
    soc = SOCTriageTenant()
    r = soc.triage("Is 45.155.205.99 a known malicious scanner?")
    assert r.hits                       # tools returned evidence
    assert r.bundle in DEFAULT_BUNDLES
    assert r.context


def test_reward_prefers_cheapest_correct_bundle():
    cheap = DEFAULT_BUNDLES[1]          # one source
    full = DEFAULT_BUNDLES[-1]          # three sources
    assert reward_triage(True, cheap) > reward_triage(True, full)
    assert reward_triage(False, cheap) == 0.0


def test_block_ip_denied_without_approver():
    soc = SOCTriageTenant(approver=lambda a: False)
    res = soc.act("185.220.101.4")
    assert not res.ok and "denied" in res.error
    assert soc.registry.audit[-1]["allowed"] is False


def test_block_ip_allowed_with_approver_but_dry_run():
    soc = SOCTriageTenant(approver=lambda a: True)
    res = soc.act("185.220.101.4")
    assert res.ok and "dry-run" in res.text    # safe even when approved (CROWDSEC_LIVE unset)


def test_tenant_learns_cheap_sufficient_bundle():
    soc = SOCTriageTenant(bandit=_soc_bandit(0.1))
    latent = "threat_intel"
    q = "Is 45.155.205.99 a known malicious scanner?"   # network_anomaly bucket, ti is decisive
    for _ in range(60):
        r = soc.triage(q)
        correct = latent in r.bundle.sources
        soc.record_outcome(q, confirmed_malicious=correct, analyst_correct=correct)
    # the learned policy for this bucket must include the decisive source
    chosen = soc.policy()[soc_bucket(q)]
    assert latent in chosen


@pytest.mark.skip(reason="Retrieval-level RBAC/data-source scoping is enforced by the ENTERPRISE "
                         "policy layer (context-runtime-v3: PolicyEngine.feasible / allowed_data_sources "
                         "/ rows_owned_by_requester), NOT the OSS retrieval core. This placeholder "
                         "documents that boundary so the 'security' suite name does not imply coverage "
                         "that lives in a different repo.")
def test_retrieval_rbac_scope_is_enterprise_layer():
    # If per-principal data scoping is ever added to the OSS core, move this from skip to a real test.
    pass


def test_soc_handle_has_no_principal_scoping_param():
    # Corollary that DOES run: the OSS SOC tenant gates side-effects (blocks) via approval, but carries
    # no per-principal data-scope parameter — confirming the RBAC gap is a real, documented boundary.
    params = set(inspect.signature(SOCTriageTenant.__init__).parameters)
    assert not ({"principal", "scope", "acl", "row_owner"} & params)
