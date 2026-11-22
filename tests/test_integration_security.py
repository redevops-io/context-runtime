"""edge-sentinel SOC tenant: tools assemble evidence, bandit learns cheap sufficient bundles, blocks are gated.

``local_run`` on the tests that expect a tool to run: a registry with no authorizer denies every call, so an
evidence pull or an approved block needs a stated authority — and the test below pins that a tenant built
without one fails at construction rather than serving denials.
"""
from __future__ import annotations

import inspect

import pytest

from context_runtime.integrations.edge_sentinel import (
    DEFAULT_BUNDLES, SOCTriageTenant, _soc_bandit, reward_triage, soc_bucket,
)
from context_runtime.tools.authz import RegistryConfigurationError


def test_soc_bucketing():
    assert soc_bucket("Is this ransomware on the endpoint?") == "threat_hunt"
    assert soc_bucket("Is 1.2.3.4 brute-forcing ssh?") == "network_anomaly"
    assert soc_bucket("suspicious powershell process persistence") == "behavioral"


def test_triage_pulls_tool_evidence(local_run):
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


def test_soc_registry_classifies_its_tools(local_run):
    """The risk class is declared where each tool is registered, and it is the shared vocabulary.

    ``block_ip`` reaches a firewall: it is ``write`` whatever the command looks like. The evidence tools read,
    so a deployment can grant a triage-only principal ``read`` and hold the block back.

    ``local_run`` is here because the tenant refuses to be constructed without an authorizer; nothing in this
    test depends on what the authorizer decides.
    """
    soc = SOCTriageTenant()
    classification = soc.registry.classification()
    assert classification["block_ip"] == "write"
    assert set(classification) == {"block_ip", "crowdsec_decisions", "edr_timeline", "threat_intel"}
    assert (classification["threat_intel"] == classification["crowdsec_decisions"]
            == classification["edr_timeline"] == "read")
    assert set(classification.values()) <= {"read", "write", "delete"}
    assert soc.registry.required_capability("block_ip") == {
        "resource_kind": "tool", "resource_name": "block_ip", "requested_action": "write"}
    assert soc.registry.validate(strict=True).ok


def test_soc_tenant_without_an_authorizer_refuses_to_start(monkeypatch):
    """No plane, no local declaration ⇒ construction fails. Local trust is never implied by the absence of one."""
    monkeypatch.delenv("CR_LOCAL_AUTHORITY", raising=False)
    with pytest.raises(RegistryConfigurationError) as exc:
        SOCTriageTenant()
    message = str(exc.value)
    for door in ("authorizer=", "set_default_authorizer", "CR_LOCAL_AUTHORITY"):
        assert door in message                         # the error names all three ways to answer it


def test_soc_tenant_declares_local_execution_out_loud(monkeypatch):
    """``local_trust=True`` is only a decision when it comes with a written reason — a bare switch is refused."""
    monkeypatch.delenv("CR_LOCAL_AUTHORITY", raising=False)
    with pytest.raises(RegistryConfigurationError):
        SOCTriageTenant(local_trust=True)
    soc = SOCTriageTenant(local_trust=True, local_trust_reason="single-operator laptop, simulated telemetry")
    assert soc.act("185.220.101.4").ok is False         # still approval-gated: local trust is not bypass
    assert any(a["plane"] == "local_trusted" for a in soc.registry.audit)


def test_block_ip_denied_without_approver(local_run):
    soc = SOCTriageTenant(approver=lambda a: False)
    res = soc.act("185.220.101.4")
    assert not res.ok and "denied" in res.error
    assert soc.registry.audit[-1]["allowed"] is False


def test_block_ip_allowed_with_approver_but_dry_run(local_run):
    soc = SOCTriageTenant(approver=lambda a: True)
    res = soc.act("185.220.101.4")
    assert res.ok and "dry-run" in res.text    # safe even when approved (CROWDSEC_LIVE unset)


def test_tenant_learns_cheap_sufficient_bundle(local_run):
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
