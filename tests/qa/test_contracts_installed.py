"""The QA planning dependency is required. This module imports it directly (no importorskip) so a missing install
is a failure with the install hint, never a skip."""
from __future__ import annotations

import runtime_contracts.qa as contracts  # hard import

from context_runtime.qa import _contracts


def test_qa_contracts_installed():
    hint = _contracts.INSTALL_HINT
    assert "pip install" in hint and "[qa]" in hint, hint
    assert contracts.QA_CONTRACT_ID == "qa/v1", f"unexpected contracts family; {hint}"
    for name in ("QAPlan", "PlanItem", "PlanGate", "ResourceClaim", "TestSpec", "QAPolicyDecision"):
        assert hasattr(contracts, name), f"runtime_contracts.qa has no {name}; {hint}"
