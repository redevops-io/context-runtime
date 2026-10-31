"""The one place where ``context_runtime.qa`` imports the neutral ``runtime_contracts`` package.

The QA planning code builds and checks the canonical ``qa/v1`` records (``QAPlan``, ``TestSpec``, the policy
decision), so the dependency is required and the import is hard. There is no fallback and no silent degradation:
when the package is missing the import fails here, with the install hint.
"""
from __future__ import annotations

INSTALL_HINT = (
    "context_runtime.qa needs the 'runtime-contracts' package (the qa/v1 schemas). "
    "Install it with: pip install \"context-runtime[qa] @ git+https://github.com/redevops-io/context-runtime.git\" "
    "or, from a clone, pip install -e \".[qa]\"."
)

try:
    import runtime_contracts.qa as qa
    from runtime_contracts.canonical import canonical_json, content_hash
    from runtime_contracts.models.capability import Idempotency, SideEffect
except ImportError as exc:  # pragma: no cover - exercised by a subprocess test with the package hidden
    raise ImportError(INSTALL_HINT) from exc

__all__ = ["INSTALL_HINT", "qa", "canonical_json", "content_hash", "Idempotency", "SideEffect"]
