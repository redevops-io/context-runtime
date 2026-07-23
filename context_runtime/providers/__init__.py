"""Cloud-provider adapters — managed clouds behind the runtime's plugin Protocols.

See ``base.py`` for the seam. AWS is the first provider (``providers.aws``); Azure/GCP/DigitalOcean
follow the same shape: a subpackage implementing the pieces it offers + one ``register_provider``.
"""
from __future__ import annotations

from .base import (
    CloudProvider,
    Guardrail,
    GuardrailVerdict,
    IdentityBroker,
    TelemetryReader,
    WarehouseBackend,
    available_providers,
    get_provider,
    register_provider,
)

__all__ = [
    "CloudProvider",
    "Guardrail",
    "GuardrailVerdict",
    "IdentityBroker",
    "TelemetryReader",
    "WarehouseBackend",
    "available_providers",
    "get_provider",
    "register_provider",
]
