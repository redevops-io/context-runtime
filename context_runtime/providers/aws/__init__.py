"""AWS provider package — the first concrete CloudProvider (Bedrock, OpenSearch, Athena, Guardrails).

``register()`` wires the factory into the provider registry; it's called lazily by
``providers.base.get_provider("aws", ...)`` so importing the runtime never imports boto3.
"""
from __future__ import annotations

from ..base import register_provider


def register() -> None:
    from .provider import AwsProvider
    register_provider("aws", AwsProvider)
