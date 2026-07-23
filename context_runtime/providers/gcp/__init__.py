"""GCP provider package — Gemini, Vertex AI Search, BigQuery, Model Armor, Cloud Monitoring.

``register()`` wires the factory into the provider registry; it's called lazily by
``providers.base.get_provider("gcp", ...)`` so importing the runtime never imports a google library.
"""
from __future__ import annotations

from ..base import register_provider


def register() -> None:
    from .provider import GcpProvider
    register_provider("gcp", GcpProvider)
