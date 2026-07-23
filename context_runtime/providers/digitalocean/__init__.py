"""DigitalOcean provider package — Gradient serverless inference + knowledge-base retrieve.

``register()`` wires the factory into the provider registry; called lazily by
``providers.base.get_provider("digitalocean", ...)``. No SDK: both surfaces are plain HTTPS.
"""
from __future__ import annotations

from ..base import register_provider


def register() -> None:
    from .provider import DoProvider
    register_provider("digitalocean", DoProvider)
