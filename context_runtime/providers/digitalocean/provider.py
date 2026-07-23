"""DoProvider — DigitalOcean as a concrete CloudProvider.

Ties the Gradient adapters (serverless inference + knowledge-base retrieve) to the neutral
``CloudProvider`` seam. DO is the lean platform, so several factories honestly return ``None``:
- no serverless SQL warehouse → ``analytical_backend()`` is None (use the in-tree DuckDB or a managed
  Postgres via pgvector instead);
- guardrails are applied *inside* an agent, not as a standalone check → ``guardrail()`` is None;
- no per-agent identity broker → ``identity_broker()`` is None.

Config (all optional; pass to ``get_provider("digitalocean", ...)``):
    api_token, inference_key           - auth (session)
    model_tiers                        - [(tier, do_model_id, cost_per_1k)]
    knowledge_base_id                  - the KB retrieve endpoint
"""
from __future__ import annotations

from ..base import CloudProvider, ModelPlugin, RetrieverPlugin
from .session import DoSession


class DoProvider(CloudProvider):
    name = "digitalocean"

    def __init__(self, *, api_token=None, inference_key=None, model_tiers=None,
                 knowledge_base_id=None, session=None):
        self.session = session or DoSession(api_token=api_token, inference_key=inference_key)
        self.model_tiers = model_tiers
        self.knowledge_base_id = knowledge_base_id

    def model(self) -> ModelPlugin | None:
        from .gradient_model import GradientModel
        return GradientModel.from_config(self.session, tiers=self.model_tiers)

    def document_retriever(self) -> RetrieverPlugin | None:
        return self.managed_kb_retriever()

    def managed_kb_retriever(self) -> RetrieverPlugin | None:
        if not self.knowledge_base_id:
            return None
        from .gradient_kb_retriever import GradientKBRetriever
        return GradientKBRetriever(self.session, knowledge_base_id=self.knowledge_base_id)

    # analytical_backend / guardrail / identity_broker / telemetry_reader inherit the base None:
    # DO ships none of those as a separable capability (see module docstring).
