"""GcpProvider — Google Cloud as a concrete CloudProvider.

Ties the GCP adapters (Gemini, Vertex AI Search, BigQuery, Model Armor, Cloud Monitoring) to the
neutral ``CloudProvider`` seam. Every factory is lazy and returns ``None`` when the deployment didn't
configure that capability (no engine id → no document retriever) so the caller falls back to the
in-tree default. The kernel never imports a google library; the adapters do, lazily.

Config (all optional; pass to ``get_provider("gcp", ...)``):
    project, location, credentials     - session
    model_tiers                        - [(tier, gemini_model_id, cost_per_1k)]
    vertex_engine_id / data_store_id   - document retriever
    bigquery_dataset                   - analytical backend
    model_armor_template               - content guardrail
"""
from __future__ import annotations

from ..base import (
    CloudProvider,
    Guardrail,
    IdentityBroker,
    ModelPlugin,
    RetrieverPlugin,
    TelemetryReader,
    WarehouseBackend,
)
from .session import GcpSession


class GcpProvider(CloudProvider):
    name = "gcp"

    def __init__(self, *, project=None, location="us-central1", credentials=None, api_key=None,
                 model_tiers=None, vertex_engine_id=None, data_store_id=None,
                 bigquery_dataset=None, model_armor_template=None):
        # api_key → Gemini Developer API (no project/ADC); omit for Vertex AI (project + ADC).
        self.session = GcpSession(project=project, location=location, credentials=credentials,
                                  api_key=api_key)
        self.model_tiers = model_tiers
        self.vertex_engine_id = vertex_engine_id
        self.data_store_id = data_store_id
        self.bigquery_dataset = bigquery_dataset
        self.model_armor_template = model_armor_template

    def model(self) -> ModelPlugin | None:
        from .gemini_model import GeminiModel
        return GeminiModel.from_config(self.session, tiers=self.model_tiers)

    def document_retriever(self) -> RetrieverPlugin | None:
        if not (self.vertex_engine_id or self.data_store_id):
            return None
        from .vertex_search_retriever import VertexSearchRetriever
        return VertexSearchRetriever(self.session, engine_id=self.vertex_engine_id,
                                     data_store_id=self.data_store_id)

    def managed_kb_retriever(self) -> RetrieverPlugin | None:
        # Vertex AI Search is the managed retrieval surface; exposed via document_retriever().
        return None

    def analytical_backend(self) -> WarehouseBackend | None:
        if not self.bigquery_dataset:
            return None
        from .bigquery_backend import BigQueryBackend
        return BigQueryBackend(self.session, dataset=self.bigquery_dataset)

    def guardrail(self) -> Guardrail | None:
        if not self.model_armor_template:
            return None
        from .model_armor import ModelArmorGuardrail
        return ModelArmorGuardrail(self.session, template=self.model_armor_template)

    def identity_broker(self) -> IdentityBroker | None:
        # A2A attested identity + workload identity are consumed at the agent layer, not brokered here.
        return None

    def telemetry_reader(self) -> TelemetryReader | None:
        from .cloud_monitoring import CloudMonitoringReader
        return CloudMonitoringReader(self.session)
