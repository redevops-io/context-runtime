"""AwsProvider — AWS as the first concrete CloudProvider.

Ties the AWS adapters (Bedrock model, OpenSearch, Bedrock KB, Athena analytical, Bedrock Guardrails)
to the neutral ``CloudProvider`` seam. Every factory is lazy: it constructs its adapter only when
called, and returns ``None`` when the deployment didn't configure that capability (no KB id → no
managed KB retriever) — the caller falls back to the in-tree default. Nothing here is imported by the
kernel; the kernel only ever holds the ModelPlugin / RetrieverPlugin these factories return.

Config (all optional; pass to ``get_provider("aws", ...)`` or ``AwsProvider(...)``):
    region, role_arn, profile        — credentials/region (session.py)
    model_tiers                      — list of (tier_name, bedrock_model_id, cost_per_1k)
    opensearch_endpoint, os_index    — document retriever
    knowledge_base_id                — managed KB retriever
    athena_database/workgroup/output — analytical backend
    guardrail_id, guardrail_version  — content guardrail
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
from .session import AwsSession


class AwsProvider(CloudProvider):
    name = "aws"

    def __init__(self, *, region=None, role_arn=None, profile=None, session=None,
                 model_tiers=None, opensearch_endpoint=None, os_index="documents",
                 knowledge_base_id=None,
                 athena_database=None, athena_workgroup="primary", athena_output=None,
                 guardrail_id=None, guardrail_version="DRAFT"):
        self.session = AwsSession(region=region, role_arn=role_arn, profile=profile, session=session)
        self.model_tiers = model_tiers
        self.opensearch_endpoint = opensearch_endpoint
        self.os_index = os_index
        self.knowledge_base_id = knowledge_base_id
        self.athena_database = athena_database
        self.athena_workgroup = athena_workgroup
        self.athena_output = athena_output
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version

    def model(self) -> ModelPlugin | None:
        from .bedrock_model import BedrockModel
        return BedrockModel.from_config(self.session, tiers=self.model_tiers)

    def document_retriever(self) -> RetrieverPlugin | None:
        if not self.opensearch_endpoint:
            return None
        from .opensearch_retriever import OpenSearchRetriever
        return OpenSearchRetriever(self.session, endpoint=self.opensearch_endpoint, index=self.os_index)

    def managed_kb_retriever(self) -> RetrieverPlugin | None:
        if not self.knowledge_base_id:
            return None
        from .bedrock_kb_retriever import BedrockKBRetriever
        return BedrockKBRetriever(self.session, knowledge_base_id=self.knowledge_base_id)

    def analytical_backend(self) -> WarehouseBackend | None:
        if not (self.athena_database and self.athena_output):
            return None
        from .athena_backend import AthenaBackend
        return AthenaBackend(self.session, database=self.athena_database,
                             workgroup=self.athena_workgroup, output_location=self.athena_output)

    def guardrail(self) -> Guardrail | None:
        if not self.guardrail_id:
            return None
        from .guardrail import BedrockGuardrail
        return BedrockGuardrail(self.session, guardrail_id=self.guardrail_id,
                                version=self.guardrail_version)

    def identity_broker(self) -> IdentityBroker | None:
        # AgentCore Identity broker — wired when the account is AgentCore-enabled; None until then.
        return None

    def telemetry_reader(self) -> TelemetryReader | None:
        from .cloudwatch import CloudWatchReader
        return CloudWatchReader(self.session)
