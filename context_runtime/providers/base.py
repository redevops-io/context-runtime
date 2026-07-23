"""Cloud-provider seam — how a managed cloud plugs into Context Runtime without the kernel
importing a provider SDK.

The AWS-fit audit's central finding was that every missing integration is an *adapter behind an
existing Protocol*, never a kernel change. This module makes that mechanical for whole clouds: a
provider (AWS today; Azure, GCP, DigitalOcean next) implements ``CloudProvider`` by returning
objects that satisfy the runtime's existing plugin Protocols — ``ModelPlugin``, ``RetrieverPlugin`` —
plus a few small neutral capability Protocols defined here (``Guardrail``, ``IdentityBroker``,
``TelemetryReader``, ``WarehouseBackend``). The kernel never sees ``"aws"`` or ``boto3``; it sees a
``ModelPlugin`` and a ``RetrieverPlugin``.

Adding a provider is therefore: a new ``providers/<cloud>/`` subpackage implementing the pieces it
offers + one ``register_provider`` call. Nothing above the seam moves. A provider implements only
what it has — every ``CloudProvider`` method defaults to ``None``, and callers treat ``None`` as
"not offered, fall back to the local/in-tree default."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

# The runtime plugin Protocols a provider's objects must satisfy. Imported for type clarity only;
# providers return duck-typed objects, so a provider module never has to import the runtime.
from ..plugins.base import ModelPlugin, RetrieverPlugin  # noqa: F401  (re-exported for providers)


# ──────────────────────────── neutral capability Protocols ────────────────────────────
# These are the capabilities a managed cloud offers that the runtime doesn't already have a Protocol
# for. Deliberately tiny and provider-agnostic: Bedrock Guardrails and Azure Content Safety both
# satisfy ``Guardrail``; CloudWatch and Azure Monitor both satisfy ``TelemetryReader``.


@dataclass(frozen=True)
class GuardrailVerdict:
    """Outcome of a content-guardrail check on model input or output."""
    allowed: bool
    action: str = "none"                 # none | blocked | masked
    text: str | None = None              # redacted/masked text when action == "masked"
    reasons: tuple[str, ...] = ()


@runtime_checkable
class Guardrail(Protocol):
    """Content guardrail over model I/O (prompt-injection, toxicity, PII). A managed service
    (Bedrock Guardrails, Azure Content Safety) or a local classifier both satisfy this."""

    def check_input(self, text: str) -> GuardrailVerdict: ...
    def check_output(self, text: str) -> GuardrailVerdict: ...


@runtime_checkable
class IdentityBroker(Protocol):
    """Delegated-access broker. Returns a short-lived token for ``subject`` to call ``scope``
    (e.g. a user's Google/GitHub/Slack token) — WITHOUT the runtime running an OAuth server.
    AgentCore Identity is the AWS implementation; the runtime just consumes the token."""

    def token_for(self, subject: str, scope: str) -> str | None: ...


@runtime_checkable
class TelemetryReader(Protocol):
    """Read-only operational telemetry (metrics/logs/traces) for the post-deploy monitor loop.
    CloudWatch, Azure Monitor, GCP Cloud Monitoring all satisfy this."""

    def query(self, expr: str, window_s: int = 300) -> list[dict]: ...


@runtime_checkable
class WarehouseBackend(Protocol):
    """A structured/analytical store the AnalyticalRetriever runs generated SQL against. A local
    DuckDB file and AWS Athena both satisfy this — the text-to-SQL engine is provider-neutral, only
    the execution backend swaps. Read-only by contract."""

    def schema(self) -> str: ...                        # DDL / column summary for the SQL generator
    def run_sql(self, sql: str, max_rows: int = 100) -> list[dict]: ...
    def dialect(self) -> str: ...                       # "duckdb" | "athena" | "postgres" | …


# ──────────────────────────── the provider itself ────────────────────────────


class CloudProvider:
    """Base class every provider subclasses. Each factory returns a plugin the runtime already knows
    how to use, or ``None`` when the provider doesn't offer that capability (caller falls back to the
    in-tree default). Providers override only the pieces they implement.

    Construction is provider-specific (region, credentials, resource ids); pass config via kwargs to
    ``get_provider(name, **cfg)``. Keeping the factories lazy means importing this module never pulls
    a cloud SDK — the SDK import happens only when a factory is actually called.
    """

    name: str = "base"

    # model plane -----------------------------------------------------------------
    def model(self) -> ModelPlugin | None:
        """A ModelPlugin over the provider's managed models (Bedrock, Azure OpenAI, Vertex)."""
        return None

    # retrieval plane -------------------------------------------------------------
    def document_retriever(self) -> RetrieverPlugin | None:
        """The provider's managed search engine as a document RetrieverPlugin (OpenSearch, Azure AI
        Search, Vertex Search)."""
        return None

    def managed_kb_retriever(self) -> RetrieverPlugin | None:
        """A managed retrieval-augmented knowledge base as a RetrieverPlugin (Bedrock Knowledge
        Bases, Azure on-your-data, Vertex RAG)."""
        return None

    def analytical_backend(self) -> WarehouseBackend | None:
        """A structured warehouse for the analytical (text-to-SQL) representation (Athena, Synapse,
        BigQuery, Managed Postgres)."""
        return None

    # governance plane ------------------------------------------------------------
    def guardrail(self) -> Guardrail | None:
        return None

    def identity_broker(self) -> IdentityBroker | None:
        return None

    # operations plane ------------------------------------------------------------
    def telemetry_reader(self) -> TelemetryReader | None:
        return None

    def info(self) -> dict:
        """Which capabilities this provider instance actually offers (factories that return non-None).
        Cheap enough to call — factories that only construct clients lazily won't hit the network."""
        offered = {}
        for cap in ("model", "document_retriever", "managed_kb_retriever", "analytical_backend",
                    "guardrail", "identity_broker", "telemetry_reader"):
            try:
                offered[cap] = getattr(self, cap)() is not None
            except Exception:  # noqa: BLE001 — a missing SDK/creds means "not offered here"
                offered[cap] = False
        return {"provider": self.name, "capabilities": offered}


# ──────────────────────────── registry ────────────────────────────

ProviderFactory = Callable[..., CloudProvider]
_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under ``name`` (e.g. "aws"). Idempotent; last registration wins."""
    _REGISTRY[name.lower()] = factory


def get_provider(name: str, **config) -> CloudProvider:
    """Construct a registered provider by name. ``config`` is passed to the provider factory
    (region, credentials, resource ids). Raises KeyError for an unknown provider."""
    key = name.lower()
    if key == "do":                     # common alias → canonical name
        key = "digitalocean"
    if key not in _REGISTRY:
        # lazy self-registration for built-ins, so callers don't have to import the subpackage
        if key == "aws":
            from .aws import register as _reg
            _reg()
        elif key == "gcp":
            from .gcp import register as _reg
            _reg()
        elif key == "digitalocean":
            from .digitalocean import register as _reg
            _reg()
    if key not in _REGISTRY:
        raise KeyError(f"unknown cloud provider '{name}'; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**config)


def available_providers() -> list[str]:
    return sorted(_REGISTRY)
