"""Assemble a Context Runtime from a CloudProvider + local defaults — one call to wire a cloud in.

This is the payoff of the seam: given a provider (``get_provider("aws", ...)``) and the local in-tree
retrievers, build the ``models`` + ``retriever`` a ``ContextRuntime`` needs, with the cloud's pieces
slotted in where offered and the in-tree defaults everywhere else. Swapping clouds swaps the provider
argument; nothing else changes.

What gets wired, each only if the provider offers it (else the local default stands):
  • model      → the provider's managed model plane (Bedrock), expanded per CR tier, optionally
                 wrapped in the provider's guardrail.
  • document   → the provider's managed KB / search engine (Bedrock KB / OpenSearch) as the router's
                 single-hop arm.
  • analytical → the provider's warehouse (Athena) behind the neutral text-to-SQL AnalyticalRetriever,
                 with the model as the SQL generator.
"""
from __future__ import annotations

from ..adapters.store_analytical import AnalyticalRetriever
from ..adapters.store_router import HopRouterRetriever
from .guarded_model import GuardedModel


def build_runtime_kwargs(provider, *, local_single_hop, local_graph=None, local_community=None,
                         local_temporal=None, base_model=None, guard: bool = True):
    """Return ``{"models", "retriever"}`` to splat into ``ContextRuntime(...)``.

    ``local_single_hop``/``local_graph``/… are the in-tree retrievers used where the provider offers
    nothing. ``base_model`` is the model plane to use when the provider has no managed model.
    """
    # ── model plane ──
    pmodel = provider.model()
    if pmodel is not None:
        models = pmodel.per_tier_models() if hasattr(pmodel, "per_tier_models") else pmodel
        sql_model = pmodel  # a concrete ModelPlugin for the analytical SQL generator
    else:
        models = base_model
        sql_model = base_model

    # optional content guardrail over the whole model plane (provider-neutral wrapper)
    g = provider.guardrail() if guard else None
    if g is not None and models is not None:
        if isinstance(models, dict):
            models = {t: GuardedModel(m, g) for t, m in models.items()}
        else:
            models = GuardedModel(models, g)

    # ── retrieval plane ──
    document = provider.managed_kb_retriever() or provider.document_retriever() or local_single_hop

    analytical = None
    backend = provider.analytical_backend()
    if backend is not None and sql_model is not None:
        analytical = AnalyticalRetriever(backend, sql_model)

    retriever = HopRouterRetriever(
        single_hop=document, graph=local_graph, community=local_community,
        temporal=local_temporal, analytical=analytical,
    )
    return {"models": models, "retriever": retriever}


def build_runtime(provider, *, local_single_hop, base_model=None, learning: bool = False, **kw):
    """Convenience: construct a ContextRuntime wired to ``provider`` (thin wrapper over the kwargs)."""
    from ..runtime.runtime import ContextRuntime
    rt_kwargs = build_runtime_kwargs(provider, local_single_hop=local_single_hop,
                                     base_model=base_model, **kw)
    return ContextRuntime(learning=learning, **rt_kwargs)
