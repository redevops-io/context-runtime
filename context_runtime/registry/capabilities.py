"""Capability Registry — capabilities live in the PLANNER, not the prompt (Whitepaper v3).

The paper's pivotal scaling property: an assistant's abilities (retrieval methods, model tiers,
reasoners, tools) are registered *out of band*, each with cost/quality priors and the intent buckets
it serves. The planner selects among them using intent, policy, and cost — the model never sees a tool
catalog in its context window. "Capability count scales in a registry — which is cheap — rather than
in the context window, which is not."

This module is the catalog. ``RegistryCandidateGenerator`` (planner/candidates_registry.py) consults it
to enumerate candidate plans, so adding a capability is a ``register()`` call, not an edit to the rule
tables or the prompt. ``CapabilityRegistry.from_rules()`` lifts the existing v0.1 rule tables into
registry form, so the default registry is behavior-equivalent to the hand-written planner.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capability:
    """One registered ability the planner may compose into a plan.

    ``value`` is the concrete token that lands in a StepSpec (a retrieval method like ``"hybrid"``, a
    model tier like ``"premium"``). ``buckets`` are the intent buckets it serves (``"*"`` = all). The
    priors are relative [0, 1] hints the cost model / planner can use before real statistics exist.
    ``local`` marks an in-house resource (a policy hint — e.g. a local model tier vs. a hosted one).
    """
    key: str                                        # unique id, e.g. "retrieval:hybrid" | "model_tier:local"
    kind: str                                       # "retrieval" | "model_tier" | "reasoner" | "tool"
    value: str                                      # the token used in a StepSpec / model_tier
    buckets: frozenset[str] = frozenset({"*"})      # intent buckets served
    cost_prior: float = 0.5                         # relative $ prior [0,1] (higher = pricier)
    quality_prior: float = 0.5                      # relative quality prior [0,1] (higher = better)
    local: bool = False                             # served by an in-house / local resource
    tags: frozenset[str] = frozenset()
    meta: dict = field(default_factory=dict)

    def serves(self, bucket: str) -> bool:
        return "*" in self.buckets or bucket in self.buckets


class CapabilityRegistry:
    """An ordered catalog of Capabilities the planner selects from, keyed by ``Capability.key``."""

    def __init__(self, capabilities: tuple[Capability, ...] | list[Capability] = ()):
        self._by_key: dict[str, Capability] = {}
        for c in capabilities:
            self.register(c)

    def register(self, cap: Capability) -> "CapabilityRegistry":
        """Add (or replace) a capability. Idempotent by key — re-registering updates in place."""
        self._by_key[cap.key] = cap
        return self

    def get(self, key: str) -> Capability | None:
        return self._by_key.get(key)

    def list(self, kind: str | None = None) -> list[Capability]:
        return [c for c in self._by_key.values() if kind is None or c.kind == kind]

    def for_intent(self, bucket: str, kind: str) -> list[Capability]:
        """Capabilities of ``kind`` that serve ``bucket``, best-quality first."""
        return sorted(
            (c for c in self.list(kind) if c.serves(bucket)),
            key=lambda c: (-c.quality_prior, c.cost_prior, c.key),
        )

    def values_for(self, bucket: str, kind: str) -> list[str]:
        return [c.value for c in self.for_intent(bucket, kind)]

    # ──────────────────────────── defaults ────────────────────────────

    @classmethod
    def from_rules(cls) -> "CapabilityRegistry":
        """Lift the v0.1 rule tables (planner/rules.py) into registry form.

        Retrieval methods and model tiers become Capabilities whose ``buckets`` are exactly the buckets
        that reference them in ``BUCKET_DEFAULTS`` / ``BUCKET_TIERS``. This keeps a single source of truth
        (the rules) while exposing it as a registry a caller can extend with ``register()``.
        """
        from ..planner import rules

        # relative priors (used before measured statistics exist)
        _METHOD_PRIOR = {   # method: (cost, quality)
            "bm25": (0.10, 0.50), "vector": (0.30, 0.70), "hybrid": (0.40, 0.80),
            "code": (0.40, 0.75), "graph": (0.60, 0.85), "community": (0.70, 0.80),
        }
        _TIER_PRIOR = {     # tier: (cost, quality, local)
            "local": (0.10, 0.55, True), "cheap": (0.30, 0.72, False), "premium": (0.90, 0.95, False),
        }

        reg = cls()
        # retrieval capabilities: a method serves every bucket whose defaults list it
        method_buckets: dict[str, set[str]] = {}
        for bucket, (methods, _strategy, _verify) in rules.BUCKET_DEFAULTS.items():
            for m in methods:
                method_buckets.setdefault(m, set()).add(bucket)
        for method, buckets in method_buckets.items():
            cost, quality = _METHOD_PRIOR.get(method, (0.5, 0.6))
            reg.register(Capability(
                key=f"retrieval:{method}", kind="retrieval", value=method,
                buckets=frozenset(buckets), cost_prior=cost, quality_prior=quality, local=True,
                tags=frozenset({"retrieval"}),
            ))

        # model-tier capabilities: a tier serves every bucket whose tier list includes it
        tier_buckets: dict[str, set[str]] = {}
        for bucket, tiers in rules.BUCKET_TIERS.items():
            for t in tiers:
                tier_buckets.setdefault(t, set()).add(bucket)
        for tier, buckets in tier_buckets.items():
            cost, quality, local = _TIER_PRIOR.get(tier, (0.5, 0.6, False))
            reg.register(Capability(
                key=f"model_tier:{tier}", kind="model_tier", value=tier,
                buckets=frozenset(buckets), cost_prior=cost, quality_prior=quality, local=local,
                tags=frozenset({"model"}),
            ))
        return reg
