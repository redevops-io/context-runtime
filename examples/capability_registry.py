"""Capabilities live in the planner, not the prompt (Whitepaper v3).

Shows the CapabilityRegistry as the out-of-band catalog the planner selects from: the default registry
is lifted from the rule tables (so behavior is unchanged), and registering ONE new capability widens
the planner's search space immediately — no edit to the rules, nothing added to the model's context.

    python examples/capability_registry.py
"""
from __future__ import annotations

from context_runtime.registry import Capability, CapabilityRegistry
from context_runtime.planner.candidates_registry import RegistryCandidateGenerator
from context_runtime.runtime.runtime import ContextRuntime

QUERY = "how does the auth service relate to the billing outage across systems"   # → multi_hop intent


def methods(exp) -> list[str]:
    return sorted({next(s.params.get("method") for s in c.steps if s.type == "retrieve")
                   for c, _ in exp.candidates})


def main() -> None:
    registry = CapabilityRegistry.from_rules()
    runtime = ContextRuntime.default(
        docs=[{"id": "d1", "text": "The auth service change preceded the billing outage."}],
        candidates=RegistryCandidateGenerator(registry),
    )

    exp = runtime.explain(QUERY, analyze=False)
    print("Query:", QUERY)
    print("Intent bucket:", exp.chosen.intent.bucket)
    print("\nRegistry today —", len(registry.list("retrieval")), "retrieval capabilities,",
          len(registry.list("model_tier")), "model tiers.")
    print("Candidate retrieval methods for this intent:", methods(exp))

    # A new retrieval capability comes online (e.g. a temporal/bi-temporal graph index). Registering it
    # is one call — the planner can now select it wherever it serves; the prompt is untouched.
    registry.register(Capability(
        key="retrieval:temporal", kind="retrieval", value="temporal",
        buckets=frozenset({"multi_hop", "incident"}), cost_prior=0.55, quality_prior=0.9, local=True,
    ))
    exp2 = runtime.explain(QUERY, analyze=False)
    print("\nAfter registering 'temporal' (no rules edit, no prompt change):")
    print("Candidate retrieval methods for this intent:", methods(exp2))
    print("\nCapability count scaled in the registry — which is cheap — not in the context window.")


if __name__ == "__main__":
    main()
