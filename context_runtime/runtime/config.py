"""Configuration (SPEC §8, plugin-first shape).

Parses context_runtime.yaml into a Config the runtime uses to wire plugins by name. Keeps
the file shape close to ARCHITECTURE §8 so the same plan runs local or cloud by
swapping the ``store``/``model`` plugin names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    model: str = "stub"               # "stub" | "litellm"
    store: str = "inmemory"           # "inmemory" | "redevops_rag"
    default_tier: str = "local"
    top_k: int = 50
    # Opt-in deterministic fast path: with Constraints.require_exact, an unambiguous, fully covered typed operation is executed without plan
    # selection, context construction or a model call. Not taken when the optimizer carries a policy (the policy is evaluated per plan).
    fast_path: bool = False
    # Check a model's narration of typed results against the facts the provider attached (omitted or changed figures); the structured
    # rendering is served instead of a narration that fails. Inert unless a provider attaches facts.
    verify_narration: bool = True
    natural_question: bool = False    # True: reasoners are asked the user's own question instead of the normalised (sorted, de-duplicated) cache key
    final_k: int = 8
    target_tokens: int = 3000
    max_tokens: int | None = 120000
    max_cost_usd: float | None = 5.0
    max_latency_seconds: float | None = 120.0
    weights: dict[str, float] = field(default_factory=dict)
    trace_dir: str | None = None
    stats_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml  # pyyaml is a core dep

        data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
        rt = data.get("runtime", {})
        plugins = rt.get("plugins", {})
        budgets = data.get("budgets", {})
        cm = data.get("costmodel", {})
        obs = data.get("observability", {})
        retr = data.get("retrieval", {})
        return cls(
            model=plugins.get("model", "stub"),
            store=plugins.get("store", "inmemory"),
            default_tier=rt.get("default_tier", "local"),
            top_k=retr.get("top_k", 50),
            final_k=retr.get("final_k", 8),
            target_tokens=retr.get("target_tokens", 3000),
            max_tokens=budgets.get("max_tokens", 120000),
            max_cost_usd=budgets.get("max_cost_usd", 5.0),
            max_latency_seconds=budgets.get("max_latency_seconds", 120.0),
            weights=cm.get("weights", {}),
            trace_dir=obs.get("trace_dir"),
            stats_path=cm.get("stats_path"),
            raw=data,
        )
