"""Warm-start priors from the benchmark — the Phase 0 → Phase 1 bridge.

`priors_from_ablation` turns eval_cube2 oracle cells into per-bucket strategy ladders (cheapest-capable
first); `load_priors`/`set_priors` apply them so a deployment's warm start is measured, not guessed.
"""
from __future__ import annotations

import json

import pytest

from context_runtime.reasoner import strategies as gs


@pytest.fixture(autouse=True)
def _restore_priors():
    """These tests mutate the module-global active ladders; snapshot + restore so they don't leak."""
    snap = dict(gs._ACTIVE_PRIORS)
    yield
    gs._ACTIVE_PRIORS.clear()
    gs._ACTIVE_PRIORS.update(snap)


def _cell(tmp, dataset, strategy, oracle):
    (tmp / f"{dataset}__hybrid__qwen__{strategy}.json").write_text(json.dumps(
        {"dataset": dataset, "method": "hybrid", "model": "qwen", "strategy": strategy,
         "n": 12, "acc_oracle": oracle}))


def _ablation(tmp):
    # musique (→multi_hop): decompose best, reason within margin, direct far below
    _cell(tmp, "musique", "direct", 0.40); _cell(tmp, "musique", "reason", 0.85); _cell(tmp, "musique", "decompose", 0.90)
    # longmemeval (→temporal): only mapreduce clears the bar
    _cell(tmp, "longmemeval", "direct", 0.00); _cell(tmp, "longmemeval", "reason", 0.10); _cell(tmp, "longmemeval", "mapreduce", 0.60)
    # popqa (→exact_lookup): terse and reason tie at ceiling
    _cell(tmp, "popqa", "direct", 1.00); _cell(tmp, "popqa", "reason", 1.00)


def test_priors_from_ablation_builds_cost_ordered_ladders(tmp_path):
    _ablation(tmp_path)
    priors = gs.priors_from_ablation(str(tmp_path), cond="oracle", margin=0.1)
    # multi_hop: keep the two within-margin winners, cheapest (reason) as the entry point
    assert priors["multi_hop"] == ("reason", "decompose")
    # temporal: only mapreduce clears best-margin → it's the sole rung (must-mapreduce)
    assert priors["temporal"] == ("mapreduce",)
    # exact_lookup: both at ceiling → cheapest-first, and the bench 'direct' is aliased to 'terse'
    assert priors["exact_lookup"] == ("terse", "reason")


def test_load_priors_overrides_strategies_for(tmp_path):
    p = tmp_path / "priors.json"
    p.write_text(json.dumps({"multi_hop": ["decompose", "reason"], "temporal": ["mapreduce"]}))
    applied = gs.load_priors(str(p))
    assert applied["temporal"] == ["mapreduce"]
    # set_priors re-orders cheapest-first regardless of input order → reason before decompose
    assert gs.strategies_for("multi_hop") == ("reason", "decompose")
    assert gs.strategies_for("temporal") == ("mapreduce",)


def test_set_priors_drops_unknown_strategies_and_aliases_direct():
    gs.set_priors({"exact_lookup": ["direct", "bogus", "reason"]})
    assert gs.strategies_for("exact_lookup") == ("terse", "reason")   # direct→terse, bogus dropped


def test_full_loop_ablation_to_active_ladder(tmp_path):
    _ablation(tmp_path)
    gs.set_priors(gs.priors_from_ablation(str(tmp_path)))
    assert gs.strategies_for("temporal") == ("mapreduce",)
    assert gs.strategies_for("multi_hop")[0] == "reason"   # measured entry point
