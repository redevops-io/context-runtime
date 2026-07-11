"""Bi-temporal retrieval: point-in-time queries over valid time AND transaction time, and a
"what changed, and when?" scan. Dependency-free; satisfies the RetrieverPlugin seam."""
from __future__ import annotations

from context_runtime.adapters.store_temporal import TemporalStore
from context_runtime.plugins.base import RetrieverPlugin


def _store() -> TemporalStore:
    s = TemporalStore()
    # the auth service's owner is revised over time (valid-time history)
    s.add("auth-service", "owner", "Alice", valid_from="2026-01-01", valid_to="2026-06-01",
          recorded_at="2026-01-01")
    s.add("auth-service", "owner", "Bob", valid_from="2026-06-01", valid_to=None,
          recorded_at="2026-07-15")   # recorded LATE (a correction filed in July)
    s.add("billing-service", "owner", "Carol", valid_from="2026-02-01", valid_to=None,
          recorded_at="2026-02-01")
    return s


def test_satisfies_retriever_plugin_seam():
    assert isinstance(_store(), RetrieverPlugin)
    info = _store().info()
    assert "temporal" in info.capabilities and info.name == "temporal"


def test_current_state_returns_the_latest_valid_fact():
    hits = _store().search("who owns auth-service", k=5)
    owners = [h.meta["object"] for h in hits if h.meta["subject"] == "auth-service"]
    assert owners == ["Bob"]        # Alice's record was superseded (valid_to set)


def test_as_of_valid_time_travels_the_world_history():
    s = _store()
    assert s.as_of("auth-service", at="2026-03-01")[0].meta["object"] == "Alice"
    assert s.as_of("auth-service", at="2026-09-01")[0].meta["object"] == "Bob"
    # boundary: valid_from inclusive, valid_to exclusive
    assert s.as_of("auth-service", at="2026-06-01")[0].meta["object"] == "Bob"


def test_as_of_transaction_time_ignores_later_corrections():
    """As known on 2026-06-15, Bob's record (filed 2026-07-15) didn't exist yet — so even at a world
    time inside Bob's validity, the state we *believed then* still shows Alice (whose interval we knew)."""
    s = _store()
    seen = s.as_of("auth-service", at="2026-06-15", known_at="2026-06-15")
    # Bob is filtered out (recorded_at 2026-07-15 > known_at); Alice's interval ended 2026-06-01, so
    # nothing valid AND known → empty (we honestly had no recorded owner for that instant yet)
    assert [h.meta["object"] for h in seen] == []
    # without the transaction-time bound, the corrected history shows Bob
    assert s.as_of("auth-service", at="2026-06-15")[0].meta["object"] == "Bob"


def test_changes_reports_what_changed_and_when():
    s = _store()
    changes = s.changes("auth-service", since="2026-01-01", until="2027-01-01")
    # Alice began (01-01) + ended (06-01), Bob began (06-01)
    assert [(c["at"], c["change"], c["object"]) for c in changes] == [
        ("2026-01-01", "began", "Alice"),
        ("2026-06-01", "ended", "Alice"),
        ("2026-06-01", "began", "Bob"),
    ]
