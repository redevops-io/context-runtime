"""Temporal / bi-temporal retrieval — "what changed, and when?" (Whitepaper v3, forthcoming retrieval).

Every fact carries valid time (when it's true in the world) and transaction time (when we learned it),
so the planner can answer point-in-time questions the other retrieval methods can't.

    python examples/temporal_retrieval.py
"""
from __future__ import annotations

from context_runtime.adapters.store_temporal import TemporalStore


def owner(hits):
    return hits[0].meta["object"] if hits else "(none on record)"


def main():
    s = TemporalStore()
    # the on-call owner of the auth service, revised over time — and one correction filed late
    s.add("auth-service", "owner", "Alice", valid_from="2026-01-01", valid_to="2026-06-01", recorded_at="2026-01-01")
    s.add("auth-service", "owner", "Bob", valid_from="2026-06-01", valid_to=None, recorded_at="2026-07-15")

    print("Current owner of auth-service:", owner(s.search("auth-service owner")))
    print()
    print("Valid-time travel (what was true in the world, using the corrected record):")
    for at in ("2026-03-01", "2026-06-15", "2026-09-01"):
        print(f"  as of {at}:  {owner(s.as_of('auth-service', at=at))}")

    print()
    print("Transaction-time travel (what we BELIEVED on a past date):")
    print(f"  as of 2026-06-15, as known then:  {owner(s.as_of('auth-service', at='2026-06-15', known_at='2026-06-15'))}")
    print("  → Bob's record was filed 2026-07-15, so on 2026-06-15 we hadn't recorded it yet — a late")
    print("    correction doesn't silently rewrite what the system knew at the time.")

    print()
    print("What changed, and when? (2026):")
    for c in s.changes("auth-service", since="2026-01-01", until="2027-01-01"):
        print(f"  {c['at']}  {c['change']:6} {c['fact']}")


if __name__ == "__main__":
    main()
