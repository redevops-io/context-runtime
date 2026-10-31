"""An independent check of a plan: it knows only the plan and the declared graph, and runs the plan the way a runner
would (lanes in order, gates as start conditions) with arbitrary real durations."""
from __future__ import annotations

import random

from context_runtime.qa import NodeKind


def locks_of(graph) -> dict[str, set[str]]:
    """Exclusive resources each test needs, found by walking the declared graph (not through the planner's code)."""
    nodes = {(n.kind, n.id): n for n in graph.nodes()}
    out: dict[str, set[str]] = {}
    for (kind, ident), node in nodes.items():
        if kind is not NodeKind.TEST:
            continue
        held: set[str] = set()
        seen = set()
        stack = [node]
        while stack:
            cur = stack.pop()
            for c in cur.own_claims():
                if c.exclusive:
                    held.add(c.resource)
            for ref in cur.requires:
                nxt = nodes.get((ref.kind, ref.id))
                if nxt is not None and nxt.kind is not NodeKind.TEST and (nxt.kind, nxt.id) not in seen:
                    seen.add((nxt.kind, nxt.id))
                    stack.append(nxt)
        out[ident] = held
    return out


def run_plan(plan, durations: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Start and finish of every item when the runner follows the plan and item durations are as given."""
    by_lane: dict[str, list] = {}
    for item in plan.items:
        by_lane.setdefault(item.lane, []).append(item)
    for items in by_lane.values():
        items.sort(key=lambda i: i.order)
    gates = {g.item: g.after for g in plan.gates}
    prev = {}
    for items in by_lane.values():
        for k, item in enumerate(items):
            prev[item.spec_ref] = items[k - 1].spec_ref if k else None
    done: dict[str, tuple[int, int]] = {}
    pending = {i.spec_ref for i in plan.items}
    while pending:
        progressed = False
        for ref in sorted(pending):
            waits = [w for w in (prev[ref], *gates.get(ref, ())) if w is not None]
            if all(w in done for w in waits):
                start = max((done[w][1] for w in waits), default=0)
                done[ref] = (start, start + durations[ref])
                pending.discard(ref)
                progressed = True
        assert progressed, "the plan can never finish: its lane order and gates deadlock"
    return done


def max_overlap(intervals: dict[str, tuple[int, int]], weight=lambda ref: 1) -> int:
    events = []
    for ref, (s, f) in intervals.items():
        events.append((s, 1, weight(ref)))
        events.append((f, 0, -weight(ref)))
    events.sort()                       # at equal times, finishes (0) are processed before starts (1)
    cur = best = 0
    for _, _, w in events:
        cur += w
        best = max(best, cur)
    return best


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def random_durations(plan, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    return {i.spec_ref: rng.randint(1, 1000) for i in sorted(plan.items, key=lambda i: i.spec_ref)}
