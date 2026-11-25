"""Where a learned value came from, and how to take a bad observation back out.

A bandit keeps running means. Without a record of the observations behind them, a poisoned or mistaken signal cannot be
found, and the only undo is deleting the policy. This is that record: an append-only JSON-lines file next to the policy.

Entries (``seq`` orders them):
  baseline  the state when the ledger began (a policy learned before there was a ledger)
  update    one observation: context, arm, reward, the step size used, and WHO or WHAT produced it
  reset     an operator cleared one context, or everything
  rollback  an operator excluded earlier updates (by ``seq``); later rebuilds skip them, so rollbacks compose

``rebuild`` replays the entries. A rollback is applied only if replaying the ledger as it stands reproduces the live
policy: if an entry was ever lost, the ledger cannot say what the policy would have been, and it refuses.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
from typing import Any, Iterable, Iterator, Mapping

#: what may be said about an observation. A closed set: this file is not a place to put request or customer data.
PROVENANCE_KEYS = ("source", "run_id", "principal", "plan_hash")
UNATTRIBUTED = {"source": "unattributed"}
_ambient: contextvars.ContextVar[Mapping[str, str] | None] = contextvars.ContextVar("cr_learning_provenance", default=None)


class LedgerIncomplete(RuntimeError):
    """The ledger does not reproduce the live policy, so it cannot be used to change it."""


def clean_provenance(p: Mapping[str, Any] | None) -> dict[str, str]:
    if p is None:
        p = _ambient.get()
    if not p:
        return dict(UNATTRIBUTED)
    extra = sorted(set(p) - set(PROVENANCE_KEYS))
    if extra:
        raise ValueError(f"provenance takes only {', '.join(PROVENANCE_KEYS)}; not {', '.join(extra)}")
    out = {k: str(p[k])[:200] for k in PROVENANCE_KEYS if p.get(k) not in (None, "")}
    return out or dict(UNATTRIBUTED)


@contextlib.contextmanager
def provenance(**who: str) -> Iterator[None]:
    """Attribute every observation recorded inside the block, without changing the learners in between:

        with provenance(source="verifier", run_id=run.id, principal=user):
            tenant.record_outcome(question, success)
    """
    token = _ambient.set(clean_provenance(who))
    try:
        yield
    finally:
        _ambient.reset(token)


@contextlib.contextmanager
def default_provenance(**who: str) -> Iterator[None]:
    """For a learner's own call site: name the signal, unless an enclosing block already said who produced it."""
    if _ambient.get():
        yield
    else:
        with provenance(**who):
            yield


def read(path: str) -> list[dict]:
    """Entries in ``seq`` order. A line that cannot be read raises: a silently shortened history is not a history."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
                int(e["seq"]), str(e["op"])
            except Exception as exc:      # noqa: BLE001
                raise LedgerIncomplete(f"line {n} of the ledger cannot be read") from exc
            out.append(e)
    return sorted(out, key=lambda e: e["seq"])


def rebuild(entries: Iterable[dict], *, exclude: Iterable[int] = ()) -> dict[str, dict[str, list[float]]]:
    entries = list(entries)
    skipped = set(exclude)
    for e in entries:
        if e["op"] == "rollback":
            skipped.update(e.get("excluded", ()))
    state: dict[str, dict[str, list[float]]] = {}
    for e in entries:
        op = e["op"]
        if op == "baseline":
            state = {c: {k: list(v) for k, v in arms.items()} for c, arms in e["stats"].items()}
        elif op == "reset":
            if e.get("ctx") is None:
                state = {}
            else:
                state.pop(e["ctx"], None)
        elif op == "update" and e["seq"] not in skipped:
            arms = state.setdefault(e["ctx"], {})
            n, mean = arms.get(e["arm"], [0.0, e["optimistic"]])
            n += 1
            alpha = e["discount"] if e["discount"] > 0.0 else 1.0 / n
            arms[e["arm"]] = [n, mean + alpha * (e["reward"] - mean)]
    return state


def same_learning(a: Mapping[str, Mapping[str, list]], b: Mapping[str, Mapping[str, list]]) -> bool:
    """Equal in everything that was LEARNED. Arms nobody has observed (n = 0) are placeholders and do not count."""
    def seen(s):
        return {(c, k): (v[0], v[1]) for c, arms in s.items() for k, v in arms.items() if v[0] > 0}
    x, y = seen(a), seen(b)
    return x.keys() == y.keys() and all(x[k][0] == y[k][0] and abs(x[k][1] - y[k][1]) < 1e-9 for k in x)
