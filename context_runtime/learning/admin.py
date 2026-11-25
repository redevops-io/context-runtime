"""What an operator can see and undo in one persisted learned policy.

    python -m context_runtime.learning.admin status   <policy.json>
    python -m context_runtime.learning.admin rollback <policy.json> --by NAME [--source S] [--principal P] [--run-id R] [--apply]
    python -m context_runtime.learning.admin reset    <policy.json> --by NAME [--context CTX] [--apply]

STOP THE SERVICE that uses the policy before ``--apply``: a running process holds the policy in memory and writes it back.
Without ``--apply`` nothing is changed; rollback prints what it would take back. To stop learning without touching what was
learned, set ``CR_LEARNING_FROZEN=1`` on the service.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from dataclasses import dataclass

from ..integrations.bandit import EpsilonGreedyBandit
from . import ledger


@dataclass(frozen=True)
class _Arm:
    key: str


def _open(path: str) -> EpsilonGreedyBandit:
    # looking at a policy must not move or rewrite it: an unreadable one is reported, not set aside by this command
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    arms = sorted({k for v in raw.values() for k in v})
    return EpsilonGreedyBandit(tuple(_Arm(k) for k in arms), persist_path=path)


def _wanted(args):
    want = {k: v for k, v in (("source", args.source), ("principal", args.principal), ("run_id", args.run_id)) if v}
    return want, (lambda e: all(e.get("provenance", {}).get(k) == v for k, v in want.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m context_runtime.learning.admin", description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("status", "rollback", "reset"))
    ap.add_argument("policy")
    ap.add_argument("--by")
    ap.add_argument("--source")
    ap.add_argument("--principal")
    ap.add_argument("--run-id")
    ap.add_argument("--context")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)
    if not os.path.isfile(args.policy):
        print(f"no policy file at {args.policy}", file=sys.stderr)
        return 1
    try:
        entries = ledger.read(f"{args.policy}.ledger.jsonl")
        b = _open(args.policy)
    except (ValueError, ledger.LedgerIncomplete) as exc:
        print(f"cannot read it: {exc}", file=sys.stderr)
        return 1
    if args.command == "status":
        by = collections.Counter(e["provenance"].get("source", "?") for e in entries if e["op"] == "update")
        print(json.dumps({**b.status(), "reproduced_by_ledger": ledger.same_learning(ledger.rebuild(entries), b.stats),
                          "updates_in_ledger_by_source": dict(by), "learned_policy": b.policy()}, indent=1))
        return 0
    if not args.by:
        print("--by NAME is required: a change to what was learned names who made it", file=sys.stderr)
        return 2
    if args.command == "reset":
        print(f"{'resetting' if args.apply else 'would reset'}: {args.context or 'every context'}")
        if args.apply:
            b.reset(args.context, by=args.by)
        return 0
    want, match = _wanted(args)
    if not want:
        print("say which observations: --source, --principal and/or --run-id", file=sys.stderr)
        return 2
    gone = {s for e in entries if e["op"] == "rollback" for s in e.get("excluded", ())}
    chosen = [e for e in entries if e["op"] == "update" and e["seq"] not in gone and match(e)]
    print(f"{len(chosen)} observation(s) match {want}")
    if args.apply and chosen:
        try:
            print(f"taken back: {b.rollback(match, by=args.by)}")
        except ledger.LedgerIncomplete as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
