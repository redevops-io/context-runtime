"""contextos CLI — run / explain / simulate / index over a folder corpus."""
from __future__ import annotations

import argparse
import sys

from .. import jsonio
from .runtime import ContextRuntime


def _runtime(args) -> ContextRuntime:
    if args.config:
        rt = ContextRuntime.from_config(args.config)
    else:
        rt = ContextRuntime.default([])
    if args.corpus:
        rt.index(args.corpus)
    return rt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="contextos", description="Query planner for LLM context.")
    p.add_argument("--config", help="path to contextos.yaml")
    p.add_argument("--corpus", help="folder to index before the command")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "explain", "simulate"):
        sp = sub.add_parser(name)
        sp.add_argument("goal")
    sub.choices["explain"].add_argument("--analyze", action="store_true")
    args = p.parse_args(argv)

    rt = _runtime(args)
    if args.cmd == "run":
        res = rt.run(args.goal)
        print(res.answer)
        print(f"\n— cost ${res.cost_usd:.4f} · {len(res.citations)} citations · plan {res.plan.id}")
    elif args.cmd == "explain":
        print(jsonio.dumps(rt.explain(args.goal, analyze=args.analyze), indent=2))
    elif args.cmd == "simulate":
        print(jsonio.dumps(rt.simulate(args.goal), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
