"""Benchmark A — independent mission nodes: serial vs bounded-parallel wave drain.

Quantifies the latency recoverable by parallelizing the Mission Runtime's ready-node wave (audit candidate #1)
*before* touching production code. It models exactly the two drains from the audit:

  * SERIAL   — the current `for node in ready: execute(node)` loop (runtime.py:342 / Go runtime.go:243):
               nodes in a ready wave run one blocking call after another.
  * PARALLEL — the proposed fix: dispatch the ready wave concurrently, bounded by max_concurrency, with a
               join barrier before the next wave. Same DAG, same dependency order, same outputs.

Each node is a stub "operator" that sleeps a fixed latency (standing in for provider/model latency — the
runtime's own CPU per node is negligible, which is the whole point). The benchmark asserts the two drains
produce byte-identical results, then reports the wall-clock speedup — a lower bound on what candidate #1 buys,
and a regression guard once the executor goes concurrent. No network, no API cost.

    python benchmarks/parallelization/bench_independent_nodes.py
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Node:
    id: str
    latency_ms: int
    depends_on: frozenset[str] = field(default_factory=frozenset)


def fanout_dag(width: int, worker_ms: int = 200, root_ms: int = 50, join_ms: int = 50) -> list[Node]:
    """root → [width independent workers] → join — the shape of the real templates (product_launch etc.)."""
    root = Node("root", root_ms)
    workers = [Node(f"w{i}", worker_ms, frozenset({"root"})) for i in range(width)]
    join = Node("join", join_ms, frozenset(w.id for w in workers))
    return [root, *workers, join]


def _execute(node: Node) -> str:
    time.sleep(node.latency_ms / 1000.0)          # stand-in for provider/model latency
    return f"{node.id}:done"


def _ready_waves(nodes: list[Node]):
    """Yield successive ready frontiers (Kahn) — the scheduler's wave computation, shared by both drains."""
    done: set[str] = set()
    remaining = {n.id: n for n in nodes}
    while remaining:
        wave = [n for n in remaining.values() if n.depends_on <= done]
        if not wave:
            raise RuntimeError("cycle or unsatisfiable deps")
        yield wave
        for n in wave:
            done.add(n.id)
            del remaining[n.id]


def run_serial(nodes: list[Node]) -> dict[str, str]:
    """The current runtime: drain each ready wave one blocking node at a time."""
    out: dict[str, str] = {}
    for wave in _ready_waves(nodes):
        for node in wave:                         # ← serial drain (runtime.py:342)
            out[node.id] = _execute(node)
    return out


def run_parallel(nodes: list[Node], max_concurrency: int) -> dict[str, str]:
    """The proposed fix: dispatch each ready wave concurrently, bounded, with a join barrier per wave."""
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for wave in _ready_waves(nodes):
            for node, result in zip(wave, pool.map(_execute, wave)):   # bounded fan-out + barrier
                out[node.id] = result
    return out


def _time(fn) -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = fn()
    return (time.perf_counter() - t0) * 1000.0, r


def run() -> None:
    width = 8
    dag = fanout_dag(width, worker_ms=200)
    serial_ms, serial_out = _time(lambda: run_serial(dag))

    print(f"Benchmark A — fan-out DAG: root(50ms) → {width}×worker(200ms) → join(50ms)\n")
    print(f"  {'drain':<26}{'wall_ms':>10}{'speedup':>10}   correct")
    print(f"  {'serial (current)':<26}{serial_ms:>10.0f}{'1.00×':>10}   —")
    for mc in (1, 2, 4, 8):
        ms, out = _time(lambda: run_parallel(dag, mc))
        correct = out == serial_out              # identical results — parallelism changes latency, not answers
        print(f"  {'parallel max='+str(mc):<26}{ms:>10.0f}{serial_ms/ms:>9.2f}×   {correct}")

    ideal = 50 + 200 + 50                          # longest dependency chain — the unavoidable floor
    print(f"\n  unavoidable floor (longest dependency chain root→worker→join): {ideal} ms")
    print(f"  recoverable latency at max=8: {serial_ms - ideal:.0f} ms of {serial_ms:.0f} ms "
          f"({(serial_ms-ideal)/serial_ms*100:.0f}%) is serial-by-implementation, not dependency-bound.")


if __name__ == "__main__":
    run()
