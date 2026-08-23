"""Reasoner fork/join parallelization (LLM-parallelization audit, P0) — opt-in, order-preserving,
result-identical. Default (CR_REASONER_CONCURRENCY unset) is the historical serial path."""
from __future__ import annotations

import threading
import time

from context_runtime.reasoner._concurrency import fanout
from context_runtime.reasoner.strategies import DebateReasoner
from context_runtime.types import ModelRequest, ModelResult, ReasonRequest  # noqa: F401


# ── the fanout primitive (used by self-consistency / debate / plan-workers) ──

def test_fanout_serial_by_default(monkeypatch):
    monkeypatch.delenv("CR_REASONER_CONCURRENCY", raising=False)
    assert fanout(lambda x: x * x, [1, 2, 3, 4]) == [1, 4, 9, 16]


def test_fanout_parallel_preserves_order_and_overlaps(monkeypatch):
    monkeypatch.setenv("CR_REASONER_CONCURRENCY", "4")

    def slow(x):
        time.sleep(0.05)
        return x * x

    t0 = time.perf_counter()
    out = fanout(slow, [1, 2, 3, 4])
    wall = time.perf_counter() - t0
    assert out == [1, 4, 9, 16]            # order preserved despite concurrency
    assert wall < 0.15                     # 4×0.05=0.2s serial → overlapped to ~0.05s


# ── a real reasoner: concurrent debate == serial debate, but faster ──

class _DetSleepingModel:
    """Deterministic per (system, prompt) reply + a fixed delay — so the final answer is identical whether
    the debaters ran serially or concurrently, and the speedup is observable."""

    def __init__(self, delay: float):
        self.delay = delay
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, req):
        time.sleep(self.delay)
        with self._lock:
            self.calls += 1
        key = f"{getattr(req, 'system', '') or ''}|{getattr(req, 'prompt', '')}"
        return ModelResult(text=f"ans:{abs(hash(key)) % 100000}", model="m", tier="cheap",
                           prompt_tokens=10, completion_tokens=5, est_cost_usd=0.001, models_used=("m",))

    def capabilities(self, model):
        from context_runtime.types import ModelCapabilities
        return ModelCapabilities()

    def count_tokens(self, text, model):
        return len(text) // 4

    def info(self):
        from context_runtime.types import PluginInfo
        return PluginInfo(name="det", kind="model")


def _ctx(text="CTX", question="the question"):
    from types import SimpleNamespace
    return SimpleNamespace(plan=SimpleNamespace(id="p1", intent=SimpleNamespace(normalized=question)),
                           assembled_text=text)


def _run_debate(concurrency: int, monkeypatch, delay=0.05):
    if concurrency <= 1:
        monkeypatch.delenv("CR_REASONER_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("CR_REASONER_CONCURRENCY", str(concurrency))
    model = _DetSleepingModel(delay)
    out = DebateReasoner(model, rounds=4).reason(ReasonRequest(context=_ctx(), capability="synthesis"))
    return out, model.calls


def test_debate_concurrent_equals_serial(monkeypatch):
    serial, calls_s = _run_debate(1, monkeypatch)
    parallel, calls_p = _run_debate(4, monkeypatch)
    assert serial.text == parallel.text          # identical final answer (order preserved → same judge input)
    assert calls_s == calls_p == 5               # 4 debaters + 1 judge, both paths


def test_debate_concurrent_is_faster(monkeypatch):
    _, _ = _run_debate(1, monkeypatch)           # warm
    t0 = time.perf_counter(); _run_debate(1, monkeypatch); serial = time.perf_counter() - t0
    t0 = time.perf_counter(); _run_debate(4, monkeypatch); parallel = time.perf_counter() - t0
    # 4 debaters overlap (judge stays serial): serial ≈ 5δ, parallel ≈ 2δ
    assert parallel < serial * 0.7, f"serial={serial:.2f}s parallel={parallel:.2f}s"
