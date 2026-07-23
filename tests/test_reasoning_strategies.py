"""Reasoning strategies: multi-call composition, cost roll-up, and the bounded tool loop.

Uses a scripted ModelPlugin (returns a fixed queue of replies, counts calls) and a lightweight
ReasonRequest context — the reasoners only read ctx.plan.intent.normalized / ctx.plan.id /
ctx.assembled_text.
"""
from types import SimpleNamespace

from context_runtime.plugins import base
from context_runtime.reasoner.strategies import (
    DebateReasoner,
    PlanWorkerCriticReasoner,
    ToolLoopReasoner,
    reasoner_for,
)
from context_runtime.reasoner.single_shot import SingleShotReasoner
from context_runtime.types import ModelResult, ReasonRequest


class ScriptedModel:
    """Returns replies from a queue (cycling the last), 10 prompt / 5 completion tokens, $0.001 each."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, req):
        self.calls.append(req)
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return ModelResult(text=text, model="m", tier="cheap",
                           prompt_tokens=10, completion_tokens=5, est_cost_usd=0.001,
                           models_used=("m",))

    def capabilities(self, model):
        from context_runtime.types import ModelCapabilities
        return ModelCapabilities()

    def count_tokens(self, text, model):
        return len(text) // 4

    def info(self):
        from context_runtime.types import PluginInfo
        return PluginInfo(name="scripted", kind="model")


def _ctx(text="CTX", question="the question"):
    plan = SimpleNamespace(id="p1", intent=SimpleNamespace(normalized=question))
    return SimpleNamespace(plan=plan, assembled_text=text)


def _req():
    return ReasonRequest(context=_ctx(), capability="synthesis")


def test_all_strategies_satisfy_reasoner_protocol():
    m = ScriptedModel(["x"])
    for r in (SingleShotReasoner(m), PlanWorkerCriticReasoner(m), DebateReasoner(m), ToolLoopReasoner(m)):
        assert isinstance(r, base.ReasonerPlugin)


def test_factory_dispatch():
    m = ScriptedModel(["x"])
    assert isinstance(reasoner_for("plan_worker_critic", m), PlanWorkerCriticReasoner)
    assert isinstance(reasoner_for("debate", m), DebateReasoner)
    assert isinstance(reasoner_for("tool_loop", m), ToolLoopReasoner)
    assert isinstance(reasoner_for("single_shot", m), SingleShotReasoner)
    assert isinstance(reasoner_for("bogus", m), SingleShotReasoner)  # unknown → safe default


def test_plan_worker_critic_composes_and_rolls_up_cost():
    # plan reply yields 2 sub-questions → 1 plan + 2 workers + 1 critic = 4 calls
    m = ScriptedModel(["sub one\nsub two", "worker-ans", "worker-ans", "FINAL SYNTHESIS"])
    res = PlanWorkerCriticReasoner(m, max_subquestions=3).reason(_req())
    assert len(m.calls) == 4
    assert res.text == "FINAL SYNTHESIS"
    assert res.prompt_tokens == 40 and res.completion_tokens == 20      # summed across 4 calls
    assert res.est_cost_usd == 0.004
    assert len(res.models_used) == 4                                    # every sub-call rolled up


def test_plan_worker_critic_degrades_when_no_subquestions():
    m = ScriptedModel(["", "answer", "SYNTH"])   # empty plan → fall back to the original question
    res = PlanWorkerCriticReasoner(m).reason(_req())
    assert len(m.calls) == 3                       # plan + 1 worker (the question) + critic
    assert res.text == "SYNTH"


def test_debate_runs_two_debaters_then_judge():
    m = ScriptedModel(["ans A", "ans B", "JUDGED FINAL"])
    res = DebateReasoner(m, rounds=2).reason(_req())
    assert len(m.calls) == 3
    assert res.text == "JUDGED FINAL"
    assert len(res.models_used) == 3


def test_tool_loop_calls_tool_then_finalizes():
    m = ScriptedModel(['ACTION: search {"q": "x"}', "FINAL: the answer [1]"])
    seen = []
    def runner(name, args):
        seen.append((name, args))
        return "observation text"
    res = ToolLoopReasoner(m, tool_runner=runner).reason(_req())
    assert seen == [("search", {"q": "x"})]
    assert res.text == "the answer [1]"
    assert len(m.calls) == 2


def test_tool_loop_without_runner_degrades_to_single_answer():
    m = ScriptedModel(["just an answer, no tool"])
    res = ToolLoopReasoner(m).reason(_req())
    assert res.text == "just an answer, no tool"
    assert len(m.calls) == 1


def test_runtime_dispatches_strategy_from_plan():
    # multi_hop now routes to plan_worker_critic → execute() must make multiple model calls
    from context_runtime import ContextRuntime
    from context_runtime.adapters.store_inmemory import InMemoryStore
    m = ScriptedModel(["sub a\nsub b", "wa", "wb", "final answer"])
    rt = ContextRuntime(models={t: m for t in ("local", "cheap", "premium")},
                        retriever=InMemoryStore([{"chunk_id": "d1", "filename": "d1",
                                                  "text": "a relates to b", "created_at": None}]))
    res = rt.run("How is A connected to B across the system?")   # → multi_hop bucket
    assert res.answer == "final answer"
    assert len(m.calls) >= 3       # decomposed, not a single shot
