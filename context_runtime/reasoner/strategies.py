"""Reasoning strategies beyond single-shot (SPEC §4.4) — the seam the planner always routed to but
never had an implementation for.

Each is a ``ReasonerPlugin`` over ≥1 ``ModelPlugin``. They compose multiple governed model calls and
**roll the sub-call costs up** into one ``ModelResult`` (``models_used`` + summed tokens/cost), so the
cost model prices a multi-shot strategy correctly and EXPLAIN shows every model that ran. Selection is
per-intent planner policy (``rules.BUCKET_DEFAULTS``) and, once the online optimizer is on, a bandit
arm — the runtime dispatches on the plan's chosen strategy via ``reasoner_for``.

Strategies (all provider-neutral — they call the model plane, not any provider):
  • plan_worker_critic — decompose → answer each sub-question → synthesize + self-critique.
  • debate           — two independent answers → a judge picks/merges the stronger.
  • tool_loop        — a bounded agentic loop; the model may call injected tools before answering.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from ..plugins.base import ModelPlugin
from ..types import ModelRequest, ModelResult, PluginInfo, ReasonRequest
from .single_shot import SingleShotReasoner, _SYSTEM as _ANSWER_SYSTEM


def _question(ctx) -> str:
    return ctx.plan.intent.normalized or ctx.plan.id


def _rollup(calls: list[ModelResult], final_text: str) -> ModelResult:
    """Fold N sub-call results into one result carrying the final answer + total cost/tokens."""
    if not calls:
        return ModelResult(text=final_text, model="none", tier="none")
    ptoks = sum(r.prompt_tokens for r in calls)
    ctoks = sum(r.completion_tokens for r in calls)
    cost = round(sum(r.est_cost_usd for r in calls), 6)
    models: tuple[str, ...] = tuple(m for r in calls for m in (r.models_used or (r.model,)))
    last = calls[-1]
    return ModelResult(text=final_text, model=last.model, tier=last.tier,
                       prompt_tokens=ptoks, completion_tokens=ctoks, est_cost_usd=cost,
                       models_used=models)


def _ask(model: ModelPlugin, prompt: str, system: str, capability: str, max_tokens: int = 1024) -> ModelResult:
    return model.complete(ModelRequest(messages=({"role": "user", "content": prompt},),
                                       system=system, capability=capability, max_tokens=max_tokens))


# ──────────────────────────── plan → worker → critic ────────────────────────────
_PLAN_SYSTEM = ("Break the question into 2-4 focused sub-questions that, answered together, resolve it. "
                "One sub-question per line, no numbering, no prose.")
_CRITIC_SYSTEM = ("Synthesize a single, correct answer from the sub-answers and the context. "
                  "Cite sources inline like [1]. Silently drop any sub-answer that the context does "
                  "not support — do not invent facts.")


class PlanWorkerCriticReasoner:
    """Decompose the question, answer each part against the context, then synthesize + self-critique."""

    def __init__(self, model: ModelPlugin, *, max_subquestions: int = 3):
        self.model = model
        self.max_subquestions = max_subquestions

    def reason(self, req: ReasonRequest) -> ModelResult:
        ctx = req.context
        q = _question(ctx)
        calls: list[ModelResult] = []

        plan = _ask(self.model, f"Question: {q}", _PLAN_SYSTEM, req.capability, max_tokens=256)
        calls.append(plan)
        subqs = [ln.strip("-• \t") for ln in plan.text.splitlines() if ln.strip()][: self.max_subquestions]
        if not subqs:
            subqs = [q]

        sub_answers = []
        for sub in subqs:
            prompt = f"Context:\n{ctx.assembled_text}\n\nSub-question: {sub}"
            w = _ask(self.model, prompt, _ANSWER_SYSTEM, req.capability, max_tokens=512)
            calls.append(w)
            sub_answers.append(f"Q: {sub}\nA: {w.text}")

        synth_prompt = (f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}\n\n"
                        f"Sub-answers:\n" + "\n\n".join(sub_answers))
        critic = _ask(self.model, synth_prompt, _CRITIC_SYSTEM, req.capability, max_tokens=1024)
        calls.append(critic)
        return _rollup(calls, critic.text)

    def info(self) -> PluginInfo:
        return PluginInfo(name="plan_worker_critic", kind="reasoner",
                          capabilities=frozenset({"plan_worker_critic"}))


# ──────────────────────────── debate → judge ────────────────────────────
_DEBATER_SYSTEMS = (
    _ANSWER_SYSTEM + " Argue for the most defensible answer you can support from the context.",
    _ANSWER_SYSTEM + " Independently answer; where the context is ambiguous, prefer the more "
                     "conservative reading and note the uncertainty.",
)
_JUDGE_SYSTEM = ("Two independent answers to the same question are given with the context. Produce the "
                 "single best final answer: keep what the context supports, resolve disagreements in "
                 "favor of the citation-backed claim, and cite sources inline like [1].")


class DebateReasoner:
    """Two independent answers, then a judge merges the stronger, citation-backed one."""

    def __init__(self, model: ModelPlugin, *, rounds: int = 2):
        self.model = model
        self.rounds = max(2, rounds)

    def reason(self, req: ReasonRequest) -> ModelResult:
        ctx = req.context
        q = _question(ctx)
        prompt = f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}"
        calls: list[ModelResult] = []
        answers = []
        for i in range(self.rounds):
            system = _DEBATER_SYSTEMS[i % len(_DEBATER_SYSTEMS)]
            r = _ask(self.model, prompt, system, req.capability, max_tokens=768)
            calls.append(r)
            answers.append(f"Answer {i + 1}: {r.text}")
        judge_prompt = f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}\n\n" + "\n\n".join(answers)
        judge = _ask(self.model, judge_prompt, _JUDGE_SYSTEM, req.capability, max_tokens=1024)
        calls.append(judge)
        return _rollup(calls, judge.text)

    def info(self) -> PluginInfo:
        return PluginInfo(name="debate", kind="reasoner", capabilities=frozenset({"debate"}))


# ──────────────────────────── bounded tool loop ────────────────────────────
# Convention (kept text-based so it works over any ModelPlugin without extending ModelResult):
#   the model emits either  ACTION: <tool_name> {"json": "args"}   to call a tool, or
#                           FINAL: <answer>                        to stop.
_ACTION_RE = re.compile(r"^\s*ACTION:\s*(\S+)\s*(\{.*\})?\s*$", re.I | re.M)
_FINAL_RE = re.compile(r"FINAL:\s*(.+)\s*$", re.I | re.S)
_TOOL_SYSTEM = (
    "You may call tools to gather facts before answering. To call a tool, output exactly:\n"
    "ACTION: <tool_name> {\"arg\": \"value\"}\n"
    "When you have enough information, output exactly:\nFINAL: <your answer with inline [1] citations>\n"
    "Answer only from the context and tool observations; do not invent facts."
)


class ToolLoopReasoner:
    """A bounded agentic loop. ``tool_runner(name, args) -> str`` executes an injected tool; without
    one (or when the model answers directly) it degrades to a single grounded answer — so it is safe
    as a default and only *adds* capability when tools are wired."""

    def __init__(self, model: ModelPlugin, *, tool_runner: Callable[[str, dict], str] | None = None,
                 tools: tuple[dict, ...] | None = None, max_iters: int = 4):
        self.model = model
        self.tool_runner = tool_runner
        self.tools = tools
        self.max_iters = max_iters

    def reason(self, req: ReasonRequest) -> ModelResult:
        ctx = req.context
        q = _question(ctx)
        transcript = f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}"
        calls: list[ModelResult] = []
        for _ in range(self.max_iters):
            r = self.model.complete(ModelRequest(
                messages=({"role": "user", "content": transcript},),
                system=_TOOL_SYSTEM, capability=req.capability, tools=self.tools, max_tokens=1024))
            calls.append(r)
            final = _FINAL_RE.search(r.text)
            if final:
                return _rollup(calls, final.group(1).strip())
            action = _ACTION_RE.search(r.text)
            if not action or self.tool_runner is None:
                # no tool requested (or none available) → this reply is the answer
                return _rollup(calls, r.text.strip())
            name = action.group(1)
            try:
                args = json.loads(action.group(2)) if action.group(2) else {}
            except json.JSONDecodeError:
                args = {}
            try:
                obs = self.tool_runner(name, args)
            except Exception as e:  # noqa: BLE001 — a failing tool is an observation, not a crash
                obs = f"tool error: {e}"
            transcript += f"\n\nACTION: {name} {action.group(2) or '{}'}\nOBSERVATION: {obs}"
        # ran out of iterations → one final grounded answer
        last = self.model.complete(ModelRequest(
            messages=({"role": "user", "content": transcript + "\n\nAnswer now."},),
            system=_ANSWER_SYSTEM, capability=req.capability, max_tokens=1024))
        calls.append(last)
        return _rollup(calls, last.text.strip())

    def info(self) -> PluginInfo:
        return PluginInfo(name="tool_loop", kind="reasoner", capabilities=frozenset({"tool_loop"}))


# ──────────────────────────── factory ────────────────────────────
def reasoner_for(strategy: str, model: ModelPlugin, **kw):
    """Map a plan's chosen reasoning strategy to a ReasonerPlugin. Unknown → single_shot (safe default)."""
    if strategy == "plan_worker_critic":
        return PlanWorkerCriticReasoner(model, **kw)
    if strategy == "debate":
        return DebateReasoner(model, **kw)
    if strategy == "tool_loop":
        return ToolLoopReasoner(model, **kw)
    return SingleShotReasoner(model)
