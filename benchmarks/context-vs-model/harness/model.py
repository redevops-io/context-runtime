"""Model client — drives any OpenAI-compatible endpoint (here: a CPU ``llama-server``)
through Context Runtime's own stdlib client, so every model under test speaks the same
``ModelRequest → ModelResult`` contract."""
from __future__ import annotations

import time

from context_runtime.adapters.model_litellm import Tier
from context_runtime.adapters.model_openai import OpenAICompatibleModel
from context_runtime.types import ModelRequest

_ANSWER_SYS = (
    "You are a financial analyst answering a question from SEC 10-K/10-Q filing excerpts. "
    "Use ONLY the provided CONTEXT. Give a single direct answer — the exact figure (with "
    "units) or a one-sentence fact. Do not show reasoning. If the answer is not in the "
    "context, reply exactly: NOT FOUND."
)


def make_client(base_url: str, model: str, *, api_key: str = "sk-noauth",
                timeout: float = 180.0) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        [Tier(name="chat", model=model, base_url=base_url.rstrip("/"),
              api_key=api_key, cost_per_1k=0.0)],
        timeout=timeout,
    )


def answer(client: OpenAICompatibleModel, question, context: str, *,
           max_tokens: int = 384) -> dict:
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question.question}\n\nANSWER:"
    req = ModelRequest(messages=({"role": "user", "content": user},),
                       system=_ANSWER_SYS, max_tokens=max_tokens, capability="draft")
    t0 = time.time()
    res = client.complete(req)
    dt = time.time() - t0
    return {"text": (res.text or "").strip(), "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "latency_s": dt}


def make_chat(client: OpenAICompatibleModel, *, max_tokens: int = 8):
    """Adapt a client into a ``chat(system, user) -> str`` (used for the grader judge)."""
    def chat(system: str, user: str) -> str:
        req = ModelRequest(messages=({"role": "user", "content": user},),
                           system=system, max_tokens=max_tokens, capability="draft")
        return client.complete(req).text or ""
    return chat
