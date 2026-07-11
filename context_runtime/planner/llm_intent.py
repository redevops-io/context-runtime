"""LLM-backed intent analysis for representation routing — the hybrid, confidence-gated head.

The v4 planner (``classify → constrain → learn``) is validated, but its *shipped* head — the keyword
``RuleIntentAnalyzer`` — is the bottleneck: it scores well on EXPLICIT cues yet ~0.00 on IMPLICIT
multi-hop / temporal intent (a question like "Who is the spouse of the Green performer?" carries no
"multi-hop"/"related to" string, so it defaults to ``document`` and the graph engine is never routed to).

This module adds an LLM head, **gated by confidence** so it stays off the hot path: run the cheap
keyword analyzer first; only fall to the LLM when the keyword head is *unsure* — i.e. it defaulted to
``document`` (or an ``unknown``/``conceptual`` bucket) or reported low confidence. That is exactly the
implicit-intent case. Explicit cues ("as of", "how many", "related to", …) are handled for free by the
keyword head, so ~all cost stays on the cheap path. This reuses the same confidence seam that
``planner/candidates.py`` already keys its widening on.

``OpenAICompatModel`` is a stdlib OpenAI-compatible chat client (no SDK dep); point it at any served
model (a vLLM NVFP4 endpoint, a hosted API, …).
"""
from __future__ import annotations

import json
import urllib.request

from . import representations
from .intent import RuleIntentAnalyzer
from ..types import Intent

_SYS = (
    "You are a retrieval router. Read the question and reply with EXACTLY ONE word naming the "
    "knowledge representation needed to answer it:\n"
    "  document   — a single fact lookup answerable from one passage\n"
    "  graph      — needs multi-hop reasoning chaining several linked entities/facts\n"
    "  temporal   — needs time-ordered or point-in-time facts from a history/conversation\n"
    "  analytical — needs a STRUCTURED data store with a known schema/tables (SQL, MongoDB or "
    "Elasticsearch): a lookup, filter, join or aggregate — e.g. counts/rankings, 'records where …', "
    "'from the <table>'\n"
    "  code       — needs reasoning over source code / a codebase\n"
    "  multimodal — needs an image/screenshot/diagram\n"
    "Reply with only one of: document, graph, temporal, analytical, code, multimodal.")

_VALID = ("temporal", "graph", "analytical", "code", "multimodal", "document")

# When the LLM corrects the representation, re-align the intent bucket so BUCKET_TIERS / strategy /
# verify follow it (candidates.py already constrains the *method* by representation; this fixes the
# *model tier* selection too). Only representations with a dedicated bucket are mapped.
_REP_TO_BUCKET = {"graph": "multi_hop", "temporal": "temporal", "code": "code_reasoning"}


class OpenAICompatModel:
    """Minimal stdlib OpenAI-compatible chat client for the intent head (no SDK dependency)."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "sk-noauth",
                 timeout: float = 60.0, disable_thinking: bool = True):
        self.base = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.disable_thinking = disable_thinking

    def classify(self, text: str) -> str | None:
        payload = {"model": self.model, "temperature": 0, "max_tokens": 6,
                   "messages": [{"role": "system", "content": _SYS},
                                {"role": "user", "content": text}]}
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        req = urllib.request.Request(self.base + "/chat/completions",
              data=json.dumps(payload).encode(),
              headers={"Content-Type": "application/json",
                       "Authorization": f"Bearer {self.api_key}"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.load(r)
            out = ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""
        except Exception:  # noqa: BLE001 — network/parse failure ⇒ defer to the keyword head
            return None
        out = out.strip().lower()
        for rep in _VALID:
            if rep in out:
                return rep
        return None


class HybridIntentAnalyzer:
    """Keyword-first, LLM-on-doubt intent analyzer.

    ``analyze(goal) -> Intent`` — drop-in for ``RuleIntentAnalyzer`` in ``ContextRuntime(intent=…)``.
    The LLM head fires only when the keyword head is unsure (defaulted to ``document`` / uncertain
    bucket / confidence below ``confidence_floor``); otherwise the cheap keyword verdict stands.
    When the LLM overrides the representation, confidence is raised to ``override_confidence`` so the
    candidate generator constrains to the new representation instead of widening.
    """

    def __init__(self, model: OpenAICompatModel, *, base=None,
                 confidence_floor: float = 0.6, override_confidence: float = 0.85,
                 unsure_buckets: tuple[str, ...] = ("unknown", "conceptual")):
        self.model = model
        self.base = base or RuleIntentAnalyzer()
        self.confidence_floor = confidence_floor
        self.override_confidence = override_confidence
        self.unsure_buckets = unsure_buckets
        # routing-cost ledger: `escalated` = model calls (the per-query intent-routing overhead to
        # weigh against a model's built-in context management), `overrides` = calls that changed the route.
        self.stats = {"analyzed": 0, "escalated": 0, "overrides": 0}

    def _unsure(self, intent: Intent) -> bool:
        # Trust a confident positive representation (graph/temporal/analytical/…) for FREE — spend a
        # model call only on the blind-spot default `document`, or a genuinely low-confidence verdict.
        if intent.representation != "document":
            return intent.confidence < self.confidence_floor
        return intent.bucket in self.unsure_buckets or intent.confidence < self.confidence_floor

    def analyze(self, goal) -> Intent:
        self.stats["analyzed"] += 1
        intent = self.base.analyze(goal)
        if not self._unsure(intent):
            return intent                      # explicit cue → trust the free keyword head
        self.stats["escalated"] += 1           # a model call — the intent-routing cost line
        rep = self.model.classify(goal.text)
        if rep is None or rep == intent.representation:
            return intent                      # LLM unavailable or agrees → keep keyword verdict
        self.stats["overrides"] += 1
        bucket = _REP_TO_BUCKET.get(rep, intent.bucket)   # re-align tiers/strategy to the new representation
        return Intent(bucket=bucket, entities=intent.entities, risk=intent.risk,
                      normalized=intent.normalized,
                      confidence=max(intent.confidence, self.override_confidence),
                      representation=rep)
