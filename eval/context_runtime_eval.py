#!/usr/bin/env python3
"""Context Runtime RAG evaluation probe.

This script probes multiple backends for retrieval quality on a fixed
suite of Russian-language patient lab questions. It relies solely on the
Python standard library so it can execute in constrained environments.

For each (question, backend) pair, we POST to `/librechat/retrieve`
expecting a JSON response with the following shape::

    {
        "strategy": "...",
        "hits": [
            {"chunk_id": "...", "filename": "...", "score": 0.0, "text": "..."},
            ...
        ],
        "context": "...",
        "suggestion": "..."
    }

The script applies simple keyword heuristics to judge the retrieval:
PASS, WEAK, or MISS. It never raises on network failures — instead
marking the judgment as MISS with an explanatory note — enabling its use
as a lightweight monitoring probe.
"""

from __future__ import annotations

import json
import sys
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Question configuration and heuristics
# ---------------------------------------------------------------------------

QuestionConfig = Dict[str, object]

QUESTIONS: Sequence[QuestionConfig] = (
    {
        "id": "steroid_profile",
        "query": "стероидный профиль тестостерон кортизол",
        "keywords": (
            "стероид",
            "стероидный",
            "тестостерон",
            "кортизол",
            "андроген",
            "гормон",
        ),
        "required": 2,
    },
    {
        "id": "lipid_profile",
        "query": "липидный профиль холестерин ЛПНП",
        "keywords": (
            "липид",
            "липидный",
            "холестерин",
            "лпнп",
            "ldl",
            "липопротеин",
            "триглицерид",
        ),
        "required": 2,
    },
    {
        "id": "gonadotropins",
        "query": "ФСГ ЛГ пролактин гормоны",
        "keywords": (
            "фсг",
            "фолликул",
            "лг",
            "лютеинизир",
            "пролактин",
            "гормон",
        ),
        "required": 2,
    },
    {
        "id": "cbc_results",
        "query": "результаты анализа крови",
        "keywords": (
            "общий анализ",
            "анализ крови",
            "клинический анализ",
            "оак",
            "гемоглобин",
            "эритроцит",
        ),
        "required": 1,
    },
    {
        "id": "vitamin_d",
        "query": "витамин Д дефицит",
        "keywords": (
            "витамин d",
            "витамин д",
            "25(oh)d",
            "25-oh",
            "кальциферол",
            "дефицит",
            "недостат",
        ),
        "required": 1,
    },
    {
        "id": "tsh",
        "query": "ТТГ щитовидная железа",
        "keywords": (
            "ттг",
            "tsh",
            "тиреотроп",
            "щитовид",
            "тироксин",
            "гипотиреоз",
        ),
        "required": 2,
    },
    {
        "id": "glucose",
        "query": "глюкоза сахар крови",
        "keywords": (
            "глюкоз",
            "сахар",
            "гликем",
            "гликирован",
            "диабет",
            "глюкоза",
        ),
        "required": 2,
    },
    {
        "id": "ferritin",
        "query": "ферритин железо",
        "keywords": (
            "ферритин",
            "железо",
            "анемия",
            "сидеропен",
            "iron",
        ),
        "required": 2,
    },
)

BACKENDS: Sequence[Tuple[str, str]] = (
    ("python", "http://localhost:8092"),
    ("go", "http://localhost:8093"),
)

VERDICTS = ("PASS", "WEAK", "MISS")

# ---------------------------------------------------------------------------
# Judging logic (exportable)
# ---------------------------------------------------------------------------


def _collect_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    """Return a sorted list of keywords present in *text*."""

    lowered = text.lower()
    matches = sorted({kw for kw in keywords if kw in lowered})
    return matches


def judge_retrieval(question: QuestionConfig, hits: Sequence[dict], context: str) -> Tuple[str, str]:
    """Return (verdict, detail) based on heuristic keyword coverage.

    PASS  – A hit covers the required number of keywords and the context
             contains at least one relevant keyword.
    WEAK  – Some signal (hit or context) mentions a keyword, but coverage
             or context sufficiency is lacking.
    MISS  – No relevant keywords appear anywhere.
    """

    keywords: Sequence[str] = question["keywords"]  # type: ignore[assignment]
    required: int = int(question.get("required", 2))

    best_hit_keywords: List[str] = []
    best_hit_ref = None
    for hit in hits:
        source = "{} {}".format(hit.get("filename", ""), hit.get("text", ""))
        hit_keywords = _collect_keywords(source, keywords)
        if len(hit_keywords) > len(best_hit_keywords):
            best_hit_keywords = hit_keywords
            best_hit_ref = hit

    context_text = context or ""
    context_keywords = _collect_keywords(context_text, keywords)

    if best_hit_keywords and len(best_hit_keywords) >= required and context_keywords:
        verdict = "PASS"
    elif best_hit_keywords or context_keywords:
        verdict = "WEAK"
    else:
        verdict = "MISS"

    detail_parts: List[str] = []
    if best_hit_ref is not None:
        chunk_id = best_hit_ref.get("chunk_id", "?")
        filename = best_hit_ref.get("filename", "?")
        detail_parts.append(
            "hit={}@{}".format(chunk_id, filename)
        )
    if best_hit_keywords:
        detail_parts.append("hit_kw={}".format(",".join(best_hit_keywords)))
    if context_keywords:
        detail_parts.append("ctx_kw={}".format(",".join(context_keywords)))
    if not detail_parts:
        detail_parts.append("no keywords")

    detail = "; ".join(detail_parts)
    return verdict, detail


# ---------------------------------------------------------------------------
# Probe execution
# ---------------------------------------------------------------------------


def _shorten(text: str, limit: int = 48) -> str:
    """Shorten *text* for display without breaking words too harshly."""

    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    with urlopen(request, timeout=timeout) as response:  # nosec - stdlib only
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def probe_backend(name: str, base_url: str) -> Dict[str, int]:
    """Probe *base_url* for all questions, returning verdict counters."""

    counters: Dict[str, int] = {verdict: 0 for verdict in VERDICTS}

    for question in QUESTIONS:
        query = question["query"]  # type: ignore[assignment]
        display_query = _shorten(str(query))
        endpoint = base_url.rstrip("/") + "/librechat/retrieve"

        verdict: str
        detail: str
        try:
            response = _post_json(endpoint, {"request": query})
            hits = response.get("hits") or []
            context_text = response.get("context") or ""
            if not isinstance(hits, list):
                raise ValueError("hits must be a list")
            verdict, detail = judge_retrieval(question, hits, context_text)
        except (URLError, HTTPError) as exc:
            verdict = "MISS"
            detail = f"error={exc}"[:200]
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            verdict = "MISS"
            detail = f"bad_response={exc}"[:200]
        except Exception as exc:  # pragma: no cover - defensive
            verdict = "MISS"
            detail = f"unexpected_error={exc}"[:200]

        counters[verdict] = counters.get(verdict, 0) + 1
        print(f"{name:<6} | {display_query:<48} | {verdict:<4} | {detail}")

    return counters


def main() -> int:
    print("backend | question                                      | verdict | detail")
    print("-" * 80)

    summaries: List[Tuple[str, Dict[str, int]]] = []
    for backend_name, base_url in BACKENDS:
        counters = probe_backend(backend_name, base_url)
        summaries.append((backend_name, counters))

    print("-" * 80)
    for backend_name, counters in summaries:
        summary = " ".join(
            f"{verdict}={counters.get(verdict, 0)}" for verdict in VERDICTS
        )
        print(f"summary {backend_name:<6}: {summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
