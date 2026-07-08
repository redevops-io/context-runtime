"""Job-lead outreach — turn AI-engineering job listings into tailored pitches (Market Radar lead-gen).

A company hiring an *AI Data Engineer / AI Developer* internally is signalling real, unmet AI-adoption
pain — exactly who an execution-layer runtime helps. This module:

  • **sources** such listings (via a pluggable search — Market Radar's web_search MCP, or any callable);
  • **filters** to genuine internal-build roles, dropping consultancies / forward-deployed / agency /
    staffing shops (they build AI *for* clients, not for themselves — no internal pain to solve);
  • **writes** a listing-tailored pitch from an EDITABLE template (agent-editable, shared by all users);
  • **dedupes** so a listing that reappears months later is never pitched twice;
  • **drafts** distribution (default: an outbox draft — sending is an approval-gated deployment step).

Dependency-light; the LLM tailoring and live search are optional/pluggable, so the logic is fully
testable offline.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── signals ───────────────────────────────────────────────────────────────────────────────────────
# A company hiring for these builds AI in-house → a lead.
_AI_ROLE = re.compile(
    r"\b(ai|ml|genai|llm)\b.*\b(engineer|developer|scientist|architect)\b"
    r"|\b(machine learning|applied ai|ai platform|ai data|prompt|rag|retrieval[- ]augmented|mlops|"
    r"llmops|foundation model|generative ai)\b", re.I)
# A company doing THIS builds AI for clients → skip (no internal pain).
_CONSULTING = re.compile(
    r"\b(consult\w*|advisory|agency|staffing|system integrator|systems integration|outsourc\w*|"
    r"forward[- ]deployed|client engagement|client delivery|professional services|body shop|"
    r"managed services|for our clients|on behalf of clients)\b", re.I)
# Extra evidence the pain is internal (a scorer, not a gate).
_INTERNAL = re.compile(
    r"\b(in[- ]house|internal|our (own )?(platform|product|data|stack|customers)|build(ing)? our|"
    r"production ai|our ai (platform|systems)|self[- ]host)\b", re.I)


@dataclass(frozen=True)
class JobListing:
    company: str
    title: str
    url: str = ""
    description: str = ""
    location: str = ""
    posted: str = ""
    source: str = ""

    def key(self) -> str:
        """Stable dedup key: normalized company + role (so a reappearing listing collides)."""
        def norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
        return f"{norm(self.company)}|{norm(self.title)}"

    def _text(self) -> str:
        return f"{self.title}\n{self.company}\n{self.description}"


@dataclass(frozen=True)
class Classification:
    is_lead: bool
    kind: str          # "internal" | "consulting" | "other"
    reason: str
    score: float       # 0..1 confidence it's a real internal-AI-pain lead


def classify(listing: JobListing) -> Classification:
    """Is this listing a real internal-AI-adoption lead (vs. a consultancy hiring, or unrelated)?"""
    text = listing._text()
    ai = bool(_AI_ROLE.search(text))
    consulting = bool(_CONSULTING.search(text))
    if consulting:
        return Classification(False, "consulting", "consultancy / forward-deployed — builds AI for clients", 0.0)
    if not ai:
        return Classification(False, "other", "not an AI-engineering role", 0.0)
    internal = bool(_INTERNAL.search(text))
    score = 0.7 + (0.3 if internal else 0.0)
    reason = "internal AI-engineering hire" + (" (explicit in-house signals)" if internal else "")
    return Classification(True, "internal", reason, score)


# ── editable pitch template (agent-editable, shared by all users) ──────────────────────────────────
DEFAULT_TEMPLATE = """Hi,

I came across your {title} opening.

{match}

I started with RAG. Then added reranking. Then memory, model routing, verification, permissions.
Eventually I realized every AI application was rebuilding the same execution layer.

That became Context Runtime — an open-source runtime that plans retrieval, memory, model selection,
verification and policy before execution, measures every decision, and continuously improves from
production feedback. It's implemented in both Python and Go, includes heterogeneous retrieval
benchmarks, execution observability and explainability, and is documented in an open whitepaper.

After reading your job description, I think there's an opportunity beyond discussing a developer role.
I'd be interested in talking about whether this architecture could simplify how your team builds and
operates enterprise AI systems.

Resources:
• Whitepaper v2 — https://redevops.io/whitepaper-v2
• Benchmarks — https://redevops.io/benchmarks
• Context Runtime — https://redevops.io/context-runtime/under-the-hood/
• Planner — https://redevops.io/planner/under-the-hood/
• Retrieval Engine — https://redevops.io/redevops-rag/under-the-hood/

Best,
Alex"""

_DEFAULT_MATCH = ("Almost every responsibility listed is something I've been working on over the past "
                  "year while building production AI systems.")


class PitchTemplate:
    """The outreach letter template, persisted to a shared path so every user's agent edits the same
    one. Placeholders: ``{title}``, ``{company}``, ``{match}`` (the listing-tailored paragraph)."""

    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("PITCH_TEMPLATE_PATH", "/data/market-radar/pitch_template.txt")

    def load(self) -> str:
        p = Path(self.path)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
        return DEFAULT_TEMPLATE

    def save(self, text: str) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        Path(tmp).write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)     # atomic; all users see the update

    def reset(self) -> None:
        self.save(DEFAULT_TEMPLATE)


def _needs_resume(listing: JobListing) -> bool:
    return bool(re.search(r"\b(resume|cv|curriculum vitae)\b", listing.description, re.I))


def write_pitch(listing: JobListing, template: str | None = None, *, model=None,
                resume_path: str | None = None) -> dict:
    """Render a listing-tailored pitch. ``{match}`` is written by the model from the listing when one is
    given (else a sensible default). Returns {subject, body, to_hint, attach}."""
    tpl = template if template is not None else DEFAULT_TEMPLATE
    match = _DEFAULT_MATCH
    if model is not None and listing.description:
        match = _llm_match(listing, model) or match
    body = tpl.format(title=listing.title.strip() or "AI role",
                      company=listing.company.strip() or "your team", match=match)
    subject = f"Re: your {listing.title.strip() or 'AI'} opening — an execution layer for enterprise AI"
    attach = (resume_path or os.getenv("RESUME_PATH", "")) if _needs_resume(listing) else ""
    return {"subject": subject, "body": body, "to_hint": listing.url, "attach": attach}


def _llm_match(listing: JobListing, model) -> str | None:
    from ..types import ModelRequest
    prompt = (
        "You are tailoring a cold outreach letter. Given the job description below, write ONE short "
        "paragraph (2 sentences, first person, confident, no fluff) noting that the listed "
        "responsibilities — name 2-3 specific ones actually present (e.g. RAG, model routing, "
        "evaluation, retrieval, agents) — are exactly what I've built into a production AI execution "
        f"layer. Do not greet or sign off.\n\nTitle: {listing.title}\nCompany: {listing.company}\n"
        f"Description:\n{listing.description[:1500]}"
    )
    try:
        res = model.complete(ModelRequest(messages=({"role": "user", "content": prompt},),
                                          system="Write only the paragraph.", max_tokens=160))
        text = (res.text or "").strip()
        return text or None
    except Exception:  # noqa: BLE001
        return None


# ── dedup ledger (never pitch the same listing twice) ──────────────────────────────────────────────
class OutreachLedger:
    """Append-only record of pitched listings. ``already_pitched`` is True for a listing whose key or
    url has been pitched within ``window_days`` (default: effectively forever), so a listing that
    reappears months later is not re-pitched."""

    def __init__(self, path: str | None = None, window_days: int = 3650):
        self.path = path or os.getenv("OUTREACH_LEDGER_PATH", "/data/market-radar/outreach_ledger.jsonl")
        self.window_s = window_days * 86400
        self._seen: dict[str, float] = {}
        self._urls: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        p = Path(self.path)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            self._seen[r.get("key", "")] = r.get("at", 0.0)
            if r.get("url"):
                self._urls[r["url"]] = r.get("at", 0.0)

    def already_pitched(self, listing: JobListing, *, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        for at in (self._seen.get(listing.key()), self._urls.get(listing.url) if listing.url else None):
            if at is not None and (now - at) <= self.window_s:
                return True
        return False

    def record(self, listing: JobListing, channel: str = "draft", *, now: float | None = None) -> None:
        at = time.time() if now is None else now
        self._seen[listing.key()] = at
        if listing.url:
            self._urls[listing.url] = at
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": listing.key(), "url": listing.url, "company": listing.company,
                                "title": listing.title, "channel": channel, "at": at}) + "\n")


# ── sourcing (pluggable) ───────────────────────────────────────────────────────────────────────────
class StaticJobSource:
    """A fixed list of listings — for tests and for feeding pre-scraped data."""

    def __init__(self, listings: list[JobListing]):
        self._listings = list(listings)

    def search(self, query: str = "", limit: int = 50) -> list[JobListing]:
        return self._listings[:limit]


class CallableJobSource:
    """Wrap any ``search_fn(query, limit) -> list[dict|JobListing]`` (e.g. Market Radar's web_search MCP
    tool, a Greenhouse/Lever/Ashby public API, or an RSS parser) into a JobSource."""

    def __init__(self, search_fn: Callable[[str, int], list]):
        self.search_fn = search_fn

    def search(self, query: str, limit: int = 50) -> list[JobListing]:
        out: list[JobListing] = []
        for row in self.search_fn(query, limit) or []:
            if isinstance(row, JobListing):
                out.append(row)
            elif isinstance(row, dict):
                out.append(JobListing(
                    company=row.get("company", ""), title=row.get("title", ""), url=row.get("url", ""),
                    description=row.get("description") or row.get("snippet", ""),
                    location=row.get("location", ""), posted=row.get("posted", ""),
                    source=row.get("source", "search")))
        return out


class AdzunaSource:
    """Live listings from the Adzuna jobs API (free tier: ``ADZUNA_APP_ID`` / ``ADZUNA_APP_KEY``; set
    ``ADZUNA_COUNTRY`` e.g. us/gb). Search-based (good for "AI Data Engineer") and ToS-friendly, unlike
    scraping LinkedIn. Degrades to no results when unconfigured or unreachable; inject ``client`` for tests."""

    def __init__(self, app_id: str | None = None, app_key: str | None = None,
                 country: str | None = None, client=None):
        self.app_id = app_id or os.getenv("ADZUNA_APP_ID", "")
        self.app_key = app_key or os.getenv("ADZUNA_APP_KEY", "")
        self.country = (country or os.getenv("ADZUNA_COUNTRY", "us")).lower()
        self._client = client

    def search(self, query: str, limit: int = 50) -> list[JobListing]:
        if not (self.app_id and self.app_key):
            return []
        params = {"app_id": self.app_id, "app_key": self.app_key, "what": query,
                  "results_per_page": min(limit, 50), "content-type": "application/json"}
        try:
            client = self._client
            if client is None:
                import httpx
                client = httpx.Client(timeout=10.0)
            r = client.get(f"https://api.adzuna.com/v1/api/jobs/{self.country}/search/1", params=params)
            data = r.json()
        except Exception:  # noqa: BLE001
            return []
        out: list[JobListing] = []
        for j in (data.get("results") or [])[:limit]:
            out.append(JobListing(
                company=(j.get("company") or {}).get("display_name", ""), title=j.get("title", ""),
                url=j.get("redirect_url", ""), description=j.get("description", ""),
                location=(j.get("location") or {}).get("display_name", ""),
                posted=j.get("created", ""), source="adzuna"))
        return out


DEFAULT_QUERIES = (
    '"AI Data Engineer" hiring', '"AI Engineer" (in-house OR platform) hiring',
    '"AI Developer" build production AI', '"Machine Learning Engineer" our platform hiring',
)


def find_leads(source, ledger: OutreachLedger | None = None, *, queries=DEFAULT_QUERIES,
               limit: int = 50, min_score: float = 0.7) -> list[dict]:
    """Search → classify → keep internal-AI leads → drop already-pitched. Returns lead dicts sorted by
    score (best first), each with the listing + its classification."""
    seen: set[str] = set()
    leads: list[dict] = []
    for q in queries:
        for listing in source.search(q, limit):
            k = listing.key()
            if k in seen:
                continue
            seen.add(k)
            c = classify(listing)
            if not c.is_lead or c.score < min_score:
                continue
            if ledger is not None and ledger.already_pitched(listing):
                continue
            leads.append({"listing": listing, "classification": c})
    leads.sort(key=lambda d: d["classification"].score, reverse=True)
    return leads[:limit]


# ── distribution (default: draft; real send is approval-gated deployment config) ───────────────────
class DraftDistributor:
    """Writes each pitch to an outbox directory instead of sending — the safe default. A real
    SMTP/CRM sender is a deployment concern behind explicit approval; this never sends on its own."""

    def __init__(self, outbox: str | None = None):
        self.outbox = outbox or os.getenv("OUTREACH_OUTBOX", "/data/market-radar/outbox")

    def send(self, listing: JobListing, pitch: dict, *, to: str = "") -> dict:
        p = Path(self.outbox)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.mkdir(parents=True, exist_ok=True)
        fname = re.sub(r"[^a-z0-9]+", "-", listing.key().replace("|", "-")).strip("-") + ".txt"
        doc = (f"To: {to or '(find via ' + (listing.url or 'company site') + ')'}\n"
               f"Subject: {pitch['subject']}\n"
               f"Attach: {pitch['attach'] or '(none)'}\n\n{pitch['body']}\n")
        (p / fname).write_text(doc, encoding="utf-8")
        return {"mode": "draft", "path": str(p / fname), "subject": pitch["subject"], "attach": pitch["attach"]}
