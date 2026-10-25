"""Intelligence Gateway — plan and run evidence acquisition from the managed gateway, inside the Context Runtime's own limits.

The gateway (``redevops-io/intelligence-gateway``) sells decision-ready external evidence (company profiles, filings, sanctions and the like) behind one API key. This module lets the
planner consider it as one more way to retrieve, and lets the runtime call it, without weakening anything the gateway or the runtime enforces:

* **Rights, authorization and spend belong to the gateway and to the goal's constraints, and are never decided here.** The planner only *offers* a gateway plan when the gateway says the
  capability is served and entitled to this caller, a quote is available, and the quote fits the goal's ``max_cost_usd``. A refusal (rights, entitlement, spend cap, an unavailable provider) removes
  the plan or yields no evidence. There is no retry through a different route and no fallback that widens what the caller asked for; the client never sends a provider list or a jurisdiction.
* **What leaves the boundary is explicit.** Only the capability, subject references, requested fields and a purpose are sent: never the goal text. Goals marked ``restricted`` never use
  the gateway, and ``internal`` ones only when the caller opts in.
* **Fail closed.** A gateway that cannot be reached, answers something unexpected, or has no answer for this need contributes nothing; the runtime then plans on what it has.

Wiring (all opt-in; a goal without a gateway source plans exactly as before)::

    client = GatewayClient("https://api.redevops.app", token_provider=my_tokens)      # token from the install identity
    book = NeedBook()
    planner = GatewayCandidateGenerator(RegistryCandidateGenerator(), client, book)
    retriever = GatewayRetriever(local_retriever, client, book)
    rt = ContextRuntime(models=..., retriever=retriever, candidates=planner)
    rt.run(Goal("is Acme sanctioned?", sources=(gateway_source("sanctions_screen", subjects=["acme"]),)))

Agent feedback is deliberately not a parameter of anything here: it may inform a later ranking of plans, but it has no path to rights, entitlement or spend.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..types import Candidate, Goal, Hit, Intent, SourceRef, StepSpec

GATEWAY_SOURCE = "gateway"
GATEWAY_METHOD_PREFIX = "gateway:"


# ───────────────────────────── the need a goal declares ─────────────────────────────

@dataclass(frozen=True)
class GatewayNeed:
    """What to ask the gateway for. ``question`` is empty unless the caller writes one: the goal text is never sent."""
    capability: str
    subject_refs: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    purpose: str = ""
    question: str = ""
    max_cost: float = 0.0               # 0 = no per-need ceiling beyond the goal's max_cost_usd
    min_confidence: float = 0.0

    @property
    def key(self) -> str:
        blob = json.dumps([self.capability, list(self.subject_refs), list(self.fields), self.purpose, self.question], separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def body(self, max_cost: Optional[float] = None) -> dict[str, Any]:
        """``max_cost`` is the most the gateway may spend on this need: the gateway treats 0 as "spend nothing", so execution passes the quoted price, never the default."""
        return {"capability": self.capability, "subject_refs": list(self.subject_refs), "fields": list(self.fields),
                "purpose": self.purpose, "question": self.question, "max_cost": self.max_cost if max_cost is None else max_cost,
                "min_confidence": self.min_confidence, "objective": "context-runtime"}


def gateway_source(capability: str, *, subjects=(), fields=(), purpose: str = "", question: str = "",
                   max_cost: float = 0.0, min_confidence: float = 0.0) -> SourceRef:
    """A goal source declaring that this goal needs gateway evidence."""
    q = urllib.parse.urlencode([("capability", capability), *[("subject", s) for s in subjects], *[("field", f) for f in fields],
                                ("purpose", purpose), ("question", question), ("max_cost", repr(float(max_cost))),
                                ("min_confidence", repr(float(min_confidence)))])
    return SourceRef(name=GATEWAY_SOURCE, kind="api", uri=f"gateway://need?{q}")


def needs_from_goal(goal: Goal) -> list[GatewayNeed]:
    out: list[GatewayNeed] = []
    for s in goal.sources:
        if s.name != GATEWAY_SOURCE or not (s.uri or "").startswith("gateway://need?"):
            continue
        q = urllib.parse.parse_qs(s.uri.split("?", 1)[1], keep_blank_values=True)
        cap = (q.get("capability") or [""])[0]
        if not cap:
            continue
        try:
            out.append(GatewayNeed(cap, tuple(q.get("subject", [])), tuple(q.get("field", [])), (q.get("purpose") or [""])[0],
                                   (q.get("question") or [""])[0], float((q.get("max_cost") or ["0"])[0]),
                                   float((q.get("min_confidence") or ["0"])[0])))
        except ValueError:
            continue                      # a malformed need is no need: it plans nothing rather than something else
    return out


class NeedBook:
    """The planner records the needs it planned; the retriever runs only needs it finds here (so a step can carry just a key)."""

    def __init__(self) -> None:
        self._needs: dict[str, GatewayNeed] = {}
        self._quotes: dict[str, float] = {}

    def add(self, need: GatewayNeed, quote_total: float, provider_cost: float = 0.0) -> str:
        self._needs[need.key] = need
        self._quotes[need.key] = (quote_total, provider_cost)
        return need.key

    def planned(self, key: str) -> "tuple[float, float]":
        """(total, provider cost) the planner was quoted for this need."""
        return self._quotes.get(key, (0.0, 0.0))

    def get(self, key: str) -> Optional[GatewayNeed]:
        return self._needs.get(key)


# ───────────────────────────── the client ─────────────────────────────

class GatewayError(RuntimeError):
    """The gateway cannot serve this: ``status`` is the HTTP status (0 = unreachable); ``rights_blocked`` is the 451 case."""

    def __init__(self, status: int, detail: str = "") -> None:
        super().__init__(f"gateway {status}: {detail}"[:300])
        self.status, self.detail = status, detail

    @property
    def rights_blocked(self) -> bool:
        return self.status == 451


Transport = Callable[[str, str, dict, Optional[bytes]], "tuple[int, Any]"]


def _urllib_transport(timeout: float) -> Transport:
    def send(method: str, url: str, headers: dict, body: Optional[bytes]):
        req = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:                         # noqa: S310 - the base URL is the caller's
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}
        except (urllib.error.URLError, OSError, ValueError):
            return 0, {}
    return send


class GatewayClient:
    """A thin client. The bearer token comes from ``token_provider`` at every call (it is short-lived); this class never stores one."""

    def __init__(self, base_url: str, token_provider: Callable[[], str], *, transport: Optional[Transport] = None, timeout: float = 10.0) -> None:
        self.base = base_url.rstrip("/")
        self._token = token_provider
        self._send = transport or _urllib_transport(timeout)

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        try:
            tok = self._token()
        except Exception as exc:                                                            # noqa: BLE001 - no credential, no call
            raise GatewayError(0, f"no credential: {type(exc).__name__}") from None
        headers = {"Authorization": f"Bearer {tok}", "Accept": "application/json"}
        data = None
        if body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
        status, payload = self._send(method, self.base + path, headers, data)
        if status != 200 or not isinstance(payload, dict):
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
            raise GatewayError(status, str(detail))
        return payload

    def capabilities(self) -> dict:
        return self._call("GET", "/v1/capabilities")

    def quote(self, need: GatewayNeed) -> dict:
        return self._call("POST", "/v1/quote", need.body())

    def resolve(self, need: GatewayNeed, *, max_cost: Optional[float] = None) -> dict:
        return self._call("POST", "/v1/resolve", need.body(max_cost))


# ───────────────────────────── planning ─────────────────────────────

@dataclass
class GatewayDecision:
    """Why a gateway plan was or was not offered: for ``explain`` and for tests."""
    capability: str
    offered: bool
    reason: str
    quote_total: float = 0.0
    provider_cost: float = 0.0


class GatewayCandidateGenerator:
    """Wraps a candidate generator. For each gateway need a goal declares it offers retrieval-from-gateway plans, but only when every check passes.

    ``mode="required"`` (default): a goal that declares a gateway need means it, so when a gateway plan is offered the other plans are pruned; when none is offered the base plans
    remain, so the goal still runs on local evidence and ``decisions`` says why. ``mode="augment"`` keeps both and lets the optimizer choose.
    """

    def __init__(self, base, client: GatewayClient, book: NeedBook, *, mode: str = "required", allow_internal: bool = False) -> None:
        if mode not in ("required", "augment"):
            raise ValueError("mode must be 'required' or 'augment'")
        self.base, self.client, self.book, self.mode, self.allow_internal = base, client, book, mode, allow_internal
        self.decisions: list[GatewayDecision] = []

    def _check(self, need: GatewayNeed, goal: Goal) -> GatewayDecision:
        sens = goal.constraints.sensitivity
        if sens == "restricted" or (sens == "internal" and not self.allow_internal):
            return GatewayDecision(need.capability, False, f"{sens} goals do not use an external service")
        try:
            caps = self.client.capabilities()
            if need.capability not in (caps.get("entitled") or []):
                return GatewayDecision(need.capability, False, "not served or not entitled for this caller")
            q = self.client.quote(need)
        except GatewayError as exc:
            return GatewayDecision(need.capability, False, "rights-blocked" if exc.rights_blocked else f"gateway refused or unavailable ({exc.status})")
        if not q.get("available"):
            return GatewayDecision(need.capability, False, "no provider available")
        try:
            total = float(q.get("total"))
        except (TypeError, ValueError):
            return GatewayDecision(need.capability, False, "quote unreadable")
        ceilings = [c for c in (goal.constraints.max_cost_usd, need.max_cost or None) if c is not None]
        if ceilings and total > min(ceilings):
            return GatewayDecision(need.capability, False, "quote is over the budget", total)
        return GatewayDecision(need.capability, True, "cleared", total, float(q.get("provider_cost") or 0.0))

    def generate(self, intent: Intent, goal: Goal) -> list[Candidate]:
        base = list(self.base.generate(intent, goal))
        self.decisions = []
        needs = needs_from_goal(goal)
        if not needs:
            return base
        offered: list[Candidate] = []
        for need in needs[:1]:                  # one gateway need per plan; more needs mean more goals
            d = self._check(need, goal)
            self.decisions.append(d)
            if not d.offered:
                continue
            key = self.book.add(need, d.quote_total, d.provider_cost)
            seen = set()
            for cand in base:
                if cand.model_tier in seen:
                    continue
                seen.add(cand.model_tier)
                steps = tuple(StepSpec(s.type, {**s.params, "method": GATEWAY_METHOD_PREFIX + key, "gateway_quote_usd": d.quote_total}, s.plugin)
                              if s.type == "retrieve" else s for s in cand.steps)
                offered.append(Candidate(steps=steps, model_tier=cand.model_tier))
        if not offered:
            return base
        return offered if self.mode == "required" else offered + base

    def prune(self, candidates: list[Candidate], goal: Goal) -> list[Candidate]:
        return self.base.prune(candidates, goal)

    def info(self):
        return self.base.info()


# ───────────────────────────── execution ─────────────────────────────

class GatewayRetriever:
    """Wraps the runtime's retriever. Only the ``gateway:<key>`` method reaches the gateway, and only for a need the planner recorded."""

    def __init__(self, base, client: GatewayClient, book: NeedBook) -> None:
        self.base, self.client, self.book = base, client, book
        self.refusals: list[dict] = []
        self.spent_usd = 0.0
        self.receipts: list[dict] = []

    def __getattr__(self, name):                # everything else (index, info, ...) is the wrapped retriever's
        return getattr(self.base, name)

    def search(self, query: str, k: int, method: str = "hybrid") -> list[Hit]:
        if not str(method).startswith(GATEWAY_METHOD_PREFIX):
            return self.base.search(query, k, method=method)
        need = self.book.get(str(method)[len(GATEWAY_METHOD_PREFIX):])
        if need is None:
            self.refusals.append({"reason": "unknown need"})
            return []
        try:
            # a price can move between planning and execution: quote again, and never spend past what the need or the plan allowed
            q = self.client.quote(need)
            if not q.get("available"):
                self.refusals.append({"capability": need.capability, "reason": "no provider available"})
                return []
            planned_total, planned_provider = self.book.planned(need.key)
            if float(q.get("total", 0.0)) > planned_total + 1e-9:
                self.refusals.append({"capability": need.capability, "reason": "price rose above the planned quote"})
                return []
            # the gateway may spend at most the provider cost that was quoted at planning (never the need's default of 0, which means "spend nothing")
            r = self.client.resolve(need, max_cost=planned_provider)
        except GatewayError as exc:
            self.refusals.append({"capability": need.capability, "reason": "rights-blocked" if exc.rights_blocked else f"refused or unavailable ({exc.status})", "status": exc.status})
            return []
        except (TypeError, ValueError):
            self.refusals.append({"capability": need.capability, "reason": "unreadable response"})
            return []
        self.spent_usd = round(self.spent_usd + float(r.get("charge_total") or 0.0), 6)
        self.receipts.append({"capability": need.capability, "decision_need_id": r.get("decision_need_id"),
                              "provider_receipts": r.get("provider_receipts", []), "charge_total": r.get("charge_total")})
        answer = r.get("answer")
        text = answer if isinstance(answer, str) else json.dumps(answer, sort_keys=True, default=str)
        if not text:
            # charged but nothing to use: record it (the money is in spent_usd) rather than let it look like a free miss
            self.refusals.append({"capability": need.capability, "reason": "empty answer", "charged_usd": float(r.get("charge_total") or 0.0)})
            return []
        return [Hit(chunk_id=f"gw:{r.get('decision_need_id') or need.key}", filename=f"gateway/{need.capability}", text=text,
                    score=float(r.get("confidence") or 0.0), source="intelligence-gateway",
                    meta={"provider_receipts": r.get("provider_receipts", []), "charge_total": r.get("charge_total"),
                          "fingerprint": r.get("fingerprint"), "known_at": r.get("known_at")},
                    version=None, content_hash=r.get("fingerprint"), observed_at=r.get("as_of"))][:max(1, k)]
