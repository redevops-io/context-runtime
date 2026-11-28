"""Narration verification: a model's wording of a structured result must not change or omit what the result says.

A provider that computes an exact result attaches the values the answer is accountable for to its hit as ``meta["facts"]``
(``Fact``: kind ``number`` | ``identifier`` | ``date`` | ``text``, a value, an optional unit). ``verify_narration`` then checks, with
token-boundary matching (``42`` is not found in ``00042`` or ``142``):

* every required fact is present, with its unit when one is declared (omission);
* no number in the answer is absent from the structured result and the question (a changed or invented figure).

A failed check never goes to the user as it is: ``serve_verified`` returns the structured rendering instead, and the report says so.
An explicit abstention is not held to the required-facts rule (it asserts nothing), but still to the unsupported-number rule.
Domain-free: nothing here knows what the facts are about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Iterable, Sequence

# a figure: no leading zeros (00042 is an identifier), not glued to a word by a hyphen (INV-00042), not a fragment of a longer token
_NUM = re.compile(r"(?<![\w.])(?<!\w-)\$?(?:0|[1-9][\d,]*)(?:\.\d+)?%?(?![\w]|\.\d|,\d)")
_CITE = re.compile(r"\[\d+\]")
_LISTMARK = re.compile(r"(?m)^\s*\d+[.)]\s")
ABSTENTION = re.compile(
    r"\b(cannot be answered|can(?:no|')t (?:be )?(?:answer|determine)|unable to|insufficient (?:evidence|information)|no information|"
    r"not (?:disclosed|available|published|provided)|does not (?:contain|disclose|provide|publish)|discloses no|no (?:such|data|record))", re.I)


@dataclass(frozen=True)
class Fact:
    kind: str                      # "number" | "identifier" | "date" | "text"
    value: str                     # the exact string the structured result carries
    unit: str | None = None        # "USD", "$", "%", "days" ... (number only)
    label: str = ""

    @staticmethod
    def of(d: "Fact | dict") -> "Fact":
        return d if isinstance(d, Fact) else Fact(kind=d.get("kind", "text"), value=str(d["value"]), unit=d.get("unit"), label=d.get("label", ""))


@dataclass(frozen=True)
class NarrationReport:
    status: str                    # "ok" | "abstained" | "replaced" | "no_facts"
    missing: tuple[str, ...] = ()
    unsupported_numbers: tuple[str, ...] = ()
    findings: tuple[str, ...] = field(default=())

    @property
    def passed(self) -> bool:
        return self.status in ("ok", "abstained", "no_facts")


def _dec(token: str) -> Decimal | None:
    t = token.replace("$", "").replace(",", "").rstrip("%")
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def _numbers(text: str) -> list[Decimal]:
    text = _CITE.sub(" ", _LISTMARK.sub(" ", text))
    return [d for d in (_dec(m.group(0)) for m in _NUM.finditer(text)) if d is not None]


def facts_from_hits(hits: Iterable) -> list[Fact]:
    out: list[Fact] = []
    for h in hits or ():
        for f in (getattr(h, "meta", None) or {}).get("facts", ()) or ():
            out.append(Fact.of(f))
    return out


def _present(answer: str, f: Fact) -> bool:
    if f.kind == "number":
        want = _dec(f.value)
        if want is None:
            return False
        for m in _NUM.finditer(_CITE.sub(" ", answer)):
            if _dec(m.group(0)) == want:
                if not f.unit:
                    return True
                window = answer[max(0, m.start() - 4): m.end() + 14].lower()
                if f.unit.lower() in window:
                    return True
        return False
    pat = re.compile(r"(?<![\w-])" + re.escape(f.value) + r"(?![\w-])", re.I if f.kind == "text" else 0)
    return bool(pat.search(answer))


def verify_narration(answer: str, facts: Sequence[Fact | dict], *, structured_text: str = "", question: str = "") -> NarrationReport:
    facts = [Fact.of(f) for f in facts]
    if not facts:
        return NarrationReport("no_facts")
    abstained = bool(ABSTENTION.search(answer))
    missing = () if abstained else tuple(f"{f.label or f.kind}={f.value}" + (f" {f.unit}" if f.unit else "") for f in facts if not _present(answer, f))
    allowed = set(_numbers(structured_text)) | set(_numbers(question)) | {d for d in (_dec(f.value) for f in facts if f.kind == "number") if d is not None}
    unsupported = tuple(sorted({str(n) for n in _numbers(answer) if n not in allowed}))
    findings = tuple([f"omitted: {m}" for m in missing] + [f"figure not in the structured result or the question: {u}" for u in unsupported])
    if missing or unsupported:
        return NarrationReport("replaced", missing, unsupported, findings)
    return NarrationReport("abstained" if abstained else "ok")


def serve_verified(answer: str, facts: Sequence[Fact | dict], structured_text: str, *, question: str = "") -> tuple[str, NarrationReport]:
    """The answer to serve: the narration if it verifies, else the structured rendering (recorded, never silent)."""
    rep = verify_narration(answer, facts, structured_text=structured_text, question=question)
    return (structured_text, rep) if rep.status == "replaced" else (answer, rep)
