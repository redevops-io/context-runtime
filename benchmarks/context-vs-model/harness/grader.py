"""Answer grading — numeric/exact-match first, neutral LLM-judge for prose.

FinanceBench answers are mostly numeric ("$1577.00", "12.5%", "$(200) million"), so the
crisp path is deterministic numeric matching with scale/sign/percent handling. The ~24
prose answers (and any numeric miss we want a second opinion on) fall to a frontier judge.
"""
from __future__ import annotations

import re

_SCALE = {"thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6,
          "billion": 1e9, "billions": 1e9, "trillion": 1e12}
_NUM = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\)?")


def _to_float(tok: str) -> float | None:
    neg = tok.strip().startswith("(") and tok.strip().endswith(")")
    pct = "%" in tok
    t = tok.replace(",", "").replace("$", "").replace("%", "").replace("(", "").replace(")", "").strip()
    try:
        v = float(t)
    except ValueError:
        return None
    if neg:
        v = -v
    return v, pct  # type: ignore


def extract_numbers(s: str) -> list:
    """All numbers in ``s`` with any explicit scale word applied, as (value, is_pct)."""
    out = []
    low = s.lower()
    for m in _NUM.finditer(s):
        parsed = _to_float(m.group())
        if parsed is None:
            continue
        v, pct = parsed
        tail = low[m.end(): m.end() + 12]
        for word, mult in _SCALE.items():
            if tail.lstrip().startswith(word):
                v *= mult
                break
        out.append((v, pct))
    return out


def numeric_match(gold: str, cand: str, *, rel_tol: float = 0.01) -> bool | None:
    """True/False if a numeric verdict is possible, else None (defer to judge).

    Gold's primary number must appear in the candidate within ``rel_tol`` (relative), with
    a scale-agnostic retry (1577 vs 1,577 million) so a right value in different units still
    matches."""
    g = extract_numbers(gold)
    if not g:
        return None
    gv, gpct = g[0]
    cands = extract_numbers(cand)
    if not cands:
        return False
    for cv, _ in cands:
        for a, b in ((gv, cv), (gv, cv * 1e6), (gv * 1e6, cv), (gv, cv * 1e3), (gv * 1e3, cv)):
            if b == 0 and a == 0:
                return True
            if b != 0 and abs(a - b) / max(abs(b), 1e-9) <= rel_tol:
                return True
    return False


_JUDGE_SYS = (
    "You are a strict grader for financial QA. Given a QUESTION, the GOLD answer, and a "
    "MODEL answer, decide if the MODEL answer is correct — i.e. it states the same value/"
    "fact as GOLD (units/rounding/phrasing may differ). Reply with exactly one word: "
    "CORRECT or INCORRECT."
)


def judge_grade(judge_chat, question: str, gold: str, cand: str) -> bool:
    """``judge_chat(system, user) -> str`` (a frontier model). Returns True iff CORRECT."""
    user = f"QUESTION:\n{question}\n\nGOLD:\n{gold}\n\nMODEL:\n{cand}\n\nVerdict:"
    verdict = (judge_chat(_JUDGE_SYS, user) or "").strip().upper()
    return verdict.startswith("CORRECT")


def grade(question, cand: str, *, judge_chat=None, prefer_judge: bool = False) -> dict:
    """Return {correct: bool, method: 'numeric'|'judge'|'numeric+judge'}.

    Numeric match is authoritative when it fires; if it can't decide (prose gold or no
    number in the candidate) we defer to the judge when one is supplied."""
    nm = None if prefer_judge else numeric_match(question.answer, cand)
    if nm is True:
        return {"correct": True, "method": "numeric"}
    if nm is False and judge_chat is not None:
        # numeric said no — let the judge catch equivalent phrasings / unit slips
        jg = judge_grade(judge_chat, question.question, question.answer, cand)
        return {"correct": jg, "method": "numeric+judge"}
    if nm is False:
        return {"correct": False, "method": "numeric"}
    # nm is None → prose or undecidable
    if judge_chat is not None:
        return {"correct": judge_grade(judge_chat, question.question, question.answer, cand),
                "method": "judge"}
    return {"correct": False, "method": "ungraded"}
