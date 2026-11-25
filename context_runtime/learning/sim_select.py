"""Choose which of a Self-Improvement Memory to show when the whole record does not fit - without losing evidence.

The record is agent-harness's ``agent-harness.sim/1`` (problems = root-cause hypotheses, each with the harness edits
tried against it, what each predicted before evaluation and what was then measured). The method it implements (arXiv
2611.18856, section 3.2) gives the optimizer the COMPLETE record; :func:`select` returns exactly that whenever it fits the
budget. When it does not, the result is a **view** for one decision, never the record: do not save it over the memory.

What a view guarantees (tests/test_sim_select.py):

* **No problem disappears.** Each one is either shown, or listed in ``selection.omitted`` with its root cause, how many
  edits were tried and accepted, how many predictions failed and which tasks its edits broke.
* **A shown problem keeps its whole history.** Every attempt, with ``predicted``, ``fixed``, ``missed``, ``regressed``,
  scores, acceptance, files and provenance exactly as recorded. To fit, a problem may be *compacted*: the prose of its
  attempts is shortened and ``rationale`` left out. Nothing measured is ever changed, shortened or dropped.
* **Regression evidence that bears on this decision is mandatory**: a problem whose edit broke a task that is failing now,
  or broke something while editing a file now in question, is always shown.
* **Relationships are stated**, also toward omitted problems: merged ids (all aliases are kept), edits of one problem that
  broke a task another one addresses, and problems that work on the same tasks.
* **It fails rather than drop evidence**: when the mandatory problems and the index do not fit the budget,
  :class:`BudgetTooSmall` says how much is needed.

Ranking is lexical and structural (shared tasks and files, term overlap, recency), deterministic, and uses no model.
Whether a view serves the optimizer as well as the complete record is an experimental question this module does not answer.
"""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "agent-harness.sim/1"
METHOD = "context-runtime.sim-select/1"
#: how much of an attempt's description a compacted attempt keeps
COMPACT_CHARS = 160
_WORD = re.compile(r"[a-z0-9_]{3,}")


class SelectionError(ValueError):
    pass


class BudgetTooSmall(SelectionError):
    """The evidence that must be shown does not fit. ``needed`` is the smallest budget (in the counter's tokens) that holds it."""

    def __init__(self, budget: int, needed: int, mandatory: Sequence[str]) -> None:
        self.budget, self.needed, self.mandatory = budget, needed, tuple(mandatory)
        super().__init__(f"a view needs at least {needed} tokens to show the mandatory evidence ({', '.join(mandatory) or 'the index of problems'}); "
                         f"the budget is {budget}. Raise the budget: evidence is not dropped to fit")


@dataclass(frozen=True)
class Query:
    """What is being decided now: the tasks that currently fail, the harness files under consideration, and any text
    describing the failure."""

    failing_tasks: FrozenSet[str] = frozenset()
    files: FrozenSet[str] = frozenset()
    text: str = ""

    @staticmethod
    def of(failing_tasks: Iterable[str] = (), files: Iterable[str] = (), text: str = "") -> "Query":
        return Query(frozenset(map(str, failing_tasks)), frozenset(map(str, files)), str(text))


def estimate_tokens(text: str) -> int:
    """A rough count (four characters to a token). Pass the target model's own counter when it matters."""
    return (len(text) + 3) // 4


def render(view: Mapping[str, Any]) -> str:
    """The text placed in a prompt: the same serialisation agent-harness uses for the complete record."""
    return json.dumps(view, indent=1, sort_keys=True, ensure_ascii=False)


# ── facts about one problem ─────────────────────────────────────────────────────────────────────────────────
def _tasks(problem: Mapping[str, Any], *fields: str) -> set:
    return {str(t) for a in problem.get("attempts", []) for f in fields for t in a.get(f, ())}


def _files(problem: Mapping[str, Any]) -> set:
    return {str(f) for a in problem.get("attempts", []) for f in a.get("files", ())}


def _words(text: str) -> set:
    return set(_WORD.findall(text.lower()))


def _prose(problem: Mapping[str, Any]) -> str:
    return " ".join([str(problem.get("name", ""))] + [f"{a.get('what', '')} {a.get('rationale', '')}" for a in problem.get("attempts", [])])


def _mandatory(problem: Mapping[str, Any], query: Query) -> bool:
    for a in problem.get("attempts", []):
        broke = {str(t) for t in a.get("regressed", ())}
        if broke and (broke & query.failing_tasks or {str(f) for f in a.get("files", ())} & query.files):
            return True
    return False


def relationships(problems: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """How problems bear on each other, from what was recorded: ``broke_task_of`` (an edit for one broke a task another
    predicted to fix or fixed) and ``shares_tasks`` (both addressed the same task)."""
    addressed = {p["id"]: _tasks(p, "predicted", "fixed", "missed") for p in problems}
    broke = {p["id"]: _tasks(p, "regressed") for p in problems}
    out: List[Dict[str, Any]] = []
    for a in problems:
        for b in problems:
            if a["id"] == b["id"]:
                continue
            hit = sorted(broke[a["id"]] & addressed[b["id"]])
            if hit:
                out.append({"kind": "broke_task_of", "from": a["id"], "to": b["id"], "tasks": hit})
            if a["id"] < b["id"]:
                shared = sorted(addressed[a["id"]] & addressed[b["id"]])
                if shared:
                    out.append({"kind": "shares_tasks", "from": a["id"], "to": b["id"], "tasks": shared})
    return out


def score(problem: Mapping[str, Any], query: Query, *, latest_round: int, idf: Mapping[str, float]) -> float:
    """How much this problem's history bears on the query. A failed prediction about a task that still fails weighs most:
    it is what must not be proposed again unchanged."""
    failing = query.failing_tasks
    s = 5.0 * len(_tasks(problem, "missed") & failing) + 4.0 * len(_tasks(problem, "regressed") & failing)
    s += 3.0 * len(_tasks(problem, "fixed") & failing) + 1.0 * len(_tasks(problem, "predicted") & failing)
    s += 2.0 * len(_files(problem) & query.files)
    if query.text:
        s += sum(idf.get(w, 0.0) for w in _words(query.text) & _words(_prose(problem)))
    rounds = [int(a.get("round", 0)) for a in problem.get("attempts", [])] or [int(problem.get("opened_round", 0))]
    return s + 0.001 * (max(rounds) / max(latest_round, 1))          # only breaks ties between equally relevant ones


def compact(problem: Mapping[str, Any]) -> Dict[str, Any]:
    """The problem with the prose of its attempts shortened. Every attempt and every measured field stays."""
    out = copy.deepcopy(dict(problem))
    for a in out.get("attempts", []):
        what = str(a.get("what", ""))
        if len(what) > COMPACT_CHARS:
            a["what"] = what[:COMPACT_CHARS].rstrip() + " […]"
        if a.get("rationale"):
            a["rationale"] = ""
        a["compacted"] = True
    return out


def _index_entry(problem: Mapping[str, Any]) -> Dict[str, Any]:
    attempts = problem.get("attempts", [])
    return {"id": problem["id"], "name": problem.get("name", ""), "attempts": len(attempts),
            "accepted": sum(bool(a.get("accepted")) for a in attempts),
            "failed_predictions": sum(len(a.get("missed", ())) for a in attempts),
            "regressed_tasks": sorted(_tasks(problem, "regressed"))}


def _assemble(record: Mapping[str, Any], shown: Mapping[str, str], links: Sequence[Mapping[str, Any]],
              mandatory: Sequence[str]) -> Dict[str, Any]:
    problems, omitted, compacted = [], [], []
    for p in record.get("problems", []):                            # the record's own order, whatever the ranking was
        how = shown.get(p["id"])
        if how == "full":
            problems.append(copy.deepcopy(dict(p)))
        elif how == "compact":
            problems.append(compact(p))
            compacted.append(p["id"])
        else:
            omitted.append(_index_entry(p))
    visible = {p["id"] for p in problems}
    view = {k: copy.deepcopy(record[k]) for k in ("schema", "run", "next_id", "carried_from", "aliases") if k in record}
    view["problems"] = problems
    view["selection"] = {
        # (the budget is not written here: the view's size must not depend on the number it is being fitted to)
        "method": METHOD, "view_not_record": True, "complete": not omitted and not compacted,
        "selected": [p["id"] for p in problems], "compacted": compacted, "mandatory": sorted(mandatory), "omitted": omitted,
        "relationships": [dict(link) for link in links if link["from"] in visible or link["to"] in visible],
    }
    return view


def select(record: Mapping[str, Any], query: Query, *, budget_tokens: int,
           count: Callable[[str], int] = estimate_tokens) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(view, report)`` for one decision. The record is not modified.

    The complete record is returned unchanged (``report["complete"] is True``) when it fits the budget. ``report`` holds
    the token counts and the ids, for comparing selection with complete memory."""
    if record.get("schema") != SCHEMA:
        raise SelectionError(f"not a {SCHEMA} record")
    if budget_tokens <= 0:
        raise SelectionError("the budget must be a positive number of tokens")
    problems = list(record.get("problems", []))
    ids = [p["id"] for p in problems]
    if len(ids) != len(set(ids)):
        raise SelectionError("problem ids repeat")
    whole = {k: copy.deepcopy(v) for k, v in record.items()}
    complete_tokens = count(render(whole))
    relevant = {p["id"] for p in problems
                if _tasks(p, "predicted", "fixed", "missed", "regressed") & query.failing_tasks or _files(p) & query.files}

    def report(view: Mapping[str, Any], tokens: int, complete: bool) -> Dict[str, Any]:
        meta = view.get("selection", {})
        return {"method": METHOD, "complete": complete, "budget_tokens": budget_tokens, "tokens": tokens,
                "tokens_complete": complete_tokens, "problems": len(problems),
                "selected": meta.get("selected", ids), "compacted": meta.get("compacted", []),
                "omitted": [o["id"] for o in meta.get("omitted", [])], "mandatory": meta.get("mandatory", []),
                # problems that share a task or a file with this decision and still did not fit: the view's known cost
                "relevant_omitted": [o["id"] for o in meta.get("omitted", []) if o["id"] in relevant]}

    if complete_tokens <= budget_tokens:
        return whole, report(whole, complete_tokens, True)

    document_frequency: Dict[str, int] = {}
    for p in problems:
        for w in _words(_prose(p)):
            document_frequency[w] = document_frequency.get(w, 0) + 1
    idf = {w: math.log(1 + len(problems) / n) for w, n in document_frequency.items()}
    latest = max([int(a.get("round", 0)) for p in problems for a in p.get("attempts", [])] or [1])
    ranked = sorted(problems, key=lambda p: (-score(p, query, latest_round=latest, idf=idf), p["id"]))
    links = relationships(problems)
    mandatory = [p["id"] for p in problems if _mandatory(p, query)]

    def cost(shown: Mapping[str, str]) -> int:
        return count(render(_assemble(record, shown, links, mandatory)))

    shown: Dict[str, str] = {pid: "compact" for pid in mandatory}
    needed = cost(shown)
    if needed > budget_tokens:
        raise BudgetTooSmall(budget_tokens, needed, mandatory)
    # mandatory problems get their full text back first, then the others by relevance; full where it fits, else compacted
    for p in sorted(ranked, key=lambda p: p["id"] not in mandatory):
        for how in ("full", "compact"):
            trial = {**shown, p["id"]: how}
            if cost(trial) <= budget_tokens:
                shown = trial
                break
    view = _assemble(record, shown, links, mandatory)
    return view, report(view, count(render(view)), False)
