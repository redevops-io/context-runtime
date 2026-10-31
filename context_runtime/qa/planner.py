"""Plan selection under budgets: static lanes, bounded parallelism, conflict keys, and FULL coverage.

``CostBasedQAPlanner`` turns a declared graph, a set of selected tests, declared estimates and budgets into a
canonical ``QAPlan`` (built with the ``runtime-contracts`` classes, so the contracts validators accept it and its
hash is the contracts hash). The plan is declarative: it names lanes, the order inside each lane and start
conditions (gates). It does not execute anything, dispatch dynamically or resolve dependencies at run time.

What is guaranteed, whatever the real durations turn out to be:

* **FULL coverage.** Every test of the closure appears in the plan exactly once. If the budget cannot cover them
  all the planner refuses (``InfeasibleBudget``); it never drops a test, and ``SUBSET`` is not a request it accepts.
* **Bounded parallelism.** The plan has at most ``max_parallel`` lanes, so at most that many tests run at once.
* **Conflict keys.** Two tests whose claims meet on an exclusive resource are ordered, in the same lane or through
  a gate chain, so they are never concurrent, whatever their real durations.
* **Pool capacities.** The lane count is capped so that even the largest claimants running together fit every
  declared capacity (the sum of the ``p`` largest claims must not exceed it). This holds for any real durations.
* **Budgets within caps.** Every item's budget and recovery ceiling are at most the declared run caps (and at most
  the spec's own budget when the specs are supplied).

Estimated times (start, finish, makespan) are derived from the declared estimates only; they are used to choose
between candidate lane counts and are reported with their basis, not promised.

Why a QA-local scheduler: ``context_runtime.scheduler.TopoScheduler`` produces Kahn waves over a retrieval
``ExecutionGraph`` with a single concurrency number. It has no durations, claims, conflict keys or capacities and
no notion of static lanes with gates, so it cannot be put behind this contract without losing those. Nothing outside
``context_runtime/qa/`` is changed.

Selection is deterministic: list scheduling by priority (longest remaining path, then risk, then duration, then
identifier), a lane chosen by earliest start with the lowest lane index on a tie, candidates compared by
(estimated makespan, lanes used, lane count). The same inputs give the same plan and the same plan hash.
"""
from __future__ import annotations

import heapq
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, Sequence, runtime_checkable

from ._contracts import content_hash, qa as contracts
from .closure import Closure, NodeKind, QAGraph, resolve
from .cost import CostModel, Estimate, bottom_levels, critical_path
from .errors import Code, InfeasibleBudget, PlanningRefusal, Problem, QAPlanningError, UnsupportedCoverage

PLANNER_ID = "cr-qa-planner"
PLANNER_VERSION = "0.1.0"
"""The algorithm version. It is part of the plan's identity: change it when the same input may yield a different plan."""

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_EVIDENCE_ORDER = {"MINIMAL": 0, "STANDARD": 1, "FULL": 2}


def _refuse(code: str, subject: str, detail: str) -> PlanningRefusal:
    return PlanningRefusal(code, detail, (Problem(code, subject, detail),))


def _int(name: str, value: object, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (
            maximum is not None and value > maximum):
        rng = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise _refuse(Code.INVALID_REQUEST, name, f"{name} is an integer {rng}")
    return value


@dataclass(frozen=True)
class RunCaps:
    """Ceilings no item of the plan may exceed. Model use is off unless a cap grants it."""

    deadline_ms: int
    max_retries: int = 0
    max_recovery_attempts: int = 0
    max_model_calls: int = 0
    max_tokens: int = 0
    recovery_ceiling: int = 0
    max_evidence: str = "FULL"

    def __post_init__(self) -> None:
        _int("deadline_ms", self.deadline_ms, 1, 86400000)
        _int("max_retries", self.max_retries, 0, 1000)
        _int("max_recovery_attempts", self.max_recovery_attempts, 0, 1000)
        _int("max_model_calls", self.max_model_calls, 0, 100000)
        _int("max_tokens", self.max_tokens, 0, 100000000)
        _int("recovery_ceiling", self.recovery_ceiling, 0, 1000)
        if self.recovery_ceiling < self.max_recovery_attempts:
            raise _refuse(Code.INVALID_REQUEST, "recovery_ceiling",
                          "the recovery ceiling is at least the cap on recovery attempts")
        level = getattr(self.max_evidence, "value", self.max_evidence)
        if level not in _EVIDENCE_ORDER:
            raise _refuse(Code.INVALID_REQUEST, "max_evidence", "max_evidence is MINIMAL, STANDARD or FULL")
        object.__setattr__(self, "max_evidence", level)

    def to_dict(self) -> dict:
        return {"deadline_ms": self.deadline_ms, "max_retries": self.max_retries,
                "max_recovery_attempts": self.max_recovery_attempts, "max_model_calls": self.max_model_calls,
                "max_tokens": self.max_tokens, "recovery_ceiling": self.recovery_ceiling,
                "max_evidence": self.max_evidence}


@dataclass(frozen=True)
class PlanBudget:
    """Limits the plan is built under."""

    max_parallel: int = 1
    """Cap on lanes, hence on tests running at once."""
    wall_time_ms: int | None = None
    """Cap on the estimated makespan; ``None`` means no cap."""
    max_cost_micros: int | None = None
    """Cap on the summed estimated cost; ``None`` means no cap."""
    capacity: tuple[tuple[str, int], ...] = ()
    """Pool capacities by resource name (``memory_mb``, ``cpu_millis``, ``browser_slots``, ...)."""
    deadline_multiplier_pct: int = 300
    """An item's deadline is its estimated duration times this percentage, never above the run cap."""
    full_evidence_from_bp: int = 7000
    """Items at or above this risk get FULL evidence (within the cap); the rest get STANDARD."""

    def __post_init__(self) -> None:
        _int("max_parallel", self.max_parallel, 1, 1000)
        if self.wall_time_ms is not None:
            _int("wall_time_ms", self.wall_time_ms, 1)
        if self.max_cost_micros is not None:
            _int("max_cost_micros", self.max_cost_micros, 0)
        _int("deadline_multiplier_pct", self.deadline_multiplier_pct, 100, 100000)
        _int("full_evidence_from_bp", self.full_evidence_from_bp, 0, 10001)
        pairs = tuple(sorted(dict(self.capacity).items())) if isinstance(self.capacity, Mapping) else tuple(
            sorted((str(k), v) for k, v in self.capacity))
        names = [k for k, _ in pairs]
        if len(set(names)) != len(names):
            raise _refuse(Code.INVALID_REQUEST, "capacity", "a capacity resource is listed more than once")
        for name, amount in pairs:
            if not _NAME.match(name):
                raise _refuse(Code.INVALID_REQUEST, name, "a resource name matches [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
            _int(f"capacity[{name}]", amount, 1)
        object.__setattr__(self, "capacity", pairs)

    def to_dict(self) -> dict:
        return {"max_parallel": self.max_parallel, "wall_time_ms": self.wall_time_ms,
                "max_cost_micros": self.max_cost_micros, "capacity": {k: v for k, v in self.capacity},
                "deadline_multiplier_pct": self.deadline_multiplier_pct,
                "full_evidence_from_bp": self.full_evidence_from_bp}


@dataclass(frozen=True)
class PlanRequest:
    """Everything a planner is given. Plain data, so any planner can be run on the same request."""

    graph: QAGraph
    targets: tuple
    """The selected tests, in the order the caller gave them (a planner that does no planning may keep this order)."""
    estimates: CostModel
    budget: PlanBudget
    caps: RunCaps
    coverage_contract: str = "FULL"
    specs: Mapping[str, object] | None = None
    """Optional ``TestSpec`` objects by id. When given, an item's budget is also bounded by its spec's budget."""
    action_policy_ref: str | None = None
    """Optional reference to the action policy the plan is to be run under (hashed into the plan)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))


@runtime_checkable
class QAPlanner(Protocol):
    """The interface every planner implements. Whatever strategy it uses, it returns the same type: a canonical
    ``QAPlan``. A planner that keeps the given order in a single lane implements this as readily as the cost-based one."""

    id: str
    version: str

    def plan(self, request: PlanRequest) -> "contracts.QAPlan": ...


# ── results ────────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScheduledItem:
    """One test as the planner placed it, with what decided its placement."""

    position: int
    """Order in which the planner placed the item."""
    test_id: str
    spec_ref: str
    lane: str
    order: int
    start_ms: int
    finish_ms: int
    ready_ms: int
    """Earliest start allowed by the tests it follows."""
    conflict_ms: int
    """Earliest start allowed by earlier tests it conflicts with."""
    lane_starts: tuple[int, ...]
    """For each lane considered, the earliest start there."""
    bound_by: str
    """What fixed the start: ``precedence``, ``conflict``, ``lane`` or ``none`` (start at zero)."""
    gate_after: tuple[str, ...]
    """Test ids of the gate emitted for this item (empty when lane order already implies every start condition)."""


@dataclass(frozen=True)
class Candidate:
    lanes: int
    makespan_ms: int | None
    lanes_used: int | None
    status: str
    """``selected``, ``rejected`` or ``unavailable``."""
    reason: str
    """A stable code: SELECTED, SLOWER, NO_FASTER_MORE_LANES, WALL_TIME_EXCEEDED, CAPACITY_BOUND, MAX_PARALLEL,
    MORE_LANES_THAN_TESTS."""
    detail: str = ""


@dataclass(frozen=True)
class PlanResult:
    plan: "contracts.QAPlan"
    closure: Closure
    schedule: tuple[ScheduledItem, ...]
    makespan_ms: int
    chosen_lanes: int
    candidates: tuple[Candidate, ...]
    limits: Mapping[str, int]
    """Upper bounds on the lane count by source: ``max_parallel``, ``tests``, ``capacity:<resource>``."""
    estimates: Mapping[str, Estimate]
    critical_path_ms: int
    critical_path: tuple[str, ...]
    total_duration_ms: int
    total_cost_micros: int
    request: PlanRequest
    request_digest: str

    @property
    def plan_hash(self) -> str:
        return self.plan.plan_hash


# ── the planner ────────────────────────────────────────────────────────────────────────────────────────────

class CostBasedQAPlanner:
    """Cost-based planner: closure, list scheduling under conflict keys and capacities, candidate lane counts."""

    id = PLANNER_ID
    version = PLANNER_VERSION

    def plan(self, request: PlanRequest) -> "contracts.QAPlan":
        return self.plan_detailed(request).plan

    def plan_detailed(self, request: PlanRequest) -> PlanResult:
        _require_full_request(request)
        closure = resolve(request.graph, request.targets)
        estimates = request.estimates.require(closure)
        pins = _require_pins(closure)
        _check_specs(request, closure)
        usage, locks = _usage(closure, request.budget)
        total_cost = sum(e.cost_micros for e in estimates.values())
        total_duration = sum(e.duration_ms for e in estimates.values())
        cap = request.budget.max_cost_micros
        if cap is not None and total_cost > cap:
            raise InfeasibleBudget(
                f"the summed estimated cost {total_cost} exceeds max_cost_micros {cap}; every test must run, so "
                "none is dropped", (Problem(Code.BUDGET_INFEASIBLE, "max_cost_micros",
                                            f"estimated {total_cost} > cap {cap}"),), binding="max_cost_micros")
        budgets = _item_budgets(request, closure, estimates)
        limits = _lane_limits(request.budget, closure, usage)
        p_max = max(1, min(limits.values()))
        cp_ms, cp_path = critical_path(closure, estimates)

        runs = {}
        for lanes in range(1, p_max + 1):
            runs[lanes] = _list_schedule(closure, estimates, lanes)
            if runs[lanes].lanes_used < lanes:
                break          # a lane stayed empty, so every larger count would be scheduled identically
        beyond = None
        if max(runs) == p_max and p_max < len(closure.test_ids):
            beyond = _list_schedule(closure, estimates, p_max + 1)   # what the lane limit rules out, for the explanation
        wall = request.budget.wall_time_ms
        feasible = [n for n, r in runs.items() if wall is None or r.makespan <= wall]
        if not feasible:
            best = min(runs.values(), key=lambda r: r.makespan)
            binding = "wall_time_ms"
            raise InfeasibleBudget(
                f"no lane count up to {p_max} fits wall_time_ms {wall}: the best estimated makespan is "
                f"{best.makespan} ms with {best.lanes_used} lane(s); every test must run, so none is dropped",
                (Problem(Code.BUDGET_INFEASIBLE, "wall_time_ms",
                         f"best estimated makespan {best.makespan} > cap {wall}; parallelism limited to {p_max} "
                         f"by {_limiting(limits, p_max)}"),),
                binding=binding,
                candidates=tuple((n, runs[n].makespan) for n in sorted(runs)))
        chosen = min(feasible, key=lambda n: (runs[n].makespan, runs[n].lanes_used, n))
        run = runs[chosen]
        candidates = _candidates(closure, runs, beyond, limits, p_max, chosen, wall)

        schedule, plan_items, gates = _materialize(closure, run, pins, budgets, estimates, request)
        plan = _emit(request, closure, plan_items, gates, run.lanes_used, usage, locks)
        result = PlanResult(
            plan=plan, closure=closure, schedule=schedule, makespan_ms=run.makespan, chosen_lanes=run.lanes_used,
            candidates=candidates, limits=dict(limits), estimates=estimates, critical_path_ms=cp_ms,
            critical_path=cp_path, total_duration_ms=total_duration, total_cost_micros=total_cost, request=request,
            request_digest=request_digest(request, closure))
        problems = verify_plan(plan, closure, budget=request.budget, caps=request.caps)
        if problems:
            raise PlanningRefusal(Code.PLAN_INVALID, "the planner's own validator rejected the plan it built",
                                  tuple(problems))
        return result


def request_digest(request: PlanRequest, closure: Closure) -> str:
    """``rcv1`` hash of the inputs: closure, estimates, budgets and caps."""
    return content_hash({
        "closure": closure.digest(), "estimates": request.estimates.to_dict(), "budget": request.budget.to_dict(),
        "caps": request.caps.to_dict(), "action_policy_ref": request.action_policy_ref,
        "planner": {"id": PLANNER_ID, "version": PLANNER_VERSION},
    })


def _require_full_request(request: PlanRequest) -> None:
    value = getattr(request.coverage_contract, "value", request.coverage_contract)
    if value != "FULL":
        raise UnsupportedCoverage(str(value))


def _require_pins(closure: Closure) -> dict[str, str]:
    pins: dict[str, str] = {}
    missing: list[Problem] = []
    for n in closure.tests:
        if not n.pin:
            missing.append(Problem(Code.MISSING_PIN, n.key, "a planned test carries its spec hash as its pin"))
        else:
            pins[n.id] = n.pin
    if missing:
        raise PlanningRefusal(Code.MISSING_PIN, f"{len(missing)} test(s) have no spec hash", tuple(missing))
    return pins


def _check_specs(request: PlanRequest, closure: Closure) -> None:
    if request.specs is None:
        return
    problems: list[Problem] = []
    for n in closure.tests:
        spec = request.specs.get(n.id)
        if spec is None:
            problems.append(Problem(Code.INVALID_REQUEST, n.key, "no spec supplied for a planned test"))
        elif spec.spec_hash != n.pin:
            problems.append(Problem(Code.PIN_MISMATCH, n.key,
                                    f"the supplied spec hashes to {spec.spec_hash}, the graph pins {n.pin}"))
    if problems:
        raise PlanningRefusal(Code.INVALID_REQUEST, "the supplied specs do not match the graph", tuple(problems))


def _usage(closure: Closure, budget: PlanBudget):
    """Pool usage per test and the set of exclusive lock resources. A claim that is neither exclusive nor against a
    declared capacity is refused: nothing bounds it."""
    capacity = dict(budget.capacity)
    usage: dict[str, dict[str, int]] = {}
    locks: set[str] = set()
    problems: list[Problem] = []
    for t in closure.test_ids:
        row: dict[str, int] = {}
        for c in closure.claims(t):
            if c.exclusive:
                locks.add(c.resource)
            if c.resource in capacity:
                row[c.resource] = c.amount
            elif not c.exclusive:
                problems.append(Problem(Code.UNDECLARED_RESOURCE, f"test:{t}",
                                        f"claims {c.amount} of {c.resource}, which is neither exclusive nor a "
                                        "declared capacity"))
        usage[t] = row
    if problems:
        raise PlanningRefusal(Code.UNDECLARED_RESOURCE, f"{len(problems)} claim(s) have no declared capacity",
                              tuple(problems))
    return usage, locks


def _lane_limits(budget: PlanBudget, closure: Closure, usage: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """Upper bounds on the lane count. For a pool, ``p`` lanes are allowed only if the ``p`` largest claims together
    fit the capacity: at most ``p`` tests run at once, so no mix of them can overcommit it."""
    n = len(closure.test_ids)
    limits = {"max_parallel": budget.max_parallel, "tests": n}
    for resource, cap in budget.capacity:
        amounts = sorted((u.get(resource, 0) for u in usage.values()), reverse=True)
        if amounts and amounts[0] > cap:
            big = next(t for t, u in usage.items() if u.get(resource, 0) == amounts[0])
            raise InfeasibleBudget(
                f"test {big} alone claims {amounts[0]} of {resource}, above the capacity {cap}; every test must run",
                (Problem(Code.BUDGET_INFEASIBLE, f"test:{big}", f"claims {amounts[0]} > capacity {cap} of {resource}"),),
                binding=f"capacity:{resource}")
        total = 0
        allowed = 0
        for a in amounts:
            if total + a > cap:
                break
            total += a
            allowed += 1
        limits[f"capacity:{resource}"] = max(allowed, 1) if allowed < len(amounts) else n
    return limits


def _limiting(limits: Mapping[str, int], p_max: int) -> str:
    return min(k for k, v in limits.items() if v == p_max)


def _item_budgets(request: PlanRequest, closure: Closure, estimates: Mapping[str, Estimate]) -> dict[str, tuple]:
    """Per-test (deadline, retries, recovery attempts, model calls, tokens, recovery ceiling, evidence level)."""
    caps, budget = request.caps, request.budget
    out: dict[str, tuple] = {}
    problems: list[Problem] = []
    for t in closure.test_ids:
        est = estimates[t]
        spec = request.specs.get(t) if request.specs else None
        sb = spec.budget if spec is not None else None
        deadline_cap = min(caps.deadline_ms, sb.deadline_ms) if sb else caps.deadline_ms
        if est.duration_ms > deadline_cap:
            problems.append(Problem(Code.BUDGET_INFEASIBLE, f"test:{t}",
                                    f"estimated duration {est.duration_ms} ms exceeds the deadline cap {deadline_cap} ms"))
            continue
        wanted = -(-est.duration_ms * budget.deadline_multiplier_pct // 100)
        deadline = min(deadline_cap, max(est.duration_ms, wanted))
        retries = min(caps.max_retries, sb.max_retries) if sb else caps.max_retries
        recovery = min(caps.max_recovery_attempts, sb.max_recovery_attempts) if sb else caps.max_recovery_attempts
        calls = min(caps.max_model_calls, sb.max_model_calls) if sb else caps.max_model_calls
        tokens = min(caps.max_tokens, sb.max_tokens) if sb else caps.max_tokens
        level = "FULL" if est.risk_bp >= budget.full_evidence_from_bp else "STANDARD"
        if _EVIDENCE_ORDER[level] > _EVIDENCE_ORDER[caps.max_evidence]:
            level = caps.max_evidence
        out[t] = (deadline, retries, recovery, calls, tokens, caps.recovery_ceiling, level)
    if problems:
        raise InfeasibleBudget("a test cannot finish inside its deadline cap; every test must run, so none is dropped",
                               tuple(problems), binding="deadline_ms")
    return out


@dataclass
class _Run:
    lanes: int
    lanes_used: int
    makespan: int
    order: list           # test ids in placement order
    lane_of: dict
    start: dict
    finish: dict
    detail: dict          # test id -> (ready, conflict, lane_starts, bound_by)
    lane_items: list      # per lane, test ids in lane order


def _conflicting_earlier(closure: Closure, test_id: str, placed: Iterable[str]) -> list[str]:
    """Already placed tests that conflict with ``test_id`` (their claims meet on an exclusive resource)."""
    return [j for j in placed if closure.conflicts(test_id, j)]


def _list_schedule(closure: Closure, estimates: Mapping[str, Estimate], lanes: int) -> _Run:
    level = bottom_levels(closure, estimates)
    ids = closure.test_ids
    preds = {t: closure.predecessors(t) for t in ids}
    succ: dict[str, list[str]] = {t: [] for t in ids}
    for t in ids:
        for p in preds[t]:
            succ[p].append(t)

    def prio(t: str) -> tuple:
        e = estimates[t]
        return (-level[t], -e.risk_bp, -e.duration_ms, t)

    waiting = {t: len(preds[t]) for t in ids}
    heap = [(prio(t), t) for t in ids if waiting[t] == 0]
    heapq.heapify(heap)
    lane_free = [0] * lanes
    lane_items: list[list[str]] = [[] for _ in range(lanes)]
    order: list[str] = []
    lane_of: dict[str, int] = {}
    start: dict[str, int] = {}
    finish: dict[str, int] = {}
    detail: dict[str, tuple] = {}
    while heap:
        _, t = heapq.heappop(heap)
        ready = max((finish[p] for p in preds[t]), default=0)
        conflict = max((finish[j] for j in _conflicting_earlier(closure, t, order)), default=0)
        floor = max(ready, conflict)
        starts = tuple(max(free, floor) for free in lane_free)
        lane = min(range(lanes), key=lambda k: (starts[k], k))
        s = starts[lane]
        bound = "none" if s == 0 else ("precedence" if s == ready else "conflict" if s == conflict else "lane")
        start[t], finish[t] = s, s + estimates[t].duration_ms
        lane_free[lane] = finish[t]
        lane_items[lane].append(t)
        lane_of[t] = lane
        order.append(t)
        detail[t] = (ready, conflict, starts, bound)
        for nxt in succ[t]:
            waiting[nxt] -= 1
            if waiting[nxt] == 0:
                heapq.heappush(heap, (prio(nxt), nxt))
    if len(order) != len(ids):  # unreachable for an acyclic closure; refuse rather than emit a partial plan
        raise _refuse(Code.PLAN_INVALID, "schedule", "the scheduler could not place every test")
    used = sum(1 for items in lane_items if items)
    return _Run(lanes, used, max(finish.values()), order, lane_of, start, finish, detail, lane_items)


def _candidates(closure: Closure, runs: Mapping[int, _Run], beyond: "_Run | None", limits: Mapping[str, int],
                p_max: int, chosen: int, wall: int | None) -> tuple[Candidate, ...]:
    """Every candidate that was scheduled, plus the first one the lane limit ruled out (with the makespan it would
    have had) and, when a lane stayed empty, one entry standing for all larger counts."""
    best = runs[chosen]
    out: list[Candidate] = []
    for lanes in sorted(runs):
        r = runs[lanes]
        if lanes == chosen:
            out.append(Candidate(lanes, r.makespan, r.lanes_used, "selected", "SELECTED",
                                 "lowest estimated makespan among feasible candidates, fewest lanes on a tie"))
        elif wall is not None and r.makespan > wall:
            out.append(Candidate(lanes, r.makespan, r.lanes_used, "rejected", "WALL_TIME_EXCEEDED",
                                 f"estimated makespan {r.makespan} ms exceeds wall_time_ms {wall}"))
        elif r.makespan == best.makespan:
            out.append(Candidate(lanes, r.makespan, r.lanes_used, "rejected", "NO_FASTER_MORE_LANES",
                                 f"same estimated makespan {r.makespan} ms with {r.lanes_used} lane(s) used "
                                 f"against {best.lanes_used}"))
        else:
            over = (r.makespan - best.makespan) * 10000 // best.makespan
            out.append(Candidate(lanes, r.makespan, r.lanes_used, "rejected", "SLOWER",
                                 f"estimated makespan {r.makespan} ms, {over} basis points above the chosen "
                                 f"{best.makespan} ms"))
    last = max(runs)
    if runs[last].lanes_used < last:
        out.append(Candidate(last + 1, runs[last].makespan, runs[last].lanes_used, "rejected", "NO_FASTER_MORE_LANES",
                             f"lane counts above {last} leave lanes unused and give the same plan"))
    elif beyond is not None:
        src = min((k for k, v in limits.items() if v == p_max and k != "tests"), default="max_parallel")
        code = "CAPACITY_BOUND" if src.startswith("capacity:") else "MAX_PARALLEL"
        out.append(Candidate(last + 1, beyond.makespan, beyond.lanes_used, "unavailable", code,
                             f"{last + 1} lanes would exceed the limit {p_max} set by {src}; the estimated makespan "
                             f"would have been {beyond.makespan} ms"))
    return tuple(out)


def _materialize(closure: Closure, run: _Run, pins: Mapping[str, str], budgets: Mapping[str, tuple],
                 estimates: Mapping[str, Estimate], request: PlanRequest):
    """Lane names, item records, and the minimal gates that make precedence and conflicts hold structurally."""
    lane_names = {}
    for index, items in enumerate(i for i in run.lane_items if i):
        for t in items:
            lane_names[t] = (f"lane-{index}", items.index(t))
    pos = {t: n for n, t in enumerate(run.order)}
    anc: dict[str, int] = {}
    gate_after: dict[str, tuple[str, ...]] = {}
    lane_prev: dict[str, str | None] = {}
    for items in run.lane_items:
        for k, t in enumerate(items):
            lane_prev[t] = items[k - 1] if k else None
    for t in run.order:
        base = 0
        prev = lane_prev[t]
        if prev is not None:
            base = anc[prev] | (1 << pos[prev])
        required = set(closure.predecessors(t)) | set(_conflicting_earlier(closure, t, run.order[:pos[t]]))
        chosen: list[str] = []
        for r in sorted(required, key=lambda x: -pos[x]):
            if not (base >> pos[r]) & 1:
                chosen.append(r)
                base |= anc[r] | (1 << pos[r])
        anc[t] = base
        gate_after[t] = tuple(sorted(chosen))
    ref = {t: contracts.make_spec_ref(t, pins[t]) for t in closure.test_ids}
    schedule = []
    items = []
    for n, t in enumerate(run.order):
        ready, conflict, starts, bound = run.detail[t]
        lane, order = lane_names[t]
        schedule.append(ScheduledItem(n, t, ref[t], lane, order, run.start[t], run.finish[t], ready, conflict,
                                      starts, bound, gate_after[t]))
        deadline, retries, recovery, calls, tokens, ceiling, level = budgets[t]
        items.append(_plan_item(ref[t], lane, order, deadline, retries, recovery, calls, tokens, ceiling, level))
    gates = [contracts.PlanGate(item=ref[t], after=tuple(ref[a] for a in after))
             for t, after in gate_after.items() if after]
    return tuple(schedule), items, gates


def _plan_item(spec_ref, lane, order, deadline, retries, recovery, calls, tokens, ceiling, level):
    try:
        return contracts.PlanItem(
            spec_ref=spec_ref, lane=lane, order=order,
            budget=contracts.Budget(deadline_ms=deadline, max_retries=retries, max_recovery_attempts=recovery,
                                    max_model_calls=calls, max_tokens=tokens),
            recovery_ceiling=ceiling, evidence_level=contracts.EvidenceLevel(level))
    except contracts.QAContractError as exc:
        raise _refuse(Code.PLAN_INVALID, spec_ref, f"the contracts validator rejected an item: {exc}") from exc


def _emit(request: PlanRequest, closure: Closure, items, gates, lanes_used: int, usage, locks):
    claims: dict[str, int] = {r: a for r, a in request.budget.capacity}
    for lock in sorted(locks):
        claims.setdefault(lock, 1)
    try:
        return contracts.QAPlan(
            items=tuple(items), lanes=tuple(f"lane-{i}" for i in range(lanes_used)), max_parallel=lanes_used,
            coverage_contract=contracts.CoverageContract.FULL,
            planner=contracts.ComponentRef(id=PLANNER_ID, version=PLANNER_VERSION), gates=tuple(gates),
            resource_claims=tuple(contracts.ResourceClaim(resource=r, amount=a) for r, a in sorted(claims.items())),
            action_policy_ref=request.action_policy_ref)
    except contracts.QAContractError as exc:
        raise _refuse(Code.PLAN_INVALID, "plan", f"the contracts validator rejected the plan: {exc}") from exc


# ── the planner's own validator ────────────────────────────────────────────────────────────────────────────

def _happens_before(plan) -> tuple[dict[str, int], dict[str, int]]:
    """For each item, a bitmask of the items that must have finished before it starts (lane order and gates, closed
    transitively), and the bit index of each item. Computed from the plan alone."""
    refs = sorted(i.spec_ref for i in plan.items)
    index = {r: n for n, r in enumerate(refs)}
    direct: dict[str, set[str]] = {r: set() for r in refs}
    by_lane: dict[str, list] = {}
    for item in plan.items:
        by_lane.setdefault(item.lane, []).append(item)
    for items in by_lane.values():
        items.sort(key=lambda i: i.order)
        for a, b in zip(items, items[1:]):
            direct[b.spec_ref].add(a.spec_ref)
    for g in plan.gates:
        direct[g.item].update(g.after)
    waiting = {r: len(direct[r]) for r in refs}
    dependents: dict[str, list[str]] = {r: [] for r in refs}
    for r in refs:
        for d in direct[r]:
            dependents[d].append(r)
    ready = [r for r in refs if waiting[r] == 0]
    before: dict[str, int] = {}
    while ready:
        cur = ready.pop()
        mask = 0
        for d in direct[cur]:
            mask |= before[d] | (1 << index[d])
        before[cur] = mask
        for nxt in dependents[cur]:
            waiting[nxt] -= 1
            if waiting[nxt] == 0:
                ready.append(nxt)
    if len(before) != len(refs):
        raise _refuse(Code.PLAN_INVALID, "plan", "lane order and gates form a cycle")
    return before, index


def _ordered(before: Mapping[str, int], index: Mapping[str, int], a: str, b: str) -> bool:
    return bool((before[b] >> index[a]) & 1) or bool((before[a] >> index[b]) & 1)


def _require_full_coverage(plan, closure: Closure, problems: list[Problem]) -> None:
    want = {contracts.make_spec_ref(n.id, n.pin) for n in closure.tests if n.pin}
    have = [i.spec_ref for i in plan.items]
    for ref in sorted(want - set(have)):
        problems.append(Problem(Code.PLAN_INVALID, ref, "a test of the closure is missing from the plan (coverage "
                                "must be FULL)"))
    for ref in sorted(set(have) - want):
        problems.append(Problem(Code.PLAN_INVALID, ref, "an item is in the plan but not in the closure"))
    if len(have) != len(set(have)):
        problems.append(Problem(Code.PLAN_INVALID, "items", "a test appears more than once"))


def verify_plan(plan, closure: Closure, *, budget: PlanBudget | None = None, caps: RunCaps | None = None) -> list[Problem]:
    """Check a plan against the closure it was built for, independently of how it was built.

    Returns the problems found; an empty list means the plan is acceptable. It checks: ``coverage_contract`` is FULL;
    every test of the closure is planned exactly once and nothing else is; every precedence and every conflict is
    ordered by lane order or a gate; the lane count is within ``max_parallel`` and the pool capacities; every item's
    budget is within the run caps.
    """
    problems: list[Problem] = []
    if plan.coverage_contract is not contracts.CoverageContract.FULL:
        problems.append(Problem(Code.UNSUPPORTED_COVERAGE_CONTRACT, "coverage_contract",
                                f"{plan.coverage_contract.value} is not supported; only FULL is accepted"))
        return problems
    _require_full_coverage(plan, closure, problems)
    if problems:
        return problems
    ref = {n.id: contracts.make_spec_ref(n.id, n.pin) for n in closure.tests}
    try:
        packed, index = _happens_before(plan)
    except QAPlanningError as exc:
        return [*problems, *exc.problems]
    for t in closure.test_ids:
        for p in closure.predecessors(t):
            if not (packed[ref[t]] >> index[ref[p]]) & 1:
                problems.append(Problem(Code.PLAN_INVALID, ref[t], f"does not wait for {ref[p]}, which it depends on"))
    ids = closure.test_ids
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if closure.conflicts(a, b) and not _ordered(packed, index, ref[a], ref[b]):
                problems.append(Problem(Code.PLAN_INVALID, f"{ref[a]},{ref[b]}",
                                        "tests that claim the same exclusive resource are not ordered, so they could "
                                        "run at the same time"))
    if budget is not None:
        if plan.max_parallel > budget.max_parallel or len(plan.lanes) > budget.max_parallel:
            problems.append(Problem(Code.PLAN_INVALID, "max_parallel",
                                    f"{len(plan.lanes)} lanes exceed the budget's max_parallel {budget.max_parallel}"))
        for resource, cap in budget.capacity:
            amounts = sorted((next((c.amount for c in closure.claims(t) if c.resource == resource), 0)
                              for t in ids), reverse=True)
            worst = sum(amounts[:len(plan.lanes)])
            if worst > cap:
                problems.append(Problem(Code.PLAN_INVALID, f"capacity:{resource}",
                                        f"{len(plan.lanes)} concurrent tests could claim {worst} of {resource}, "
                                        f"above the capacity {cap}"))
    if caps is not None:
        for item in plan.items:
            b = item.budget
            over = [n for n, v, c in (("deadline_ms", b.deadline_ms, caps.deadline_ms),
                                      ("max_retries", b.max_retries, caps.max_retries),
                                      ("max_recovery_attempts", b.max_recovery_attempts, caps.max_recovery_attempts),
                                      ("max_model_calls", b.max_model_calls, caps.max_model_calls),
                                      ("max_tokens", b.max_tokens, caps.max_tokens),
                                      ("recovery_ceiling", item.recovery_ceiling, caps.recovery_ceiling)) if v > c]
            if _EVIDENCE_ORDER[item.evidence_level.value] > _EVIDENCE_ORDER[caps.max_evidence]:
                over.append("evidence_level")
            if over:
                problems.append(Problem(Code.PLAN_INVALID, item.spec_ref, f"above the run caps: {over}"))
    return problems


def require_valid_plan(plan, closure: Closure, **kw) -> None:
    problems = verify_plan(plan, closure, **kw)
    if problems:
        raise PlanningRefusal(problems[0].code, f"{len(problems)} problem(s) in the plan", tuple(problems))
