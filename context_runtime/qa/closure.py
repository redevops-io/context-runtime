"""Dependency closure over a declared graph of tests, fixtures, environment profiles and resource claims.

The graph is *declared*: nodes say what they require and what they claim, and nothing is discovered at run time.
``resolve`` turns a set of selected tests into the minimal closure that has to be scheduled with them, in a
deterministic dependency order, with an explicit reason for every included node.

Properties, all covered by tests:

* **Minimal.** Only nodes reachable from the selected tests are in the closure.
* **Deterministic.** The order is a lexicographic topological order (kind rank, then identifier by Unicode code
  point). It does not depend on input order, hash seed, locale or process.
* **Explained.** Every node carries the reasons it is included: selected, or required by which node through which
  relation.
* **Fail closed.** An unknown reference, a pin that does not match the declared node, a cycle (named) or a
  duplicate declaration is refused with every finding listed. Nothing is skipped or guessed.
* **Conflict keys.** A test's effective claims are its own plus those of every non-test node it needs. A claim on
  a resource that is exclusive is a conflict key: two tests whose claims meet on one exclusive resource must never
  run at the same time.
"""
from __future__ import annotations

import heapq
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from ._contracts import content_hash
from .errors import ClosureError, Code, Problem


class NodeKind(str, Enum):
    ENVIRONMENT = "environment"
    FIXTURE = "fixture"
    SETUP = "setup"
    TEST = "test"


_RANK = {NodeKind.ENVIRONMENT: 0, NodeKind.FIXTURE: 1, NodeKind.SETUP: 2, NodeKind.TEST: 3}
_RESOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")

NodeId = tuple  # (NodeKind, str)


def node_key(kind: NodeKind, ident: str) -> str:
    """The display form of a node: ``kind:identifier``."""
    return f"{kind.value}:{ident}"


def _sort_key(kind: NodeKind, ident: str) -> tuple:
    return (_RANK[kind], ident)


def _invalid(subject: str, detail: str) -> ClosureError:
    return ClosureError(Code.INVALID_GRAPH, detail, (Problem(Code.INVALID_GRAPH, subject, detail),))


def _check_ident(subject: str, ident: object) -> str:
    if not isinstance(ident, str) or not ident or len(ident) > 512 or _CONTROL.search(ident) \
            or unicodedata.normalize("NFC", ident) != ident:
        raise _invalid(subject, "an identifier is a non-empty NFC string of at most 512 characters with no "
                       "control character")
    return ident


@dataclass(frozen=True)
class Claim:
    """A claim a node makes on a named resource (``browser_slots``, ``memory_mb``, a fixture lock, ...)."""

    resource: str
    amount: int = 1
    exclusive: bool = False
    """An exclusive claim is a conflict key: no other claimant of the resource may run at the same time."""

    def __post_init__(self) -> None:
        if not isinstance(self.resource, str) or not _RESOURCE.match(self.resource):
            raise _invalid(str(self.resource), "a resource name matches [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount < 1:
            raise _invalid(self.resource, "a claim amount is a positive integer")
        if not isinstance(self.exclusive, bool):
            raise _invalid(self.resource, "exclusive is a boolean")


@dataclass(frozen=True)
class Ref:
    """A reference to a declared node, optionally pinned to the content hash the node must have."""

    kind: NodeKind
    id: str
    pin: str | None = None
    relation: str = "requires"
    """Why the reference exists (``depends_on``, ``fixture``, ``environment``, ``precondition``, ...). Used in the
    explanation, not in identity."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", NodeKind(self.kind))
        _check_ident("ref", self.id)
        if self.pin is not None and (not isinstance(self.pin, str) or not self.pin):
            raise _invalid(self.id, "a pin is a non-empty string or absent")


@dataclass(frozen=True)
class Node:
    """A declared test, fixture, environment profile or setup step."""

    kind: NodeKind
    id: str
    pin: str | None = None
    """The node's content hash (a test's spec hash, a fixture's content hash). A pinned reference must match it."""
    requires: tuple[Ref, ...] = ()
    claims: tuple[Claim, ...] = ()
    exclusive: bool = False
    """Shorthand for an exclusive claim on the resource ``<kind>:<id>``: the node is a lock."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", NodeKind(self.kind))
        subject = f"{self.kind.value}:{self.id}"
        _check_ident(subject, self.id)
        if self.pin is not None and (not isinstance(self.pin, str) or not self.pin):
            raise _invalid(subject, "a pin is a non-empty string or absent")
        if not isinstance(self.exclusive, bool):
            raise _invalid(subject, "exclusive is a boolean")
        reqs = tuple(sorted(set(self.requires), key=lambda r: (_sort_key(r.kind, r.id), r.pin or "", r.relation)))
        object.__setattr__(self, "requires", reqs)
        claims = tuple(sorted(self.claims, key=lambda c: c.resource))
        names = [c.resource for c in claims]
        if len(set(names)) != len(names):
            raise _invalid(subject, "a node lists a resource more than once; merge the claims")
        object.__setattr__(self, "claims", claims)

    @property
    def key(self) -> str:
        return node_key(self.kind, self.id)

    def own_claims(self) -> tuple[Claim, ...]:
        """Declared claims plus the implicit exclusive claim of a lock node."""
        out = list(self.claims)
        if self.exclusive:
            lock = f"{self.kind.value}:{self.id}"
            if _RESOURCE.match(lock) and all(c.resource != lock for c in out):
                out.append(Claim(lock, 1, True))
        return tuple(sorted(out, key=lambda c: c.resource))


class QAGraph:
    """A declared graph. Constructing it refuses duplicate declarations; it does not check references (``resolve``
    does, for the part of the graph a selection reaches)."""

    def __init__(self, nodes: Iterable[Node]):
        seen: dict[tuple, Node] = {}
        problems: list[Problem] = []
        for n in nodes:
            if not isinstance(n, Node):
                raise _invalid("graph", "a graph is built from Node objects")
            ident = (n.kind, n.id)
            if ident in seen:
                problems.append(Problem(Code.DUPLICATE_NODE, n.key, "declared more than once"))
            else:
                seen[ident] = n
        if problems:
            problems.sort(key=lambda p: p.subject)
            raise ClosureError(Code.DUPLICATE_NODE, "a node is declared more than once", tuple(problems))
        self._nodes = seen

    def get(self, kind: NodeKind, ident: str) -> Node | None:
        return self._nodes.get((NodeKind(kind), ident))

    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes[k] for k in sorted(self._nodes, key=lambda k: _sort_key(*k)))

    def __len__(self) -> int:
        return len(self._nodes)


def graph_from_specs(specs: Iterable, *, fixtures: Iterable[Node] = (), environments: Iterable[Node] = (),
                     setups: Iterable[Node] = (), claims: Mapping[str, Iterable[Claim]] | None = None) -> QAGraph:
    """Build the declared graph from ``TestSpec`` objects.

    Each spec becomes a test node pinned to its ``spec_hash``. It requires its ``depends_on`` specs (pinned to the
    hash in the reference), its ``fixtures`` (pinned to the fixture's content hash), its environment profile and
    each of its ``preconditions`` as a declared ``setup`` node. Fixtures, environment profiles and setup steps are
    declared by the caller (they carry the resource claims); one that a spec names and the caller did not declare is
    an unknown reference when the graph is resolved. ``claims`` adds claims to a test by spec id.
    """
    from ._contracts import qa as contracts

    extra = {k: tuple(v) for k, v in (claims or {}).items()}
    nodes: list[Node] = [*fixtures, *environments, *setups]
    for spec in specs:
        requires: list[Ref] = []
        for ref in spec.depends_on:
            dep_id, dep_hash = contracts.split_spec_ref(ref)
            requires.append(Ref(NodeKind.TEST, dep_id, dep_hash, "depends_on"))
        for fx in spec.fixtures:
            requires.append(Ref(NodeKind.FIXTURE, fx.id, fx.content_hash, "fixture"))
        requires.append(Ref(NodeKind.ENVIRONMENT, spec.target.environment_profile, None, "environment"))
        for pre in spec.preconditions:
            requires.append(Ref(NodeKind.SETUP, pre, None, "precondition"))
        nodes.append(Node(NodeKind.TEST, spec.id, spec.spec_hash, tuple(requires), extra.get(spec.id, ())))
    return QAGraph(nodes)


# ── the result ─────────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Reason:
    """Why a node is in the closure."""

    code: str
    """``selected`` (a target of the selection) or ``required_by``."""
    by: str = ""
    """For ``required_by``: the key of the requiring node."""
    relation: str = ""

    def text(self) -> str:
        if self.code == "selected":
            return "selected"
        return f"required by {self.by} ({self.relation})"

    def to_dict(self) -> dict:
        return {"code": self.code, "by": self.by, "relation": self.relation}


@dataclass(frozen=True)
class ClosureNode:
    kind: NodeKind
    id: str
    pin: str | None
    requires: tuple[str, ...]
    """Keys of the declared nodes this node requires."""
    reasons: tuple[Reason, ...]

    @property
    def key(self) -> str:
        return node_key(self.kind, self.id)

    def to_dict(self) -> dict:
        return {"key": self.key, "kind": self.kind.value, "id": self.id, "pin": self.pin,
                "requires": list(self.requires), "reasons": [r.to_dict() for r in self.reasons]}


@dataclass(frozen=True)
class EffectiveClaim:
    """What a test holds while it runs, summed over its own claims and those of the non-test nodes it needs."""

    resource: str
    amount: int
    exclusive: bool
    via: tuple[str, ...]
    """Keys of the nodes that contribute the claim."""

    def to_dict(self) -> dict:
        return {"resource": self.resource, "amount": self.amount, "exclusive": self.exclusive,
                "via": list(self.via)}


def claims_conflict(a: Sequence[EffectiveClaim], b: Sequence[EffectiveClaim]) -> bool:
    """True when two tests meet on a resource that either holds exclusively."""
    held = {c.resource: c.exclusive for c in a}
    for c in b:
        if c.resource in held and (held[c.resource] or c.exclusive):
            return True
    return False


@dataclass(frozen=True)
class Closure:
    targets: tuple[str, ...]
    nodes: tuple[ClosureNode, ...]
    """Every node of the closure, dependencies first."""
    _predecessors: Mapping[str, tuple[str, ...]] = field(repr=False, compare=False)
    _claims: Mapping[str, tuple[EffectiveClaim, ...]] = field(repr=False, compare=False)

    @property
    def tests(self) -> tuple[ClosureNode, ...]:
        return tuple(n for n in self.nodes if n.kind is NodeKind.TEST)

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.tests)

    def node(self, kind: NodeKind, ident: str) -> ClosureNode:
        for n in self.nodes:
            if n.kind is kind and n.id == ident:
                return n
        raise KeyError(node_key(kind, ident))

    def predecessors(self, test_id: str) -> tuple[str, ...]:
        """Tests that must finish before ``test_id`` starts: those it requires directly or through fixtures,
        environments and setup steps."""
        return self._predecessors[test_id]

    def claims(self, test_id: str) -> tuple[EffectiveClaim, ...]:
        return self._claims[test_id]

    def exclusive_keys(self, test_id: str) -> tuple[str, ...]:
        return tuple(c.resource for c in self._claims[test_id] if c.exclusive)

    def conflicts(self, a: str, b: str) -> bool:
        return a != b and claims_conflict(self._claims[a], self._claims[b])

    def to_dict(self) -> dict:
        return {
            "targets": list(self.targets),
            "nodes": [n.to_dict() for n in self.nodes],
            "predecessors": {t: list(self._predecessors[t]) for t in sorted(self._predecessors)},
            "claims": {t: [c.to_dict() for c in self._claims[t]] for t in sorted(self._claims)},
        }

    def digest(self) -> str:
        """``rcv1`` hash of the closure's content: the same selection over the same graph gives the same value."""
        return content_hash(self.to_dict())


# ── resolution ─────────────────────────────────────────────────────────────────────────────────────────────

_WHITE, _GREY, _BLACK = 0, 1, 2


def _normalize_targets(targets: Iterable) -> list[Ref]:
    out: list[Ref] = []
    for t in targets:
        if isinstance(t, Ref):
            if t.kind is not NodeKind.TEST:
                raise _invalid(t.id, "a target is a test")
            out.append(t)
        elif isinstance(t, str):
            out.append(Ref(NodeKind.TEST, t, None, "selected"))
        else:
            raise _invalid(str(t), "a target is a test id or a Ref to a test")
    return out


def resolve(graph: QAGraph, targets: Iterable) -> Closure:
    """The minimal, ordered closure of ``targets`` over ``graph``, or a ``ClosureError``."""
    refs = _normalize_targets(targets)
    if not refs:
        raise ClosureError(Code.EMPTY_SELECTION, "a selection names at least one test",
                           (Problem(Code.EMPTY_SELECTION, "targets", "no test selected"),))

    problems: list[Problem] = []
    target_ids: list[str] = []
    for r in sorted(refs, key=lambda r: (r.id, r.pin or "")):
        node = graph.get(NodeKind.TEST, r.id)
        if node is None:
            problems.append(Problem(Code.UNKNOWN_TARGET, node_key(NodeKind.TEST, r.id), "not a declared test"))
            continue
        if r.pin is not None and node.pin != r.pin:
            problems.append(Problem(Code.PIN_MISMATCH, node.key, f"selected at {r.pin}, declared at {node.pin}"))
        if r.id not in target_ids:
            target_ids.append(r.id)

    # Reachability, collecting reasons and every unknown reference or pin mismatch (not only the first).
    reasons: dict[tuple, list[Reason]] = {}
    declared: dict[tuple, Node] = {}
    queue: list[tuple] = []
    for tid in target_ids:
        ident = (NodeKind.TEST, tid)
        reasons.setdefault(ident, []).append(Reason("selected"))
        declared[ident] = graph.get(NodeKind.TEST, tid)  # type: ignore[assignment]
        queue.append(ident)
    seen_unknown: set[tuple] = set()
    i = 0
    while i < len(queue):
        ident = queue[i]
        i += 1
        node = declared[ident]
        for ref in node.requires:
            target = graph.get(ref.kind, ref.id)
            label = node_key(ref.kind, ref.id)
            if target is None:
                mark = (node.key, label)
                if mark not in seen_unknown:
                    seen_unknown.add(mark)
                    problems.append(Problem(Code.UNKNOWN_REFERENCE, node.key,
                                            f"requires {label} ({ref.relation}), which is not declared"))
                continue
            if ref.pin is not None and target.pin != ref.pin:
                problems.append(Problem(Code.PIN_MISMATCH, node.key,
                                        f"requires {label} ({ref.relation}) at {ref.pin}, "
                                        f"declared at {target.pin}"))
            tid2 = (target.kind, target.id)
            reasons.setdefault(tid2, []).append(Reason("required_by", node.key, ref.relation))
            if tid2 not in declared:
                declared[tid2] = target
                queue.append(tid2)

    cycles = _find_cycles(declared)
    for cyc in cycles:
        problems.append(Problem(Code.CYCLE, cyc[0], " -> ".join((*cyc, cyc[0]))))

    if problems:
        problems.sort(key=lambda p: (_PRIORITY.get(p.code, 9), p.subject, p.detail))
        first = problems[0]
        cycle = cycles[0] if first.code == Code.CYCLE and cycles else ()
        raise ClosureError(first.code, _summary(problems), tuple(problems), cycle=cycle)

    order = _topological(declared)
    nodes: list[ClosureNode] = []
    for ident in order:
        n = declared[ident]
        reqs = tuple(node_key(r.kind, r.id) for r in n.requires)
        rs = tuple(sorted(set(reasons[ident]), key=lambda r: (r.code, r.by, r.relation)))
        nodes.append(ClosureNode(n.kind, n.id, n.pin, tuple(sorted(set(reqs))), rs))

    preds, claims = _derive(order, declared)
    return Closure(tuple(sorted(target_ids)), tuple(nodes), MappingProxyType(preds), MappingProxyType(claims))


_PRIORITY = {Code.UNKNOWN_TARGET: 0, Code.UNKNOWN_REFERENCE: 1, Code.PIN_MISMATCH: 2, Code.CYCLE: 3}


def _summary(problems: Sequence[Problem]) -> str:
    return f"{len(problems)} problem(s): " + "; ".join(str(p) for p in problems[:5]) + (
        "; ..." if len(problems) > 5 else "")


def _find_cycles(declared: Mapping[tuple, Node]) -> list[tuple[str, ...]]:
    """Every cycle found by a depth-first search over the declared subgraph, each as node keys starting at the
    smallest key. Iterative, so a long chain cannot exhaust the interpreter stack."""
    color = {ident: _WHITE for ident in declared}
    found: dict[tuple[str, ...], None] = {}
    for root in sorted(declared, key=lambda k: _sort_key(*k)):
        if color[root] != _WHITE:
            continue
        stack: list[tuple[tuple, list[tuple]]] = []
        path: list[tuple] = []
        color[root] = _GREY
        path.append(root)
        stack.append((root, _successors(declared, root)))
        while stack:
            ident, succ = stack[-1]
            if not succ:
                color[ident] = _BLACK
                stack.pop()
                path.pop()
                continue
            nxt = succ.pop(0)
            state = color[nxt]
            if state == _GREY:
                ring = path[path.index(nxt):]
                keys = [node_key(*k) for k in ring]
                start = min(range(len(keys)), key=lambda j: keys[j])
                found[tuple(keys[start:] + keys[:start])] = None
            elif state == _WHITE:
                color[nxt] = _GREY
                path.append(nxt)
                stack.append((nxt, _successors(declared, nxt)))
    return sorted(found)


def _successors(declared: Mapping[tuple, Node], ident: tuple) -> list[tuple]:
    out = {(r.kind, r.id) for r in declared[ident].requires if (r.kind, r.id) in declared}
    return sorted(out, key=lambda k: _sort_key(*k))


def _topological(declared: Mapping[tuple, Node]) -> list[tuple]:
    """Dependencies first; among nodes that are ready at once, the smallest (kind rank, identifier) first."""
    indeg = {ident: len(_successors(declared, ident)) for ident in declared}
    dependents: dict[tuple, list[tuple]] = {ident: [] for ident in declared}
    for ident in declared:
        for dep in _successors(declared, ident):
            dependents[dep].append(ident)
    heap = [(_sort_key(*i), i) for i, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    order: list[tuple] = []
    while heap:
        _, ident = heapq.heappop(heap)
        order.append(ident)
        for nxt in dependents[ident]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(heap, (_sort_key(*nxt), nxt))
    if len(order) != len(declared):  # defence in depth: the cycle search above should have refused first
        left = sorted(node_key(*i) for i in declared if i not in set(order))
        raise ClosureError(Code.CYCLE, f"unresolved nodes remain: {left}",
                           (Problem(Code.CYCLE, left[0], "unresolved nodes remain"),), cycle=tuple(left))
    return order


def _derive(order: Sequence[tuple], declared: Mapping[tuple, Node]):
    """Direct predecessor tests and effective claims of every test, in one pass over the topological order."""
    frontier: dict[tuple, set[str]] = {}
    held: dict[tuple, set[tuple]] = {}   # non-test nodes reachable without passing through another test
    for ident in order:
        f: set[str] = set()
        h: set[tuple] = set()
        for dep in _successors(declared, ident):
            if dep[0] is NodeKind.TEST:
                f.add(dep[1])
            else:
                f |= frontier[dep]
                h.add(dep)
                h |= held[dep]
        frontier[ident] = f
        held[ident] = h
    preds: dict[str, tuple[str, ...]] = {}
    claims: dict[str, tuple[EffectiveClaim, ...]] = {}
    for ident in order:
        if ident[0] is not NodeKind.TEST:
            continue
        preds[ident[1]] = tuple(sorted(frontier[ident]))
        agg: dict[str, list] = {}
        for src in sorted([ident, *held[ident]], key=lambda k: _sort_key(*k)):
            for c in declared[src].own_claims():
                slot = agg.setdefault(c.resource, [0, False, []])
                slot[0] += c.amount
                slot[1] = slot[1] or c.exclusive
                slot[2].append(node_key(*src))
        claims[ident[1]] = tuple(EffectiveClaim(r, v[0], v[1], tuple(sorted(set(v[2]))))
                                 for r, v in sorted(agg.items()))
    return preds, claims
