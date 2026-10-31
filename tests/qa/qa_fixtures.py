"""Builders for the QA tests. Every digest is computed at run time; no literal digest or credential is stored."""
from __future__ import annotations

import hashlib
import random

from runtime_contracts.models.capability import SideEffect
from runtime_contracts.qa import (
    Action, ActionKind, AssertionSpec, AssertionSubject, Budget, Executor, FixtureRef, LocatorDescriptor, Predicate,
    PredicateOp, Step, StepTarget, SubjectKind, TestSpec, TestTarget,
)

from context_runtime.qa import Claim, Node, NodeKind, QAGraph, Ref


def digest(label: object, prefix: str = "rcv1") -> str:
    return f"{prefix}:{hashlib.sha256(str(label).encode()).hexdigest()}"


def make_spec(spec_id: str, *, depends_on=(), fixtures=(), env: str = "env-a", preconditions=(),
              expected: str = "Saved", side_effect=SideEffect.NONE, allowed=()) -> TestSpec:
    assertion = AssertionSpec(
        assertion_id="a-1", executor=Executor.PLAYWRIGHT,
        subject=AssertionSubject(kind=SubjectKind.UI_ELEMENT, locator=LocatorDescriptor(test_id="status")),
        predicate=Predicate(PredicateOp.EQUALS, expected), verifier_id="verifier-1",
        verifier_version_range=">=1.0,<2")
    step = Step(step_id="s-1", intent="press save", target=StepTarget.ui(LocatorDescriptor(role="button", name="Save")),
                action=Action(ActionKind.CLICK), expected_transition="status reads saved",
                observation_deadline_ms=1000, side_effect=side_effect)
    return TestSpec(
        id=spec_id, version=1, target=TestTarget(app_id="app-1", build_sha="abcdef0", environment_profile=env),
        steps=(step,), assertions=(assertion,), budget=Budget(deadline_ms=60000, max_recovery_attempts=2),
        egress_profile="egress-a", preconditions=tuple(preconditions),
        fixtures=tuple(FixtureRef(f, digest("fixture:" + f, "sha256")) for f in fixtures),
        depends_on=tuple(depends_on), allowed_side_effects=tuple(allowed))


def fixture_node(name: str, *, exclusive: bool = False, claims=()) -> Node:
    return Node(NodeKind.FIXTURE, name, digest("fixture:" + name, "sha256"), (), tuple(claims), exclusive)


def env_node(name: str = "env-a", claims=()) -> Node:
    return Node(NodeKind.ENVIRONMENT, name, None, (), tuple(claims))


def tnode(name: str, *, after=(), fixtures=(), env: str | None = None, claims=()) -> Node:
    reqs = [Ref(NodeKind.TEST, a, relation="depends_on") for a in after]
    reqs += [Ref(NodeKind.FIXTURE, f, digest("fixture:" + f, "sha256"), "fixture") for f in fixtures]
    if env:
        reqs.append(Ref(NodeKind.ENVIRONMENT, env, None, "environment"))
    return Node(NodeKind.TEST, name, digest("test:" + name), tuple(reqs), tuple(claims))


def random_dag(seed: int, n: int, *, p_edge: float = 0.25, n_fixtures: int = 3, p_fixture: float = 0.3,
               p_exclusive: float = 0.5) -> QAGraph:
    """A seeded random DAG of ``n`` tests (edges only from lower to higher index) over a few fixtures."""
    rng = random.Random(seed)
    nodes: list[Node] = []
    names = [f"t{i:03d}" for i in range(n)]
    fx = [f"fx{i}" for i in range(n_fixtures)]
    for f in fx:
        nodes.append(fixture_node(f, exclusive=rng.random() < p_exclusive))
    nodes.append(env_node("env-a"))
    for i, name in enumerate(names):
        after = [names[j] for j in range(i) if rng.random() < p_edge]
        fixtures = [f for f in fx if rng.random() < p_fixture]
        nodes.append(tnode(name, after=after, fixtures=fixtures, env="env-a"))
    rng.shuffle(nodes)
    return QAGraph(nodes)
