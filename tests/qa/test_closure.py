"""Closure over a declared graph: minimal, deterministic, explained, fail closed."""
from __future__ import annotations

import random

import pytest
from qa_fixtures import digest, env_node, fixture_node, make_spec, random_dag, tnode

from context_runtime.qa import (
    Claim, ClosureError, Code, Node, NodeKind, QAGraph, Ref, graph_from_specs, node_key, resolve,
)


def keys(closure):
    return [n.key for n in closure.nodes]


def test_chain_orders_dependencies_first():
    g = QAGraph([tnode("c", after=["b"]), tnode("b", after=["a"]), tnode("a")])
    c = resolve(g, ["c"])
    assert c.test_ids == ("a", "b", "c")
    assert c.predecessors("c") == ("b",) and c.predecessors("a") == ()


def test_diamond_is_ordered_and_has_direct_predecessors_only():
    g = QAGraph([tnode("top", after=["left", "right"]), tnode("left", after=["base"]),
                 tnode("right", after=["base"]), tnode("base")])
    c = resolve(g, ["top"])
    assert c.test_ids == ("base", "left", "right", "top")
    assert c.predecessors("top") == ("left", "right")
    assert c.predecessors("left") == ("base",)


def test_closure_is_minimal_for_the_selected_targets():
    g = QAGraph([tnode("a"), tnode("b", after=["a"]), tnode("unrelated"),
                 fixture_node("unused"), tnode("c", fixtures=["unused"])])
    c = resolve(g, ["b"])
    assert keys(c) == ["test:a", "test:b"]
    assert "test:unrelated" not in keys(resolve(g, ["b", "c"]))
    assert keys(resolve(g, ["b", "c"])) == ["fixture:unused", "test:a", "test:b", "test:c"]


def test_every_included_node_has_an_explicit_reason():
    g = QAGraph([tnode("b", after=["a"], fixtures=["f"], env="e"), tnode("a"), fixture_node("f"),
                 env_node("e")])
    c = resolve(g, ["b"])
    reasons = {n.key: [r.text() for r in n.reasons] for n in c.nodes}
    assert reasons["test:b"] == ["selected"]
    assert reasons["test:a"] == ["required by test:b (depends_on)"]
    assert reasons["fixture:f"] == ["required by test:b (fixture)"]
    assert reasons["environment:e"] == ["required by test:b (environment)"]
    assert all(n.reasons for n in c.nodes)


def test_a_node_required_twice_lists_both_reasons_in_a_stable_order():
    g = QAGraph([tnode("x", fixtures=["f"]), tnode("y", fixtures=["f"]), fixture_node("f")])
    c = resolve(g, ["y", "x"])
    assert [r.text() for r in c.node(NodeKind.FIXTURE, "f").reasons] == [
        "required by test:x (fixture)", "required by test:y (fixture)"]


def test_a_selected_test_that_is_also_required_keeps_both_reasons():
    g = QAGraph([tnode("a"), tnode("b", after=["a"])])
    c = resolve(g, ["a", "b"])
    assert [r.code for r in c.node(NodeKind.TEST, "a").reasons] == ["required_by", "selected"]


def test_cycle_is_refused_and_named():
    g = QAGraph([tnode("a", after=["c"]), tnode("b", after=["a"]), tnode("c", after=["b"]),
                 tnode("ok")])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["b"])
    assert e.value.code == Code.CYCLE
    assert e.value.cycle == ("test:a", "test:c", "test:b")      # rotated to start at the smallest key
    assert "test:a -> test:c -> test:b -> test:a" in e.value.problems[0].detail


def test_cycle_outside_the_selection_does_not_block_it():
    g = QAGraph([tnode("p", after=["q"]), tnode("q", after=["p"]), tnode("fine")])
    assert resolve(g, ["fine"]).test_ids == ("fine",)
    with pytest.raises(ClosureError):
        resolve(g, ["p"])


def test_self_dependency_is_a_one_node_cycle():
    g = QAGraph([tnode("a", after=["a"])])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["a"])
    assert e.value.code == Code.CYCLE and e.value.cycle == ("test:a",)


def test_cycle_through_a_fixture_is_found():
    g = QAGraph([Node(NodeKind.FIXTURE, "f", None, (Ref(NodeKind.TEST, "a"),)), tnode("a", fixtures=[])])
    g = QAGraph([Node(NodeKind.FIXTURE, "f", None, (Ref(NodeKind.TEST, "a"),)),
                 Node(NodeKind.TEST, "a", None, (Ref(NodeKind.FIXTURE, "f"),))])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["a"])
    assert e.value.code == Code.CYCLE and e.value.cycle == ("fixture:f", "test:a")


def test_a_very_long_chain_does_not_exhaust_the_stack():
    n = 3000
    nodes = [tnode(f"t{i:05d}", after=[f"t{i - 1:05d}"] if i else []) for i in range(n)]
    c = resolve(QAGraph(nodes), [f"t{n - 1:05d}"])
    assert len(c.tests) == n and c.test_ids[0] == "t00000"


def test_unknown_reference_is_refused_and_every_missing_reference_is_reported():
    g = QAGraph([tnode("a", after=["ghost"], fixtures=["nofx"]), tnode("b", after=["a", "other"])])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["b"])
    assert e.value.code == Code.UNKNOWN_REFERENCE
    found = {(p.subject, p.detail.split(" ")[1]) for p in e.value.problems if p.code == Code.UNKNOWN_REFERENCE}
    assert found == {("test:a", "test:ghost"), ("test:a", "fixture:nofx"), ("test:b", "test:other")}


def test_unknown_target_and_empty_selection_are_refused():
    g = QAGraph([tnode("a")])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["a", "nope"])
    assert e.value.code == Code.UNKNOWN_TARGET
    with pytest.raises(ClosureError) as e2:
        resolve(g, [])
    assert e2.value.code == Code.EMPTY_SELECTION


def test_a_fixture_is_not_a_valid_target():
    g = QAGraph([fixture_node("f"), tnode("a")])
    with pytest.raises(ClosureError) as e:
        resolve(g, [Ref(NodeKind.FIXTURE, "f")])
    assert e.value.code == Code.INVALID_GRAPH


def test_pin_mismatch_is_refused():
    a = tnode("a")
    wrong = Node(NodeKind.TEST, "b", digest("b"), (Ref(NodeKind.TEST, "a", digest("another version"), "depends_on"),))
    with pytest.raises(ClosureError) as e:
        resolve(QAGraph([a, wrong]), ["b"])
    assert e.value.code == Code.PIN_MISMATCH
    with pytest.raises(ClosureError) as e2:
        resolve(QAGraph([a]), [Ref(NodeKind.TEST, "a", digest("stale"))])
    assert e2.value.code == Code.PIN_MISMATCH


def test_a_pinned_reference_to_an_unpinned_node_is_a_mismatch():
    nodes = [Node(NodeKind.FIXTURE, "f", None), Node(NodeKind.TEST, "a", None, (Ref(NodeKind.FIXTURE, "f", "sha256:x"),))]
    with pytest.raises(ClosureError) as e:
        resolve(QAGraph(nodes), ["a"])
    assert e.value.code == Code.PIN_MISMATCH


def test_duplicate_declaration_is_refused():
    with pytest.raises(ClosureError) as e:
        QAGraph([tnode("a"), tnode("a")])
    assert e.value.code == Code.DUPLICATE_NODE


def test_graph_construction_refuses_malformed_nodes():
    with pytest.raises(ClosureError):
        Node(NodeKind.TEST, "")
    with pytest.raises(ClosureError):
        Node(NodeKind.TEST, "a", claims=(Claim("r"), Claim("r")))
    with pytest.raises(ClosureError):
        Claim("has space")
    with pytest.raises(ClosureError):
        Claim("r", 0)
    with pytest.raises(ClosureError):
        QAGraph(["not a node"])  # type: ignore[list-item]


def test_shared_exclusive_fixture_is_a_conflict_key_for_its_users():
    g = QAGraph([fixture_node("db", exclusive=True), fixture_node("cache"),
                 tnode("a", fixtures=["db"]), tnode("b", fixtures=["db", "cache"]),
                 tnode("c", fixtures=["cache"])])
    c = resolve(g, ["a", "b", "c"])
    assert c.exclusive_keys("a") == ("fixture:db",) and c.exclusive_keys("c") == ()
    assert c.conflicts("a", "b") and c.conflicts("b", "a")
    assert not c.conflicts("a", "c") and not c.conflicts("b", "c")
    assert not c.conflicts("a", "a")


def test_a_shared_fixture_without_exclusivity_does_not_conflict():
    g = QAGraph([fixture_node("cache"), tnode("a", fixtures=["cache"]), tnode("b", fixtures=["cache"])])
    c = resolve(g, ["a", "b"])
    assert not c.conflicts("a", "b")


def test_exclusive_and_shared_claims_on_one_resource_conflict_but_two_shared_do_not():
    g = QAGraph([
        tnode("x", claims=[Claim("gpu", 1, exclusive=True)]),
        tnode("y", claims=[Claim("gpu", 1)]),
        tnode("z", claims=[Claim("gpu", 1)]),
    ])
    c = resolve(g, ["x", "y", "z"])
    assert c.conflicts("x", "y") and not c.conflicts("y", "z")


def test_effective_claims_sum_own_and_inherited_claims_and_name_their_source():
    g = QAGraph([
        env_node("e", claims=[Claim("memory_mb", 512)]),
        fixture_node("f", claims=[Claim("memory_mb", 256), Claim("browser_slots", 1)]),
        tnode("t", fixtures=["f"], env="e", claims=[Claim("memory_mb", 128)]),
    ])
    c = resolve(g, ["t"])
    by = {x.resource: x for x in c.claims("t")}
    assert by["memory_mb"].amount == 512 + 256 + 128
    assert by["memory_mb"].via == ("environment:e", "fixture:f", "test:t")
    assert by["browser_slots"].via == ("fixture:f",)


def test_claims_do_not_flow_through_another_test():
    g = QAGraph([fixture_node("db", exclusive=True), tnode("a", fixtures=["db"]), tnode("b", after=["a"])])
    c = resolve(g, ["b"])
    assert c.claims("b") == () and c.predecessors("b") == ("a",)


def test_predecessors_pass_through_non_test_nodes():
    setup = Node(NodeKind.SETUP, "seed", None, (Ref(NodeKind.TEST, "loader", relation="requires"),))
    g = QAGraph([setup, tnode("loader"), Node(NodeKind.TEST, "t", digest("t"), (Ref(NodeKind.SETUP, "seed", None, "precondition"),))])
    c = resolve(g, ["t"])
    assert c.predecessors("t") == ("loader",)
    assert c.test_ids == ("loader", "t")


def test_order_does_not_depend_on_declaration_or_target_order():
    base = random_dag(7, 40)
    targets = ["t039", "t020", "t010"]
    ref = resolve(base, targets)
    rng = random.Random(1)
    for _ in range(10):
        nodes = list(base.nodes())
        rng.shuffle(nodes)
        shuffled = list(targets)
        rng.shuffle(shuffled)
        again = resolve(QAGraph(nodes), shuffled)
        assert again.digest() == ref.digest() and keys(again) == keys(ref)


def test_ties_break_by_kind_rank_then_identifier():
    g = QAGraph([tnode("b", fixtures=["z"]), tnode("a", fixtures=["y"]), fixture_node("z"), fixture_node("y"),
                 env_node("e")])
    c = resolve(g, ["a", "b"])
    assert keys(c) == ["fixture:y", "fixture:z", "test:a", "test:b"]


def test_identifiers_sort_by_code_point_not_locale_or_case():
    g = QAGraph([tnode(n) for n in ("a-10", "a-9", "B-1", "a-1")])
    assert resolve(g, ["a-10", "a-9", "B-1", "a-1"]).test_ids == ("B-1", "a-1", "a-10", "a-9")


@pytest.mark.parametrize("seed", range(12))
def test_property_every_closure_is_a_valid_topological_order_with_minimal_membership(seed):
    g = random_dag(seed, 30)
    rng = random.Random(seed)
    targets = rng.sample([f"t{i:03d}" for i in range(30)], rng.randint(1, 6))
    c = resolve(g, targets)
    pos = {n.key: i for i, n in enumerate(c.nodes)}
    # dependencies come first
    for n in c.nodes:
        for dep in n.requires:
            assert pos[dep] < pos[n.key]
    # every node is a target or required by a member of the closure (minimal)
    members = set(pos)
    for n in c.nodes:
        assert n.reasons
        for r in n.reasons:
            assert r.code == "selected" or r.by in members
    assert set(c.targets) == {t for t in targets}
    # nothing outside the transitive requirement set is included
    needed: set[str] = set()
    stack = [node_key(NodeKind.TEST, t) for t in targets]
    byk = {x.key: x for x in g.nodes()}
    while stack:
        k = stack.pop()
        if k in needed:
            continue
        needed.add(k)
        stack.extend(node_key(r.kind, r.id) for r in byk[k].requires)
    assert members == needed


def test_graph_from_specs_pins_dependencies_to_spec_hashes():
    a = make_spec("spec-a")
    b = make_spec("spec-b", depends_on=[a.spec_ref], fixtures=["db"])
    g = graph_from_specs([b, a], fixtures=[fixture_node("db", exclusive=True)], environments=[env_node("env-a")])
    c = resolve(g, ["spec-b"])
    assert c.test_ids == ("spec-a", "spec-b")
    assert c.node(NodeKind.TEST, "spec-b").pin == b.spec_hash
    assert c.exclusive_keys("spec-b") == ("fixture:db",)


def test_graph_from_specs_spec_hash_mismatch_in_depends_on_is_refused():
    a = make_spec("spec-a")
    stale_ref = f"spec-a@{digest('an older version of spec-a')}"
    b = make_spec("spec-b", depends_on=[stale_ref])
    g = graph_from_specs([a, b], environments=[env_node("env-a")])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["spec-b"])
    assert e.value.code == Code.PIN_MISMATCH


def test_graph_from_specs_undeclared_fixture_environment_or_precondition_is_refused():
    a = make_spec("spec-a", fixtures=["db"], preconditions=["user is signed out"])
    g = graph_from_specs([a])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["spec-a"])
    assert e.value.code == Code.UNKNOWN_REFERENCE
    details = " | ".join(p.detail for p in e.value.problems)
    for want in ("fixture:db", "environment:env-a", "setup:user is signed out"):
        assert f"requires {want} (" in details, details


def test_graph_from_specs_fixture_content_hash_must_match_the_declared_fixture():
    a = make_spec("spec-a", fixtures=["db"])
    other = Node(NodeKind.FIXTURE, "db", digest("a different content", "sha256"))
    g = graph_from_specs([a], fixtures=[other], environments=[env_node("env-a")])
    with pytest.raises(ClosureError) as e:
        resolve(g, ["spec-a"])
    assert e.value.code == Code.PIN_MISMATCH


def test_graph_from_specs_preconditions_resolve_when_declared():
    a = make_spec("spec-a", preconditions=["user is signed out"])
    g = graph_from_specs([a], environments=[env_node("env-a")],
                         setups=[Node(NodeKind.SETUP, "user is signed out", None, (), (Claim("session", 1, True),))])
    c = resolve(g, ["spec-a"])
    assert c.exclusive_keys("spec-a") == ("session",)
