"""Nothing learns without being listed (context_runtime/learning/registry.py), and what a learner may change is closed."""
import ast
from pathlib import Path

from context_runtime.learning.registry import LEARNERS, LEARNING_METHODS, MAY_CHANGE, NOT_LEARNERS

PKG = Path(__file__).resolve().parents[1] / "context_runtime"


def _classes_with_learning_methods():
    found = {}
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = sorted(f.name for f in node.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name in LEARNING_METHODS)
                if methods:
                    found[f"{path.relative_to(PKG).as_posix()}:{node.name}"] = (methods, node)
    return found


def test_every_class_with_a_learning_method_is_listed_once_and_nothing_listed_is_gone():
    found = _classes_with_learning_methods()
    listed = [x.key for x in LEARNERS]
    assert len(listed) == len(set(listed)) and not set(listed) & set(NOT_LEARNERS)
    unlisted = sorted(set(found) - set(listed) - set(NOT_LEARNERS))
    assert not unlisted, f"a class learns (or looks as if it does) and is not in learning/registry.py: {unlisted}"
    gone = sorted((set(listed) | set(NOT_LEARNERS)) - set(found) - set(_bandit_drivers()))
    assert not gone, f"listed in learning/registry.py but not in the package any more: {gone}"


def _bandit_drivers():
    out = []
    for path in sorted(PKG.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ClassDef):
                body = ast.get_source_segment(source, node) or ""
                if any(m in body for m in ("bandit.update(", "EpsilonGreedyBandit(", "bandit.learn(", "optimizer.learn(")):
                    out.append(f"{path.relative_to(PKG).as_posix()}:{node.name}")
    return out


def test_a_class_that_drives_a_bandit_is_listed_whatever_its_methods_are_called():
    unlisted = sorted(set(_bandit_drivers()) - {x.key for x in LEARNERS} - set(NOT_LEARNERS))
    assert not unlisted, f"these classes drive a bandit and are not in learning/registry.py: {unlisted}"


def test_only_two_places_register_arms_at_run_time():
    """Both register PLAN SHAPES. The optimizer then chooses only among the candidates it is given; the aggregator's arm
    comes from an outcome event, which is why its entry says so."""
    grows = [x.key for x in LEARNERS if x.changes == "choice among plan shapes named by outcome events"]
    assert grows == ["learning/aggregator.py:LearningAggregator"]
    adds = sorted(p.relative_to(PKG).as_posix() for p in PKG.rglob("*.py") if "bandit.arms =" in p.read_text(encoding="utf-8"))
    assert adds == ["learning/aggregator.py", "optimizer/online.py"], f"something else adds arms to a bandit at run time: {adds}"


def test_what_a_learner_may_change_is_a_closed_list_and_every_entry_is_complete():
    for x in LEARNERS:
        assert x.changes in MAY_CHANGE, (x.key, x.changes)
        assert all(str(v).strip() for v in (x.learns, x.persisted, x.kill_switch, x.reset, x.provenance)), x.key
    assert all(reason.strip() for reason in NOT_LEARNERS.values())
    for word in ("policy", "approval", "authorization", "egress", "verification"):
        assert not any(word in c for c in MAY_CHANGE), "no learner may change policy, approval, authorization, egress or verification"


def test_an_entry_that_claims_the_bandits_switch_really_learns_only_through_a_bandit():
    """A tenant listed as inheriting the bandit's kill switch must not keep a learned value of its own that the switch
    would not stop: its class has to use a bandit, or the optimizer that wraps one."""
    found = _classes_with_learning_methods()
    for x in LEARNERS:
        if "CR_LEARNING_FROZEN" in x.kill_switch and not x.key.endswith(":EpsilonGreedyBandit"):
            module = x.key.split(":")[0]
            source = (PKG / module).read_text(encoding="utf-8")
            assert "bandit" in source.lower(), f"{x.key} claims the bandit's kill switch but its module uses no bandit"
            assert x.key in found or x.key in _bandit_drivers()
