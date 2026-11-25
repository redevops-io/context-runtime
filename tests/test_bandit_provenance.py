"""The shared learning core: where a learned value came from, how to take a bad observation back, how to stop it, and what
happens to a policy file that cannot be read."""
import json
import logging
import os
import threading
from dataclasses import dataclass

import pytest

from context_runtime.integrations.bandit import EpsilonGreedyBandit
from context_runtime.learning import ledger
from context_runtime.learning.ledger import LedgerIncomplete, provenance


@dataclass(frozen=True)
class A:
    key: str


ARMS = (A("x"), A("y"))


def _bandit(tmp_path, **kw):
    return EpsilonGreedyBandit(ARMS, epsilon=0.0, persist_path=str(tmp_path / "policy.json"), **kw)


def _entries(b):
    return ledger.read(b.ledger_path)


def test_every_observation_says_who_produced_it(tmp_path):
    b = _bandit(tmp_path)
    b.update("c", ARMS[0], 1.0, provenance={"source": "verifier", "run_id": "r1"})
    with provenance(source="user-feedback", principal="alice"):
        b.update("c", ARMS[1], 0.0)
    b.update("c", ARMS[1], 0.5)
    who = [e["provenance"] for e in _entries(b)]
    assert who == [{"source": "verifier", "run_id": "r1"}, {"source": "user-feedback", "principal": "alice"}, {"source": "unattributed"}]
    assert ledger.same_learning(ledger.rebuild(_entries(b)), b.stats)
    assert oct(os.stat(b.ledger_path).st_mode & 0o777) == "0o600"


def test_the_ledger_takes_no_request_data(tmp_path):
    b = _bandit(tmp_path)
    with pytest.raises(ValueError, match="provenance takes only"):
        b.update("c", ARMS[0], 1.0, provenance={"source": "x", "question": "what is the customer's address"})
    assert b.value("c", "x") == (0, 1.0)                                    # refused before anything was learned


def test_a_poisoned_signal_is_taken_back_and_the_rest_is_kept(tmp_path):
    b = _bandit(tmp_path)
    for r in (0.7, 0.6):
        b.update("c", ARMS[0], r, provenance={"source": "verifier"})
    for _ in range(5):
        b.update("c", ARMS[1], 1.0, provenance={"source": "feedback", "principal": "mallory"})
    b.update("c", ARMS[1], 0.1, provenance={"source": "verifier"})
    assert b.policy()["c"] == "y"
    n = b.rollback(lambda e: e["provenance"].get("principal") == "mallory", by="operator-1")
    assert n == 5 and b.value("c", "y") == (1, pytest.approx(0.1)) and b.value("c", "x") == (2, pytest.approx(0.65)) and b.policy()["c"] == "x"
    assert b.rollback(lambda e: e["provenance"].get("principal") == "mallory", by="operator-1") == 0      # not twice
    assert b.rollback(lambda e: e["reward"] == 0.7, by="operator-1") == 1 and b.value("c", "x") == (1, pytest.approx(0.6))   # rollbacks compose
    again = _bandit(tmp_path)                                               # and survive a restart
    assert again.value("c", "x") == (1, pytest.approx(0.6)) and ledger.same_learning(ledger.rebuild(_entries(again)), again.stats)


def test_rollback_with_a_discounted_learner(tmp_path):
    b = _bandit(tmp_path, discount=0.5)
    for r in (1.0, 0.0, 1.0):
        b.update("c", ARMS[0], r, provenance={"source": "s", "run_id": str(r)})
    b.rollback(lambda e: e["reward"] == 0.0, by="op")
    assert b.value("c", "x") == (2, pytest.approx(1.0))


def test_rollback_refuses_when_the_history_does_not_reproduce_the_policy(tmp_path):
    b = _bandit(tmp_path)
    b.update("c", ARMS[0], 1.0)
    b.update("c", ARMS[0], 0.0)
    lines = open(b.ledger_path, encoding="utf-8").read().splitlines()
    open(b.ledger_path, "w", encoding="utf-8").write(lines[0] + "\n")                          # an entry was lost
    with pytest.raises(LedgerIncomplete):
        b.rollback(lambda e: True, by="op")
    assert b.value("c", "x") == (2, pytest.approx(0.5))                      # nothing changed
    open(b.ledger_path, "w", encoding="utf-8").write(lines[0] + "\nnot json\n")
    with pytest.raises(LedgerIncomplete):
        b.rollback(lambda e: True, by="op")


def test_a_policy_learned_before_the_ledger_is_its_starting_point(tmp_path):
    (tmp_path / "policy.json").write_text(json.dumps({"c": {"x": [4, 0.25], "y": [0, 1.0]}}), encoding="utf-8")
    b = _bandit(tmp_path)
    b.update("c", ARMS[0], 1.0)
    assert [e["op"] for e in _entries(b)] == ["baseline", "update"]
    assert b.rollback(lambda e: True, by="op") == 1 and b.value("c", "x") == (4, pytest.approx(0.25))


def test_a_ledger_that_cannot_be_written_never_stops_learning_but_stops_rollback(tmp_path, caplog):
    b = _bandit(tmp_path)
    os.mkdir(b.ledger_path)                                                 # nothing can be appended to a directory
    with caplog.at_level(logging.WARNING, logger="context_runtime.learning"):
        b.update("c", ARMS[0], 1.0)
        b.update("c", ARMS[0], 1.0)
    assert b.value("c", "x")[0] == 2 and b.status()["ledger_errors"] == 2
    assert sum("cannot be written" in r.message for r in caplog.records) == 1


def test_a_rotated_ledger_still_reproduces_the_policy_under_concurrency(tmp_path):
    b = _bandit(tmp_path, ledger_max_bytes=2000)

    def work(i):
        for n in range(60):
            b.update("c", ARMS[(i + n) % 2], (n % 3) / 2, provenance={"source": "w", "run_id": f"{i}-{n}"})
    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert os.path.exists(b.ledger_path + ".1")
    entries = _entries(b)
    assert any(e["op"] == "baseline" for e in entries)
    assert ledger.same_learning(ledger.rebuild(entries), b.stats), "a baseline must never let an observation be counted twice"


def test_the_kill_switch_stops_learning_and_exploration(tmp_path, monkeypatch):
    b = EpsilonGreedyBandit(ARMS, epsilon=1.0, persist_path=str(tmp_path / "p.json"))
    b.update("c", ARMS[0], 0.1)
    b.update("c", ARMS[1], 0.9)
    before = json.dumps(b.stats)
    monkeypatch.setenv("CR_LEARNING_FROZEN", "1")
    b.update("c", ARMS[0], 1.0)
    assert json.dumps(b.stats) == before and b.status() == {**b.status(), "frozen": True, "refused_while_frozen": 1}
    assert {b.select("c").key for _ in range(40)} == {"y"}                   # with epsilon 1.0: no exploration while frozen
    monkeypatch.delenv("CR_LEARNING_FROZEN")
    b.frozen = True
    b.update("c", ARMS[0], 1.0)
    assert json.dumps(b.stats) == before
    b.frozen = False
    assert {b.select("c").key for _ in range(60)} == {"x", "y"}


def test_reset_forgets_and_names_who_asked(tmp_path):
    b = _bandit(tmp_path)
    b.update("c", ARMS[0], 0.0)
    b.update("d", ARMS[0], 0.0)
    with pytest.raises(ValueError):
        b.reset("c", by="")
    b.reset("c", by="op")
    assert b.value("c", "x") == (0, 1.0) and b.value("d", "x") == (1, 0.0)
    b.reset(by="op")
    assert b.status()["observations"] == 0 and [e["op"] for e in _entries(b)][-2:] == ["reset", "reset"]
    assert ledger.same_learning(ledger.rebuild(_entries(b)), b.stats)


@pytest.mark.parametrize("content", ['{"c": ', '[1, 2]', '{"c": {"x": "nonsense"}}', ''])
def test_an_unreadable_policy_file_is_kept_and_said_so(tmp_path, caplog, content):
    (tmp_path / "policy.json").write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="context_runtime.learning"):
        b = _bandit(tmp_path)
    kept = [f for f in os.listdir(tmp_path) if ".corrupt-" in f]
    assert len(kept) == 1 and (tmp_path / kept[0]).read_text(encoding="utf-8") == content and any("could not be read" in r.message for r in caplog.records)
    b.update("c", ARMS[0], 1.0)
    assert b.value("c", "x") == (1, 1.0)


def test_restarting_blank_after_a_corrupt_file_keeps_the_history_true(tmp_path):
    b = _bandit(tmp_path)
    b.update("c", ARMS[0], 0.2)
    (tmp_path / "policy.json").write_text("{broken", encoding="utf-8")
    again = _bandit(tmp_path)
    assert again.status()["observations"] == 0 and ledger.same_learning(ledger.rebuild(_entries(again)), again.stats)


def test_an_unreadable_file_that_cannot_be_moved_is_never_overwritten(tmp_path, monkeypatch):
    (tmp_path / "policy.json").write_text("{broken", encoding="utf-8")
    real = os.replace

    def refuse(src, dst, *a, **k):
        if ".corrupt-" in str(dst):
            raise PermissionError("no")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(os, "replace", refuse)
    b = _bandit(tmp_path)
    b.update("c", ARMS[0], 1.0)
    assert (tmp_path / "policy.json").read_text(encoding="utf-8") == "{broken" and b.status()["persist_blocked"]


def test_a_persisted_arm_that_is_not_declared_can_never_be_selected(tmp_path):
    """A removed (or invented) arm with the best score made select() raise StopIteration."""
    (tmp_path / "policy.json").write_text(json.dumps({"c": {"removed": [9, 99.0], "x": [1, 0.1], "y": [1, 0.2]}}), encoding="utf-8")
    b = _bandit(tmp_path)
    assert b.select("c").key == "y"


def test_the_operator_command_previews_before_it_changes_anything(tmp_path, capsys):
    from context_runtime.learning import admin
    b = _bandit(tmp_path)
    b.update("c", ARMS[0], 0.9, provenance={"source": "verifier"})
    b.update("c", ARMS[1], 1.0, provenance={"source": "feedback", "principal": "mallory"})
    path = str(tmp_path / "policy.json")
    before = (tmp_path / "policy.json").read_text(encoding="utf-8")
    assert admin.main(["status", path]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["reproduced_by_ledger"] and out["updates_in_ledger_by_source"] == {"verifier": 1, "feedback": 1}
    assert admin.main(["rollback", path, "--by", "op", "--principal", "mallory"]) == 0 and "1 observation(s) match" in capsys.readouterr().out
    assert (tmp_path / "policy.json").read_text(encoding="utf-8") == before                       # a preview changes nothing
    assert admin.main(["rollback", path, "--principal", "mallory", "--apply"]) == 2               # who is required
    assert admin.main(["rollback", path, "--by", "op", "--apply"]) == 2                           # and which observations
    assert admin.main(["rollback", path, "--by", "op", "--principal", "mallory", "--apply"]) == 0
    assert _bandit(tmp_path).value("c", "y")[0] == 0
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    assert admin.main(["status", str(tmp_path / "bad.json")]) == 1 and (tmp_path / "bad.json").read_text(encoding="utf-8") == "{broken"
