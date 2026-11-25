"""A reward that is not a finite number must not be learned, persisted, or survive a restart."""
import json
import math

import pytest

from context_runtime.integrations.bandit import EpsilonGreedyBandit


class Arm:
    def __init__(self, key):
        self.key = key


A, B = Arm("a"), Arm("b")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "0.9", True])
def test_a_non_finite_reward_is_refused_and_changes_nothing(tmp_path, bad):
    path = tmp_path / "policy.json"
    b = EpsilonGreedyBandit((A, B), epsilon=0.0, persist_path=str(path))
    b.update("ctx", A, 0.9)
    before = b.value("ctx", "a")
    with pytest.raises(ValueError, match="finite number"):
        b.update("ctx", A, bad)
    assert b.value("ctx", "a") == before
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_before_the_fix_one_nan_made_the_arm_nan_forever_and_a_poisoned_file_is_healed_on_load(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text('{"ctx": {"a": [3, NaN], "b": [2, 0.4]}}')            # what a NaN reward left on disk
    b = EpsilonGreedyBandit((A, B), epsilon=0.0, persist_path=str(path))
    n, mean = b.value("ctx", "a")
    assert n == 0 and mean == b.optimistic and math.isfinite(mean)       # forgotten, tried again as unseen
    assert b.value("ctx", "b") == (2, 0.4)
    b.update("ctx", A, 0.2)
    n, mean = json.loads(path.read_text())["ctx"]["a"]
    assert n == 1 and mean == pytest.approx(0.2)
