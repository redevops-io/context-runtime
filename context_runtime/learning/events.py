"""Outcome events — what execution emits for the async learning loop (Whitepaper v3, Phase 4).

Every served (or abstained) plan produces one ``OutcomeEvent``: the selection context + arm, the
measured reward, the selection propensity (for off-policy), and any operator/verification signals. The
event is published to a bus and folded into learned state OFF the serving path — it never touches the
model's context window. This is the "policy loop" half of the paper's Kafka nervous system.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OutcomeEvent:
    context: str                       # the selection context (intent bucket)
    arm: str                           # the chosen plan shape (method:tier)
    reward: float                      # measured reward for this execution
    representation: str = ""           # v4: knowledge representation the plan routed to (attribution)
    seq: int = 0                       # monotonic sequence number (ordering / idempotency)
    mode: str = ""                     # exploit | explore | shadow (from the bandit metadata)
    p: float = 1.0                     # selection propensity, for off-policy evaluation
    abstained: bool = False            # the planner declined to serve
    confidence: float | None = None    # calibrated confidence of the served plan
    # optional operator/verification signals, for a trust sink
    accepted: bool | None = None
    overridden: bool = False
    regenerated: bool = False
    verified: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutcomeEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_plan(cls, plan, reward: float, *, seq: int = 0, **signals) -> "OutcomeEvent":
        """Build an event from a served Plan (reading the optimizer's ``extra['bandit']`` / ``extra
        ['abstention']``) plus the measured reward and any operator signals."""
        extra = getattr(plan, "extra", None) or {}
        b = extra.get("bandit") or {}
        ab = extra.get("abstention") or {}
        return cls(
            context=b.get("context", ""),
            arm=b.get("arm", ""),
            reward=reward,
            representation=getattr(getattr(plan, "intent", None), "representation", "") or "",
            seq=seq,
            mode=b.get("mode", ""),
            p=float(b.get("p", 1.0)),
            abstained=ab.get("action") == "abstain",
            confidence=ab.get("confidence"),
            **signals,
        )
