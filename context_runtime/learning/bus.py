"""Event bus — the transport for the async learning loop (Whitepaper v3, Phase 4).

The planner never learns on the hot path. Instead it publishes ``OutcomeEvent``s to a bus; an
aggregator consumes them off the serving path and republishes learned-state snapshots. The engine
depends only on this small ``EventBus`` Protocol, so the default in-process ``InMemoryBus`` (for a
single node / tests) and a Kafka binding (the paper's "nervous system", for a fleet) are
interchangeable — the two streams never touch the model's context window.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Protocol, runtime_checkable


@runtime_checkable
class EventBus(Protocol):
    def publish(self, topic: str, event) -> None: ...
    def poll(self, topic: str) -> list: ...


class InMemoryBus:
    """A thread-safe, in-process bus. ``poll`` drains all pending events for a topic (at-most-once,
    FIFO). Sufficient for a single node and for deterministic tests; swap in a KafkaBus (same
    publish/poll seam) to distribute the loop across stateless replicas."""

    def __init__(self):
        self._topics: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def publish(self, topic: str, event) -> None:
        with self._lock:
            self._topics[topic].append(event)

    def poll(self, topic: str) -> list:
        with self._lock:
            q = self._topics.get(topic)
            if not q:
                return []
            items = list(q)
            q.clear()
            return items

    def pending(self, topic: str) -> int:
        with self._lock:
            return len(self._topics.get(topic, ()))
