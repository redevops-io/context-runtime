"""KafkaEventBus satisfies the EventBus seam — tested against an injected fake client (no broker),
so the aggregator/loop code is identical whether the bus is in-process or Kafka."""
from __future__ import annotations

from context_runtime.integrations.bandit import EpsilonGreedyBandit
from context_runtime.learning import LearningAggregator, OutcomeEvent
from context_runtime.learning.kafka_bus import KafkaEventBus


class _Rec:
    def __init__(self, value):
        self.value = value


class _FakeKafka:
    """Stand-in for a broker: send() appends; consumer(topic) drains what's queued."""

    def __init__(self):
        self.topics: dict[str, list] = {}

    def send(self, topic, value):
        self.topics.setdefault(topic, []).append(value)

    def flush(self):
        pass

    def consumer(self, topic):
        recs = [_Rec(v) for v in self.topics.get(topic, [])]
        self.topics[topic] = []
        return recs


def test_event_roundtrips_through_the_kafka_seam():
    bus = KafkaEventBus("x:9092", client=_FakeKafka(),
                        codec_out=OutcomeEvent.to_dict, codec_in=OutcomeEvent.from_dict)
    ev = OutcomeEvent(context="mh", arm="graph:cheap", reward=0.9, seq=1, accepted=True)
    bus.publish("outcomes", ev)
    got = bus.poll("outcomes")
    assert got == [ev]                    # serialized out, reconstructed in
    assert bus.poll("outcomes") == []     # drained


def test_plain_dicts_roundtrip_for_the_cdc_stream():
    """No codec → payloads pass through as-is (the vuln / context-freshness stream ships dict rows)."""
    bus = KafkaEventBus("x:9092", client=_FakeKafka())
    row = {"cve_id": "CVE-2026-1", "package": "openssl", "cvss": 9.1}
    bus.publish("vuln-freshness", row)
    assert bus.poll("vuln-freshness") == [row]


def test_aggregator_drains_from_the_kafka_bus():
    bus = KafkaEventBus("x:9092", client=_FakeKafka(),
                        codec_out=OutcomeEvent.to_dict, codec_in=OutcomeEvent.from_dict)
    for i, (arm, r) in enumerate([("graph:cheap", 0.9), ("graph:cheap", 0.9), ("hybrid:cheap", 0.2)]):
        bus.publish("outcomes", OutcomeEvent(context="mh", arm=arm, reward=r, seq=i + 1))
    agg = LearningAggregator(EpsilonGreedyBandit(arms=()))
    assert agg.drain(bus) == 3
    assert agg.bandit.value("mh", "graph:cheap")[1] > agg.bandit.value("mh", "hybrid:cheap")[1]
