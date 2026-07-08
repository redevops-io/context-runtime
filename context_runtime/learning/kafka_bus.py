"""Kafka binding for the learning loop's EventBus (Whitepaper v3, Phase 4 — the "nervous system").

``KafkaEventBus`` satisfies the same ``publish`` / ``poll`` seam as ``InMemoryBus``, so the async policy
loop (and the context-freshness stream) run unchanged across stateless replicas over a real broker. The
engine stays dependency-free: ``kafka-python`` is imported lazily, only when this binding is constructed.

    bus = KafkaEventBus("broker:9092", codec_out=OutcomeEvent.to_dict, codec_in=OutcomeEvent.from_dict)
    bus.publish("outcomes", event)          # producer → topic
    for ev in bus.poll("outcomes"): ...     # consumer drains what's available

Serialization is JSON by default; pass ``codec_out``/``codec_in`` to (de)serialize a specific event type
(e.g. ``OutcomeEvent``). Payloads that are already dicts round-trip as-is (handy for the CDC/vuln stream).
"""
from __future__ import annotations

import json
from typing import Callable


class KafkaEventBus:
    def __init__(
        self,
        bootstrap_servers: str,
        *,
        group_id: str = "context-runtime",
        codec_out: Callable[[object], dict] | None = None,
        codec_in: Callable[[dict], object] | None = None,
        poll_timeout_ms: int = 800,
        auto_offset_reset: str = "earliest",
        client=None,   # inject a fake (must expose .send/.flush and a consumer factory) for tests
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.codec_out = codec_out or (lambda e: e.to_dict() if hasattr(e, "to_dict") else e)
        self.codec_in = codec_in or (lambda d: d)
        self.poll_timeout_ms = poll_timeout_ms
        self.auto_offset_reset = auto_offset_reset
        self._injected = client
        self._producer = None
        self._consumers: dict[str, object] = {}   # one consumer per polled topic
        if client is None:
            try:
                import kafka  # noqa: F401
            except ImportError as e:   # pragma: no cover - environment dependent
                raise ImportError(
                    "KafkaEventBus needs the 'kafka-python' package (pip install kafka-python), "
                    "or inject a fake client for tests."
                ) from e

    # ── producer ──
    def _get_producer(self):
        if self._injected is not None:
            return self._injected
        if self._producer is None:
            from kafka import KafkaProducer
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        return self._producer

    def publish(self, topic: str, event) -> None:
        self._get_producer().send(topic, self.codec_out(event))

    def flush(self) -> None:
        p = self._get_producer()
        if hasattr(p, "flush"):
            p.flush()

    # ── consumer ──
    def _get_consumer(self, topic: str):
        if self._injected is not None:
            return self._injected.consumer(topic)   # fake factory in tests
        if topic not in self._consumers:
            from kafka import KafkaConsumer
            self._consumers[topic] = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=True,
                consumer_timeout_ms=self.poll_timeout_ms,
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
            )
        return self._consumers[topic]

    def poll(self, topic: str) -> list:
        """Drain the records available within one poll window and decode them (at-least-once)."""
        consumer = self._get_consumer(topic)
        out = []
        for record in consumer:                      # bounded by consumer_timeout_ms
            out.append(self.codec_in(record.value))
        return out

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close()
        for c in self._consumers.values():
            c.close()
