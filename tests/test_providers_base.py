"""The cloud-provider seam works with no cloud SDK installed.

Proves the architecture promise: providers plug in behind neutral Protocols, the kernel imports no
SDK, and an unconfigured capability cleanly returns None (caller falls back to the in-tree default).
"""
from context_runtime.providers import base
from context_runtime.providers.base import (
    CloudProvider,
    GuardrailVerdict,
    available_providers,
    get_provider,
    register_provider,
)


def test_base_provider_offers_nothing_by_default():
    p = CloudProvider()
    for cap in ("model", "document_retriever", "managed_kb_retriever", "analytical_backend",
                "guardrail", "identity_broker", "telemetry_reader"):
        assert getattr(p, cap)() is None
    assert p.info()["capabilities"] == {c: False for c in p.info()["capabilities"]}


def test_registry_roundtrip():
    class DummyGuard:
        def check_input(self, text):
            return GuardrailVerdict(allowed=True)
        def check_output(self, text):
            return GuardrailVerdict(allowed=True)

    class DummyProvider(CloudProvider):
        name = "dummy"
        def guardrail(self):
            return DummyGuard()

    register_provider("dummy", DummyProvider)
    assert "dummy" in available_providers()
    p = get_provider("dummy")
    assert p.name == "dummy"
    assert p.guardrail().check_input("hi").allowed is True
    # info() reports exactly the offered capability
    caps = p.info()["capabilities"]
    assert caps["guardrail"] is True and caps["model"] is False


def test_unknown_provider_raises():
    try:
        get_provider("nope")
        assert False, "expected KeyError"
    except KeyError as e:
        assert "nope" in str(e)


def test_aws_provider_constructs_without_boto3_and_unconfigured_caps_are_none():
    # No endpoint/kb/athena/guardrail configured, and no boto3 installed → these must be None, not raise.
    p = get_provider("aws")
    assert p.name == "aws"
    assert p.managed_kb_retriever() is None
    assert p.analytical_backend() is None
    assert p.guardrail() is None
    assert p.identity_broker() is None
    assert "aws" in available_providers()
