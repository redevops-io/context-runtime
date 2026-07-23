"""BedrockModel is a ModelPlugin over `converse`, exercised with a fake client (no boto3)."""
from context_runtime.adapters.model_litellm import Tier
from context_runtime.plugins import base
from context_runtime.providers.aws.bedrock_model import BedrockModel
from context_runtime.types import ModelRequest


class FakeBedrock:
    """Records converse() calls and returns a canned Bedrock response shape."""
    def __init__(self):
        self.calls = []

    def converse(self, **kw):
        self.calls.append(kw)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello from bedrock"}]}},
            "usage": {"inputTokens": 12, "outputTokens": 3, "totalTokens": 15},
            "stopReason": "end_turn",
        }


def _model():
    tiers = [Tier(name="cheap", model="amazon.nova-lite-v1:0", cost_per_1k=0.001)]
    return BedrockModel(tiers, client=FakeBedrock(), default_tier="cheap")


def test_satisfies_model_plugin_protocol():
    assert isinstance(_model(), base.ModelPlugin)


def test_complete_maps_request_and_response():
    m = _model()
    res = m.complete(ModelRequest(messages=({"role": "user", "content": "hi"},),
                                  system="be brief", max_tokens=64, capability="synthesis"))
    assert res.text == "hello from bedrock"
    assert res.model == "amazon.nova-lite-v1:0" and res.tier == "cheap"
    assert res.prompt_tokens == 12 and res.completion_tokens == 3
    assert res.est_cost_usd == round(15 / 1000 * 0.001, 6)
    assert res.models_used == ("amazon.nova-lite-v1:0",)


def test_converse_payload_shape():
    fake = FakeBedrock()
    m = BedrockModel([Tier(name="cheap", model="amazon.nova-lite-v1:0")], client=fake)
    m.complete(ModelRequest(messages=({"role": "user", "content": "count to 3"},),
                            system="sys", max_tokens=32))
    call = fake.calls[0]
    assert call["modelId"] == "amazon.nova-lite-v1:0"
    assert call["messages"] == [{"role": "user", "content": [{"text": "count to 3"}]}]
    assert call["system"] == [{"text": "sys"}]
    assert call["inferenceConfig"] == {"maxTokens": 32}


def test_from_config_defaults_to_nova_claude_map():
    m = BedrockModel.from_config(session=None)
    assert set(m.tiers) == {"local", "cheap", "premium"}
    assert "nova-micro" in m.tiers["local"].model


def test_works_as_the_runtime_model_plane():
    from context_runtime import ContextRuntime
    m = _model()
    rt = ContextRuntime(models={t: m for t in ("local", "cheap", "premium")},
                        retriever=__import__("context_runtime.adapters.store_inmemory",
                                             fromlist=["InMemoryStore"]).InMemoryStore(
                            [{"chunk_id": "d1", "filename": "d1", "text": "bedrock rocks", "created_at": None}]))
    res = rt.run("what rocks?")
    assert "bedrock" in res.answer.lower()
