"""DigitalOcean provider: Gradient inference (OpenAI-compatible) + KB retrieve, via injected transport."""
from context_runtime.plugins import base
from context_runtime.providers.base import get_provider
from context_runtime.providers.digitalocean.session import DoSession
from context_runtime.types import ModelRequest


def _session(response, capture=None):
    def transport(url, body, headers, timeout):
        if capture is not None:
            capture.append({"url": url, "body": body, "headers": headers})
        return response(url, body) if callable(response) else response
    return DoSession(api_token="tok", inference_key="ikey", transport=transport)


# ── Gradient model (OpenAI-compatible) ───────────────────────────────────────────────────────────
def test_gradient_model_completes_over_openai_shape():
    from context_runtime.providers.digitalocean.gradient_model import GradientModel
    from context_runtime.adapters.model_litellm import Tier
    cap = []
    resp = {"model": "llama3.3-70b-instruct", "choices": [{"message": {"content": "do says hi"}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3}}
    m = GradientModel([Tier(name="cheap", model="llama3.3-70b-instruct", cost_per_1k=0.001)],
                      session=_session(resp, cap))
    assert isinstance(m, base.ModelPlugin)
    res = m.complete(ModelRequest(messages=({"role": "user", "content": "hi"},), system="sys", max_tokens=50))
    assert res.text == "do says hi" and res.prompt_tokens == 8 and res.completion_tokens == 3
    call = cap[0]
    assert call["url"] == "https://inference.do-ai.run/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer ikey"
    assert call["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert call["body"]["max_tokens"] == 50


# ── Gradient knowledge base retrieve ─────────────────────────────────────────────────────────────
def test_gradient_kb_retriever_maps_chunks():
    from context_runtime.providers.digitalocean.gradient_kb_retriever import GradientKBRetriever
    cap = []
    resp = {"results": [
        {"id": "c1", "content": "reset the widget", "score": 0.88, "metadata": {"source": "manual.pdf"}},
        {"text": "then reboot", "relevance": 0.4},
    ]}
    r = GradientKBRetriever(_session(resp, cap), knowledge_base_id="kb-123")
    assert isinstance(r, base.RetrieverPlugin)
    hits = r.search("widget", k=5, method="hybrid")
    assert hits[0].chunk_id == "c1" and hits[0].text == "reset the widget"
    assert hits[0].filename == "manual.pdf" and hits[0].source == "gradient_kb" and hits[0].score == 0.88
    assert hits[1].text == "then reboot"        # tolerates the alternate field names
    call = cap[0]
    assert call["url"] == "https://kbaas.do-ai.run/v1/kb-123/retrieve"
    assert call["headers"]["Authorization"] == "Bearer tok"        # KB uses the DO API token
    assert call["body"] == {"query": "widget", "k": 5}


# ── provider wiring ───────────────────────────────────────────────────────────────────────────────
def test_do_provider_registers_and_lean_caps_are_none():
    p = get_provider("digitalocean")
    assert p.name == "digitalocean"
    # lean platform: no serverless warehouse, no standalone guardrail, no identity broker
    assert p.analytical_backend() is None
    assert p.guardrail() is None
    assert p.identity_broker() is None
    assert p.managed_kb_retriever() is None       # no KB id configured


def test_do_alias_resolves():
    assert get_provider("do").name == "digitalocean"


def test_do_kb_wires_as_document_arm_in_router():
    from context_runtime.adapters.store_router import HopRouterRetriever
    from context_runtime.adapters.store_hipporag import SimGraphRetriever
    from context_runtime.providers.digitalocean.gradient_kb_retriever import GradientKBRetriever
    resp = {"results": [{"id": "c1", "content": "x", "score": 0.5}]}
    r = GradientKBRetriever(_session(resp), knowledge_base_id="kb-1")
    router = HopRouterRetriever(single_hop=r, graph=SimGraphRetriever([]))
    hits = router.search("q", k=3, method="hybrid")
    assert hits and hits[0].source == "gradient_kb"
