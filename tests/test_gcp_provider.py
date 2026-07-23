"""GCP provider adapters (Gemini, Vertex AI Search, BigQuery, Model Armor), all with injected fakes."""
from types import SimpleNamespace

from context_runtime.plugins import base
from context_runtime.providers.base import Guardrail, get_provider
from context_runtime.types import ModelRequest


# ── Gemini model ────────────────────────────────────────────────────────────────────────────────
class FakeGenAI:
    def __init__(self):
        self.calls = []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text="gemini says hi",
                               usage_metadata=SimpleNamespace(prompt_token_count=9, candidates_token_count=4))


def test_gemini_model_maps_request_and_usage():
    from context_runtime.providers.gcp.gemini_model import GeminiModel
    from context_runtime.adapters.model_litellm import Tier
    fake = FakeGenAI()
    m = GeminiModel([Tier(name="cheap", model="gemini-2.0-flash", cost_per_1k=0.001)], client=fake)
    assert isinstance(m, base.ModelPlugin)
    res = m.complete(ModelRequest(messages=({"role": "user", "content": "hi"},), system="sys", max_tokens=64))
    assert res.text == "gemini says hi" and res.model == "gemini-2.0-flash"
    assert res.prompt_tokens == 9 and res.completion_tokens == 4
    call = fake.calls[0]
    assert call["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert call["config"]["system_instruction"] == "sys" and call["config"]["max_output_tokens"] == 64


# ── Vertex AI Search ───────────────────────────────────────────────────────────────────────────
class FakeDE:
    def search(self, request):
        self.request = request
        return [
            SimpleNamespace(relevance_score=0.9, document=SimpleNamespace(
                id="d1", derived_struct_data={"title": "Q3", "link": "gs://x/q3",
                                              "snippets": [{"snippet": "revenue grew 20%"}]})),
        ]


def test_vertex_search_maps_hits():
    from context_runtime.providers.gcp.vertex_search_retriever import VertexSearchRetriever
    sess = SimpleNamespace(project="p", location="us-central1")
    r = VertexSearchRetriever(sess, engine_id="eng", client=FakeDE())
    assert isinstance(r, base.RetrieverPlugin)
    hits = r.search("revenue", k=5, method="hybrid")
    assert hits[0].chunk_id == "d1" and hits[0].text == "revenue grew 20%"
    assert hits[0].source == "vertex_search" and hits[0].score == 0.9


# ── BigQuery analytical backend ──────────────────────────────────────────────────────────────────
class FakeBQJob:
    def __init__(self, rows):
        self._rows = rows

    def result(self, max_results=None):
        return self._rows[:max_results] if max_results else self._rows


class FakeBQ:
    def query(self, sql):
        self.sql = sql
        if "INFORMATION_SCHEMA" in sql:
            return FakeBQJob([{"table_name": "invoices", "column_name": "status", "data_type": "STRING"}])
        return FakeBQJob([{"status": "open", "n": 2}, {"status": "paid", "n": 1}])


def test_bigquery_backend_schema_and_query():
    from context_runtime.providers.gcp.bigquery_backend import BigQueryBackend
    be = BigQueryBackend(dataset="lake", project="p", client=FakeBQ())
    assert be.dialect() == "bigquery"
    assert "invoices(" in be.schema()
    rows = be.run_sql("SELECT status, count(*) n FROM invoices GROUP BY status", max_rows=10)
    assert rows == [{"status": "open", "n": 2}, {"status": "paid", "n": 1}]


def test_bigquery_analytical_end_to_end():
    from context_runtime.adapters.store_analytical import AnalyticalRetriever
    from context_runtime.providers.gcp.bigquery_backend import BigQueryBackend

    class SqlModel:
        def complete(self, req):
            from context_runtime.types import ModelResult
            return ModelResult(text="SELECT status, count(*) n FROM invoices GROUP BY status",
                               model="m", tier="cheap")
        def capabilities(self, m):
            from context_runtime.types import ModelCapabilities
            return ModelCapabilities()
        def count_tokens(self, t, m):
            return 1
        def info(self):
            from context_runtime.types import PluginInfo
            return PluginInfo(name="m", kind="model")

    r = AnalyticalRetriever(BigQueryBackend(dataset="lake", project="p", client=FakeBQ()), SqlModel())
    hits = r.search("per status", k=10, method="sql")
    assert {h.meta["row"]["status"] for h in hits} == {"open", "paid"}
    assert all(h.source == "analytical:bigquery" for h in hits)


# ── Model Armor guardrail ─────────────────────────────────────────────────────────────────────────
class FakeArmor:
    def __init__(self, state):
        self.state = state

    def sanitize_user_prompt(self, request):
        return SimpleNamespace(sanitization_result=SimpleNamespace(filter_match_state=self.state))

    def sanitize_model_response(self, request):
        return SimpleNamespace(sanitization_result=SimpleNamespace(filter_match_state=self.state))


def test_model_armor_allows_and_blocks():
    from context_runtime.providers.gcp.model_armor import ModelArmorGuardrail
    ok = ModelArmorGuardrail(template="t", client=FakeArmor("NO_MATCH_FOUND"))
    bad = ModelArmorGuardrail(template="t", client=FakeArmor("MATCH_FOUND"))
    assert isinstance(ok, Guardrail)
    assert ok.check_input("hi").allowed is True
    v = bad.check_output("bad")
    assert v.allowed is False and v.action == "blocked" and "model_armor" in v.reasons


# ── provider wiring ───────────────────────────────────────────────────────────────────────────────
def test_gcp_provider_registers_and_unconfigured_caps_none():
    p = get_provider("gcp")
    assert p.name == "gcp"
    assert p.document_retriever() is None      # no engine/data_store configured
    assert p.analytical_backend() is None       # no dataset
    assert p.guardrail() is None                # no template
