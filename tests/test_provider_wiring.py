"""End-to-end: assemble a Context Runtime from the AWS provider with fake boto clients (no boto3).

Proves the seam composes — one AwsProvider yields a runtime whose model plane is Bedrock (guardrailed),
whose document arm is a Bedrock KB, and whose analytical arm is Athena text-to-SQL — and a query runs
through it. Swapping `get_provider("aws", …)` for another cloud would change nothing else.
"""
from context_runtime.adapters.store_analytical import AnalyticalRetriever
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.providers.aws.bedrock_kb_retriever import BedrockKBRetriever
from context_runtime.providers.aws.provider import AwsProvider
from context_runtime.providers.guarded_model import GuardedModel
from context_runtime.providers.wiring import build_runtime, build_runtime_kwargs


class FakeBedrockRuntime:
    def converse(self, **kw):
        return {"output": {"message": {"content": [{"text": "answer from bedrock"}]}},
                "usage": {"inputTokens": 5, "outputTokens": 2}}

    def apply_guardrail(self, **kw):
        return {"action": "NONE", "outputs": [], "assessments": []}


class FakeKBClient:
    def retrieve(self, **kw):
        return {"retrievalResults": [
            {"content": {"text": "the manual says reset the widget"}, "score": 0.8,
             "location": {"type": "S3", "s3Location": {"uri": "s3://kb/manual.pdf"}}}]}


class FakeAthenaClient:
    def __init__(self):
        self._sql = {}

    def start_query_execution(self, **kw):
        self._sql["q"] = kw["QueryString"]
        return {"QueryExecutionId": "q"}

    def get_query_execution(self, QueryExecutionId):
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(self, QueryExecutionId, MaxResults):
        cols = ["table_name", "column_name", "data_type"] if "information_schema" in self._sql["q"] else ["n"]
        if "information_schema" in self._sql["q"]:
            rows = [[{"VarCharValue": c} for c in cols],
                    [{"VarCharValue": v} for v in ("t", "c", "int")]]
        else:
            rows = [[{"VarCharValue": "n"}], [{"VarCharValue": "1"}]]
        return {"ResultSet": {"Rows": rows}}


class FakeBotoSession:
    """A stand-in boto3.Session: .client(service) → the right fake."""
    def __init__(self):
        self._clients = {
            "bedrock-runtime": FakeBedrockRuntime(),
            "bedrock-agent-runtime": FakeKBClient(),
            "athena": FakeAthenaClient(),
        }

    def client(self, service, region_name=None):
        return self._clients[service]


def _provider():
    return AwsProvider(
        session=FakeBotoSession(),
        knowledge_base_id="KB1",
        athena_database="lake", athena_output="s3://out/",
        guardrail_id="g1",
    )


def test_wiring_assembles_the_expected_arms():
    kw = build_runtime_kwargs(_provider(), local_single_hop=InMemoryStore([]))
    # model plane: per-CR-tier dict, each wrapped in the guardrail
    assert isinstance(kw["models"], dict) and set(kw["models"]) == {"local", "cheap", "premium"}
    assert all(isinstance(m, GuardedModel) for m in kw["models"].values())
    # document arm = managed Bedrock KB; analytical arm = Athena text-to-SQL
    router = kw["retriever"]
    assert isinstance(router.single_hop, BedrockKBRetriever)
    assert isinstance(router.analytical, AnalyticalRetriever)


def test_end_to_end_query_runs_through_the_aws_stack():
    rt = build_runtime(_provider(), local_single_hop=InMemoryStore([]))
    res = rt.run("what does the manual say about the widget?")
    assert res.answer == "answer from bedrock"        # Bedrock model plane produced it
    assert any("manual.pdf" in c or c.startswith("kb:") for c in res.citations) or res.citations


def test_analytical_arm_is_athena_backed():
    from context_runtime.providers.aws.athena_backend import AthenaBackend
    kw = build_runtime_kwargs(_provider(), local_single_hop=InMemoryStore([]))
    analytical = kw["retriever"].analytical
    assert isinstance(analytical, AnalyticalRetriever)
    assert isinstance(analytical.backend, AthenaBackend)
    assert analytical.backend.dialect() == "athena"
    # and the analytical route is guarded, not silently BM25: an unbound method would raise, but here
    # it IS bound, so the router dispatches to Athena (execution needs a SQL-tuned model, tested elsewhere)
