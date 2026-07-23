"""OpenSearchRetriever — a document ``RetrieverPlugin`` over Amazon OpenSearch (Serverless or managed).

Registers OpenSearch as a document-representation retriever the KR router and bandit can select
alongside the local BM25/hybrid arms — so Context Runtime *learns* when OpenSearch beats the in-tree
engines for a query class, rather than the deployment hard-wiring it.

The data-plane client is injectable (duck-typed: ``.search(index=..., body=...) -> dict``). The real
path lazily builds a SigV4-signed ``opensearch-py`` client from an ``AwsSession`` (needs the
``opensearch-py`` package); tests pass a fake and never touch the network. Vector/kNN retrieval needs
an embedding-configured index; the default query is a lexical ``multi_match`` over the text field, so
it works on any index out of the box.
"""
from __future__ import annotations

from ...types import Hit, PluginInfo, Retrieval


class OpenSearchRetriever:
    def __init__(self, session=None, *, endpoint: str | None = None, index: str = "documents",
                 client=None, text_field: str = "text", filename_field: str = "filename"):
        self._session = session
        self.endpoint = endpoint
        self.index = index
        self.text_field = text_field
        self.filename_field = filename_field
        self._client = client

    def _os_client(self):
        if self._client is not None:
            return self._client
        # lazy, optional: opensearch-py + botocore SigV4. Kept out of the base install.
        from urllib.parse import urlparse

        from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

        sess = self._session._effective_session()  # boto3 session (creds resolved / role assumed)
        creds = sess.get_credentials()
        region = self._session.region
        # 'aoss' for Serverless collections, 'es' for managed domains
        service = "aoss" if "aoss" in (self.endpoint or "") else "es"
        host = urlparse(self.endpoint).netloc or self.endpoint
        self._client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=AWSV4SignerAuth(creds, region, service),
            use_ssl=True, verify_certs=True, connection_class=RequestsHttpConnection,
        )
        return self._client

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        body = {"size": k, "query": {"multi_match": {"query": query, "fields": [self.text_field]}}}
        resp = self._os_client().search(index=self.index, body=body)
        out: list[Hit] = []
        for h in (resp.get("hits", {}) or {}).get("hits", []) or []:
            src = h.get("_source", {}) or {}
            out.append(Hit(
                chunk_id=str(h.get("_id", "")),
                filename=str(src.get(self.filename_field, self.index)),
                text=str(src.get(self.text_field, "")),
                score=float(h.get("_score") or 0.0),
                created_at=src.get("created_at"),
                source="opensearch",
                meta={kk: vv for kk, vv in src.items() if kk not in (self.text_field,)},
            ))
        return out

    def index(self, path: str) -> dict:  # OpenSearch is populated out-of-band (ingest pipeline / bulk)
        return {"opensearch": "index managed externally", "index": self.index}

    def info(self) -> PluginInfo:
        return PluginInfo(name="opensearch", kind="retriever",
                          capabilities=frozenset({"vector", "bm25", "hybrid"}))
