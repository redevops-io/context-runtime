"""BedrockKBRetriever — a ``RetrieverPlugin`` over a Bedrock Knowledge Base (``Retrieve``).

A managed retrieval-augmented store as one more RetrieverPlugin. Wired alongside OpenSearch and the
local hybrid/graph/temporal arms, it becomes an option the KR router and bandit can select — the
mechanism behind the article's core claim: Context Runtime *learns* when a Bedrock KB beats
OpenSearch or a local engine for a given query class.

Uses the ``bedrock-agent-runtime`` ``retrieve`` call (retrieval only — generation stays with CR's
model plane and reasoner, so EXPLAIN + learning still apply). Client is injectable for tests.
"""
from __future__ import annotations

from ...types import Hit, PluginInfo, Retrieval


class BedrockKBRetriever:
    def __init__(self, session=None, *, knowledge_base_id: str, client=None):
        self._session = session
        self.knowledge_base_id = knowledge_base_id
        self._client = client

    def _kb(self):
        if self._client is None:
            self._client = self._session.client("bedrock-agent-runtime")
        return self._client

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        resp = self._kb().retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
        )
        out: list[Hit] = []
        for i, r in enumerate(resp.get("retrievalResults", []) or []):
            content = (r.get("content", {}) or {}).get("text", "")
            loc = r.get("location", {}) or {}
            # surface a stable, human-readable source location (S3 uri / web url / …) as the filename
            uri = (loc.get("s3Location", {}) or {}).get("uri") \
                or (loc.get("webLocation", {}) or {}).get("url") \
                or loc.get("type", self.knowledge_base_id)
            out.append(Hit(
                chunk_id=f"kb:{i}",
                filename=str(uri),
                text=str(content),
                score=float(r.get("score") or 0.0),
                source="bedrock_kb",
                meta={"location": loc, **(r.get("metadata", {}) or {})},
            ))
        return out

    def index(self, path: str) -> dict:  # a KB is ingested/synced in AWS, not by the runtime
        return {"bedrock_kb": "ingestion managed in AWS", "knowledge_base_id": self.knowledge_base_id}

    def info(self) -> PluginInfo:
        return PluginInfo(name="bedrock_kb", kind="retriever",
                          capabilities=frozenset({"vector", "hybrid"}))
