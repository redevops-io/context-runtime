"""web_search — a keyless, dependency-free MCP web-search server (stdio transport).

Exposes ONE read-only tool, ``web_search``, that searches the open web for a topic, company,
competitor, or trend and returns titles + URLs. Sources are keyless JSON APIs so it runs in a
slim container with no API key:

  * Wikipedia opensearch — reference/knowledge results
  * Hacker News (Algolia) — live discussion / launch / news results

Speaks newline-delimited JSON-RPC 2.0 (the MCP stdio transport) — one JSON message per line —
matching ``context_runtime.tools.mcp.MCPClient.stdio``. The tool is annotated ``readOnlyHint``
so, once mounted, the agent-harness ApprovalPolicy lets it run without a gate.

Run:  python -m context_runtime.tools.mcp_servers.web_search
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

_UA = {"User-Agent": "context-runtime-web-search/0.1"}
_TIMEOUT = 12.0


def _get(url: str) -> str:
    return urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=_TIMEOUT).read().decode("utf-8", "ignore")


def _wikipedia(query: str, k: int) -> list[dict]:
    try:
        d = json.loads(_get("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "opensearch", "search": query, "limit": k, "format": "json"})))
        return [{"title": t, "url": u, "source": "Wikipedia"} for t, u in zip(d[1], d[3])]
    except Exception:  # noqa: BLE001
        return []


def _hackernews(query: str, k: int) -> list[dict]:
    try:
        d = json.loads(_get("https://hn.algolia.com/api/v1/search?" + urllib.parse.urlencode(
            {"query": query, "hitsPerPage": k})))
        out: list[dict] = []
        for h in d.get("hits", [])[:k]:
            title = h.get("title") or h.get("story_title") or ""
            url = h.get("url") or h.get("story_url") or f"https://news.ycombinator.com/item?id={h.get('objectID', '')}"
            if title:
                out.append({"title": title, "url": url, "source": "HackerNews", "points": h.get("points")})
        return out
    except Exception:  # noqa: BLE001
        return []


def web_search(query: str, max_results: int = 6) -> str:
    query = (query or "").strip()
    if not query:
        return "Give me something to search for."
    k = max(1, min(int(max_results or 6), 10))
    results = (_wikipedia(query, 2) + _hackernews(query, k))[:k]
    if not results:
        return f"No web results found for: {query}"
    lines = [f'Web search — "{query}" ({len(results)} results):']
    for i, r in enumerate(results, 1):
        extra = f" · {r['points']} pts" if r.get("points") else ""
        lines.append(f"{i}. {r['title']} [{r['source']}{extra}]\n   {r['url']}")
    return "\n".join(lines)


_TOOLS = [{
    "name": "web_search",
    "description": ("Search the open web (Wikipedia + Hacker News) for information, discussions, or news "
                    "about a topic, company, competitor, or trend. Returns titles and URLs."),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "max results to return (default 6)"},
        },
        "required": ["query"],
    },
    "annotations": {"readOnlyHint": True, "openWorldHint": True},
}]


def _result(rid, result=None, error=None) -> dict:
    m: dict = {"jsonrpc": "2.0", "id": rid}
    if error is not None:
        m["error"] = error
    else:
        m["result"] = result
    return m


def _handle(req: dict) -> dict | None:
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}
    if method == "initialize":
        return _result(rid, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                             "serverInfo": {"name": "web-search", "version": "0.1.0"}})
    if method == "tools/list":
        return _result(rid, {"tools": _TOOLS})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name != "web_search":
            return _result(rid, {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True})
        try:
            text = web_search(args.get("query", ""), args.get("max_results", 6))
            return _result(rid, {"content": [{"type": "text", "text": text}], "isError": False})
        except Exception as e:  # noqa: BLE001
            return _result(rid, {"content": [{"type": "text", "text": f"search error: {e}"}], "isError": True})
    if rid is not None:  # unknown request (notifications have no id → ignored)
        return _result(rid, error={"code": -32601, "message": f"method not found: {method}"})
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        resp = _handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
