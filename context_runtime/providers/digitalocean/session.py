"""DigitalOcean session — endpoints + tokens for Gradient serverless inference and the KB retrieve API.

No SDK required: both DO surfaces are plain HTTPS (inference is OpenAI-compatible; the knowledge base
exposes a REST ``/retrieve``). The HTTP transport is injectable so tests never touch the network.

Auth:
  • inference (https://inference.do-ai.run/v1) uses a **model access key** (DO_INFERENCE_KEY).
  • the knowledge base (https://kbaas.do-ai.run/v1) uses a **DO API token** with GenAI:read
    (DIGITALOCEAN_TOKEN).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _urllib_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode("utf-8", "ignore")
        raise RuntimeError(f"digitalocean http {e.code}: {detail}") from e


class DoSession:
    def __init__(self, *, api_token: str | None = None, inference_key: str | None = None,
                 inference_base: str = "https://inference.do-ai.run/v1",
                 kb_base: str = "https://kbaas.do-ai.run/v1", transport=None, timeout: float = 60.0):
        self.api_token = api_token or os.environ.get("DIGITALOCEAN_TOKEN") or os.environ.get("DO_API_TOKEN")
        self.inference_key = inference_key or os.environ.get("DO_INFERENCE_KEY") \
            or os.environ.get("GRADIENT_MODEL_ACCESS_KEY")
        self.inference_base = inference_base.rstrip("/")
        self.kb_base = kb_base.rstrip("/")
        self._transport = transport      # (url, body, headers, timeout) -> dict; injected in tests
        self.timeout = timeout

    def post(self, url: str, body: dict, headers: dict) -> dict:
        return (self._transport or _urllib_post)(url, body, headers, self.timeout)
