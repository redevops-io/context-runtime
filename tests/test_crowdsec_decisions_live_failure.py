"""A configured CrowdSec that cannot be read is an error. The simulated feed names addresses; it is shown only where no CrowdSec is
configured, and it says that it is simulated."""
import io
import json
import urllib.error

import pytest

from context_runtime.integrations import edge_sentinel as es

TOOL = next(v for k, v in vars(es).items() if isinstance(v, type) and hasattr(v, "_live") and hasattr(v, "_sim"))
SIMULATED = {r["value"] for r in TOOL._sim("")}


def _tool(monkeypatch, behaviour):
    monkeypatch.setattr(es.urllib.request, "urlopen", behaviour)
    return TOOL(lapi_url="http://crowdsec.example.test:8080", bouncer_key="k")


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _raises(exc):
    def f(req, timeout=10): raise exc
    return f


@pytest.mark.parametrize("behaviour,word", [
    (_raises(urllib.error.URLError("connection refused")), "unreachable"),
    (_raises(TimeoutError("timed out")), "unreachable"),
    (_raises(urllib.error.HTTPError("http://x", 403, "forbidden", {}, None)), "403"),
    (lambda req, timeout=10: _Resp(b"<html>gateway</html>"), "not JSON"),
    (lambda req, timeout=10: _Resp(b'{"message": "no"}'), "not a list"),
])
def test_a_configured_crowdsec_that_cannot_be_read_is_an_error_and_shows_no_decision(monkeypatch, behaviour, word):
    out = _tool(monkeypatch, behaviour).run({"query": ""})
    assert out.ok is False and out.hits == [] and out.data == []
    assert word in (out.error or "") and "No decisions are shown" in out.text
    assert not any(ip in json.dumps(out.data) + out.text for ip in SIMULATED)        # nothing from the simulated feed


def test_a_configured_crowdsec_with_no_decisions_answers_an_empty_list_not_the_simulated_one(monkeypatch):
    out = _tool(monkeypatch, lambda req, timeout=10: _Resp(b"null")).run({})
    assert out.ok is True and out.data == [] and out.hits == [] and "simulated" not in out.text


def test_a_configured_crowdsec_returns_its_own_decisions(monkeypatch):
    body = json.dumps([{"type": "ban", "value": "198.51.100.7", "scenario": "crowdsecurity/ssh-bf", "scope": "Ip", "duration": "4h"}]).encode()
    out = _tool(monkeypatch, lambda req, timeout=10: _Resp(body)).run({})
    assert out.ok and [r["value"] for r in out.data] == ["198.51.100.7"] and "simulated" not in out.text and "simulated" not in out.data[0]


def test_without_a_configured_crowdsec_the_feed_is_simulated_and_says_so(monkeypatch):
    monkeypatch.delenv("CROWDSEC_LAPI_URL", raising=False); monkeypatch.delenv("CROWDSEC_BOUNCER_KEY", raising=False)
    out = TOOL(lapi_url="", bouncer_key="").run({})
    assert out.ok and out.data and all(r.get("simulated") is True for r in out.data)
    assert "simulated" in out.text and all(h.text.startswith("[simulated] ") for h in out.hits)
