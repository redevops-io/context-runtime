"""AgentConsole ⇄ Policy Runtime: /command dispatch, input/output guardrails, tool approval, policy[]."""
from __future__ import annotations

from types import SimpleNamespace

from context_runtime.integrations.agent_console import AgentConsole
from context_runtime.policy import Command, CommandRegistry, Policy, RuleStore
from context_runtime.tools.base import ToolResult, function_tool


def _who(user="", app="market-radar"):
    return SimpleNamespace(user=user, app=app, roles=frozenset())


def test_slash_command_is_dispatched_without_the_model():
    reg = CommandRegistry()
    reg.register(Command("ping", lambda a, p: {"text": "pong " + a["text"]}, "ping back"))
    c = AgentConsole("T", "primer", commands=reg)
    assert c.respond("/ping there")["text"] == "pong there"


def test_keyword_fallback_routes_howto_to_help_not_a_tool():
    """A how-to question must not be keyword-matched into a side-effecting tool just because it
    shares an incidental word ('...part of a page?' vs the add_watch 'monitor a page' description)."""
    c = AgentConsole("Market Radar", "radar", tools=[
        function_tool("add_watch", lambda a: ToolResult(ok=True, text="ok"),
                      description="start monitoring a new page (non-destructive)")])
    howto = c._keyword_route("How do I track just part of a page?")
    assert howto["mode"] == "help" and howto["reason"] == "how-to → help"
    # explicit actions still keyword-match to the tool
    assert c._keyword_route("add a watch on this page")["mode"] == "tool"


def test_input_guardrail_blocks_and_reports(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("global", "guardrail", "forbidden phrase", action="deny", match={"phase": "input"})
    c = AgentConsole("T", "primer", policy=Policy(store=store, app="market-radar"), app="market-radar")
    r = c.respond("here is the forbidden phrase", principal=_who("bob"))
    assert r["intent"] == "blocked" and r["policy"][0]["decision"] == "deny" and r["policy"][0]["phase"] == "input"


def test_tool_requires_approval_and_is_not_run(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("global", "approval", "confirm irreversible sends", match={"tool": "send"}, action="require_approval")
    ran = []
    c = AgentConsole("T", "primer",
                     tools=[function_tool("send", lambda a: ran.append(1) or ToolResult(ok=True, text="sent"),
                                          side_effecting=True)],
                     policy=Policy(store=store, app="market-radar"), app="market-radar", allow_side_effects=["send"])
    out = c._answer_tool("send", {"to": "x"}, principal=_who("alex"))
    assert out["approved"] is False and out["policy"][0]["decision"] == "require_approval" and not ran   # not executed


def test_policy_summary_folds_allows(tmp_path):
    c = AgentConsole("T", "how things work", policy=Policy(store=RuleStore(dir=str(tmp_path)), app="x"), app="x")
    out = c.respond("how do I do the thing?", principal=_who("bob"))
    assert out["policy"] == [] and out["policy_checks"] >= 2      # input+output allow → folded, no chips


def test_backward_compatible_without_policy_or_commands():
    out = AgentConsole("T", "primer").respond("hello")           # no policy/commands ⇒ unchanged path
    assert "text" in out and "policy" not in out


def test_the_panel_refuses_a_mount_or_endpoint_that_would_add_markup_or_script():
    """``mount`` becomes element ids and ``endpoint`` a script string; the template escapes neither."""
    import pytest
    from context_runtime.integrations.agent_console import AgentConsole
    c = AgentConsole.__new__(AgentConsole)
    c.tenant, c.subtitle, c.suggestions = "Acme <b>", "sub", ["hi <i>"]
    html = c.panel_html()
    assert 'id="agent"' in html and 'EP="api/agent"' in html and "Acme &lt;b&gt;" in html and "hi &lt;i&gt;" in html
    assert "esc(e.n)" in html                                        # an evidence index is escaped like every other field
    assert 'EP="/t/1/api/agent?x=1&y=2"' in c.panel_html(mount="a-1", endpoint="/t/1/api/agent?x=1&y=2")
    for mount in ('a"><script>alert(1)</script>', "1x", "", "a b"):
        with pytest.raises(ValueError):
            c.panel_html(mount=mount)
    for endpoint in ('";alert(1);//', "</script><script>alert(1)</script>", "javascript:alert(1)", "a b", ""):
        with pytest.raises(ValueError):
            c.panel_html(endpoint=endpoint)
