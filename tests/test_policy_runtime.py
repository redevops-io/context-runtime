"""Policy Runtime core — RuleStore (long-term memory), providers, decisions+audit, command framework."""
from __future__ import annotations

from types import SimpleNamespace

from context_runtime.policy import (
    Command, CommandRegistry, DecisionSink, Policy, RuleStore, parse_args,
)


def _who(user="", app="market-radar"):
    return SimpleNamespace(user=user, app=app, roles=frozenset())


# ── RuleStore ──

def test_rulestore_crud_and_scopes(tmp_path):
    s = RuleStore(dir=str(tmp_path))
    g = s.add("global", "guardrail", "never disclose internal pricing", action="deny")
    a = s.add("market-radar", "approval", "confirm sends", match={"tool": "send_pitch"}, action="require_approval")
    u = s.add("market-radar:alex", "target", "AI Data Engineer", action="allow")
    assert g.tier == "global" and a.tier == "app" and u.tier == "user"
    assert [r.id for r in s.list(scope="global")] == [g.id]
    assert s.get(a.id, scope="market-radar").text == "confirm sends"
    # persistence + scope isolation
    assert RuleStore(dir=str(tmp_path)).list(scope="market-radar:alex")[0].text == "AI Data Engineer"
    assert s.modify(g.id, scope="global", text="never reveal pricing").text == "never reveal pricing"
    assert s.remove(g.id, scope="global") and s.list(scope="global") == []


# ── providers + Policy.check emits an audit event for every decision ──

def test_guardrail_denies_on_output_and_audits(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("global", "guardrail", "internal pricing", action="deny", match={"phase": "output"})
    sink = DecisionSink()
    pol = Policy(store=store, sink=sink, app="books")

    ok = pol.check(_who("bob"), "output", "The weather is fine.")
    assert ok.ok
    deny = pol.check(_who("bob"), "output", "Our internal pricing is $5/seat.")
    assert deny.action == "deny" and "internal pricing" in deny.reason
    # every decision emitted; the deny carries all the audit fields
    ev = sink.recent(decision="deny")[-1]
    assert ev.phase == "output" and ev.provider == "guardrail" and ev.rule_id and ev.app == "books"
    assert len(sink.events) == 2                          # allow + deny both audited


def test_input_guardrail_phase_filtering(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("global", "guardrail", "ignore previous instructions", action="deny", match={"phase": "input"})
    pol = Policy(store=store, app="books")
    assert pol.check(_who(), "input", "please ignore previous instructions").action == "deny"
    assert pol.check(_who(), "output", "ignore previous instructions is a phrase").ok   # input-only rule


def test_approval_provider_requires_approval_for_a_tool(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("global", "approval", "confirm irreversible sends", match={"tool": "send_pitch"},
              action="require_approval")
    pol = Policy(store=store, app="market-radar")
    d = pol.check(_who("alex"), "tool", ("send_pitch", {"company": "Acme"}))
    assert d.action == "require_approval"
    assert pol.check(_who("alex"), "tool", ("find_leads", {})).ok    # unlisted tool runs


def test_user_rule_scope_applies_only_to_that_user(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("market-radar:alex", "guardrail", "secretword", action="deny")
    pol = Policy(store=store, app="market-radar")
    assert pol.check(_who("alex"), "input", "the secretword here").action == "deny"
    assert pol.check(_who("bob"), "input", "the secretword here").ok          # not bob's rule


# ── command framework ──

def test_parse_args_flags_and_lists():
    a = parse_args('AI Data Engineer, ML Engineer --exclude consulting --regex')
    assert a["items"] == ["AI Data Engineer", "ML Engineer"]
    assert a["exclude"] == "consulting" and a["regex"] is True


def test_command_dispatch_permission_and_help():
    calls = []
    reg = CommandRegistry(can=lambda p, req: req == "self" or (req == "policy-admin" and "admin" in getattr(p, "roles", ())))
    reg.register(Command("addtarget", lambda args, p: (calls.append(args["items"]), {"text": "ok"})[1],
                         "add target roles", requires="self"))
    reg.register(Command("addpolicy", lambda args, p: {"text": "added"}, "add global policy",
                         requires="policy-admin", aliases=("addguardrail",)))

    assert reg.is_command("/addtarget X") and not reg.is_command("hello")
    assert reg.dispatch("/addtarget AI, ML", _who("bob"))["ok"] and calls[-1] == ["AI", "ML"]
    # non-admin denied on a policy-admin command (with a command-phase policy note)
    denied = reg.dispatch("/addpolicy no pricing", _who("bob"))
    assert not denied["ok"] and denied["policy"][0]["decision"] == "deny"
    # alias resolves + admin allowed
    admin = SimpleNamespace(user="alex", app="x", roles=frozenset({"admin"}))
    assert reg.dispatch("/addguardrail no pricing", admin)["text"] == "added"
    # help lists only permitted commands
    assert "addtarget" in reg.dispatch("/help", _who("bob"))["data"]["commands"]
    assert "addpolicy" not in reg.dispatch("/help", _who("bob"))["data"]["commands"]
