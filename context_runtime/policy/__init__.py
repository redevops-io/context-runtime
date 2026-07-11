"""Policy Runtime — the feasible-execution-space plane (commands, rules, enforcement).

See docs/policy-runtime.md. Policy defines the feasible execution space before planning and enforces
across phases; commands read/write rules (long-term memory); every decision emits a PolicyDecision.
"""
from __future__ import annotations

from .commands import Command, CommandRegistry, parse_args, role_gate
from .plane import (
    ACTIONS, ALLOW, ApprovalProvider, Decision, DecisionSink, GuardrailProvider, Policy, PolicyDecision,
    current_policy, set_default_policy,
)
from .store import Rule, RuleStore, rule_id, scopes_for
from .factories import global_policy_commands, policy_commands

__all__ = [
    "Rule", "RuleStore", "rule_id", "scopes_for",
    "Decision", "ALLOW", "ACTIONS", "PolicyDecision", "DecisionSink",
    "GuardrailProvider", "ApprovalProvider", "Policy", "set_default_policy", "current_policy",
    "Command", "CommandRegistry", "parse_args", "role_gate",
    "policy_commands", "global_policy_commands",
]
