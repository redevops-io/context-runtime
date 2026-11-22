"""Who authorizes a Context Runtime tool call — and how a local run says so out loud.

Context Runtime owns the seam (``ToolRegistry.authorizer``, defined upstream in ``agent_harness.tools``) and the
vocabulary a registration declares its risk in (``ToolSpec.action``). It deliberately does NOT own a policy engine:
the engine is whoever installs an authorizer. Three engines plug into the same seam today —
``agentic_os.permissions.make_authorizer(store, tool_actions=...)`` in a governed deployment, an operator's own
callable in a customer-controlled install, and :func:`~context_runtime.tools.base.local_trusted_authorizer` for a
single-operator run — and this module is the only place that chooses between them, so the twelve integration
tenants cannot each invent their own answer.

**Three doors, in precedence order** (``resolve_authorizer``):

1. *Per-registry*: the caller passed ``authorizer=...``. The integrator's explicit choice always wins.
2. *Process-wide*: an authorizer is installed with ``set_default_authorizer(...)`` — what
   ``agentic_os.app_gate.install_gate`` does at start-up. Returning ``None`` here means "inherit it", read at
   call time, so a plane installed after the tenant was built still governs it.
3. *Local authority*: the operator declared this process runs locally, on their own machine or their own
   single-tenant box. That is the harness's local-trust authorizer: an attributed, greppable wildcard grant,
   not a silent hole. The first use warns once per process and every decision is audited as
   ``plane="local_trusted"``.

**Missing configuration is not local trust.** With no authorizer, no installed plane and no declared local
authority, ``resolve_authorizer`` raises :class:`~context_runtime.tools.base.RegistryConfigurationError` at
construction, naming all three doors. Letting it pass would build a registry that answers every call with
``denied: missing_explicit_grant`` — correct, but discovered by a user instead of in a start-up log.

A service that serves requests does not get to discover it there either: it calls :func:`require_plane` before
it stands anything up, and a process that cannot answer its own tool calls does not start. The exception is
the mode in which there is nothing to answer — :data:`EXECUTION_MODE_ENV` — and not a warning beside the
denials, which is what used to be here and read as healthy.

Local authority is the FLOOR, never the ceiling: it is consulted only when neither of the first two doors is
open. A copied example with ``local_trust=True`` inside a service that installed a permissions plane therefore
cannot loosen that plane; the plane wins.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import Authorizer, RegistryConfigurationError, default_authorizer_installed, local_trusted_authorizer

#: Set this to a sentence — not to ``1`` — to declare that the process runs with the operator's own authority
#: ("single-operator laptop demo", "offline benchmark, simulated tools"). The reason is required because a
#: value that says nothing is how an allow-all switch gets copied into a production compose file.
LOCAL_AUTHORITY_ENV = "CR_LOCAL_AUTHORITY"

#: Values people reach for when switching something on. They carry no reason, so they are refused.
_BARE_TRUTHY = {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}
#: What a process reads to declare that it is NOT a privileged execution service. Absent means ``execute``:
#: a service that runs tools has to be told, in words, that it is allowed to come up without a plane.
EXECUTION_MODE_ENV = "CR_EXECUTION_MODE"

#: And why it is a diagnostic instance. A sentence for the same reason :data:`LOCAL_AUTHORITY_ENV` takes one:
#: a bare switch is what turns an escape hatch into a default, and this one is the difference between a
#: read-only console and a production service that quietly stopped being able to act.
DIAGNOSTIC_REASON_ENV = "CR_DIAGNOSTIC_REASON"

EXECUTE = "execute"
DIAGNOSTIC = "diagnostic"

#: The words that mean each mode. Spelled out because a deployment that types something else must be told,
#: not quietly routed into the permissive branch — which is the mistake that makes a mode switch useless.
_MODE_WORDS = {
    EXECUTE: {EXECUTE, "privileged", "full", "production"},
    DIAGNOSTIC: {DIAGNOSTIC, "diagnostics", "read-only", "read_only", "readonly"},
}


class PrivilegedExecutionUnavailable(RuntimeError):
    """Something asked a diagnostic instance to execute. Refused by construction, not by policy.

    A separate exception type from :class:`RegistryConfigurationError` because the two answers mean different
    things to a caller: a configuration error says the deployment is wrong and should not have started, this
    says the deployment is fine and this operation does not exist in it.
    """


def _refuse(reason: str, *, context: str, source: str,
            what: str = "local authority") -> str:
    """A reason that has to mean something. Raises rather than accepting an empty or bare-truthy declaration.

    Shared by the two places in this module that take a sentence rather than a switch — local authority and
    the diagnostic mode — because both fail the same way: a value that says nothing is how an escape hatch
    gets copied into a production compose file.
    """
    text = (reason or "").strip()
    where = f" ({context})" if context else ""
    if not text:
        raise RegistryConfigurationError(
            f"{what} was requested{where} without a reason. {source} takes a sentence describing why this "
            f"process is allowed to be the exception — the reason is what makes the choice reviewable in a "
            f"manifest or a pull request.")
    if text.lower() in _BARE_TRUTHY:
        raise RegistryConfigurationError(
            f"{source}={text!r} is not a reason{where}. Give it a sentence, e.g. "
            f'{source}="single-operator local demo, no tenant in scope".')
    return text


def local_authority(reason: str, *, context: str = "", source: str = "local_trust_reason") -> Authorizer:
    """The explicit local-authority authorizer. ``reason`` is mandatory; see :func:`resolve_authorizer`."""
    _refuse(reason, context=context, source=source)
    return local_trusted_authorizer()


def local_authority_from_env(*, context: str = "") -> tuple[bool, str]:
    """``(declared, reason)`` from :data:`LOCAL_AUTHORITY_ENV`. Unset is ``(False, "")`` — never a trust grant."""
    raw = (os.environ.get(LOCAL_AUTHORITY_ENV) or "").strip()
    if not raw:
        return False, ""
    return True, _refuse(raw, context=context, source=LOCAL_AUTHORITY_ENV)


def execution_mode(*, context: str = "") -> tuple[str, str]:
    """``("execute", "")`` or ``("diagnostic", reason)`` — whether this process may run tools at all.

    The default is the privileged answer, on purpose: a service that can act is the ordinary case, and the
    read-only instance is the exception that has to be declared. A diagnostic instance must give a reason, and
    an unrecognised value raises instead of falling through, because a typo in a mode variable that lands on
    the permissive branch is indistinguishable from a deployment that meant to be privileged.
    """
    raw = (os.environ.get(EXECUTION_MODE_ENV) or "").strip().lower()
    if not raw or raw in _MODE_WORDS[EXECUTE]:
        return EXECUTE, ""
    if raw in _MODE_WORDS[DIAGNOSTIC]:
        return DIAGNOSTIC, _refuse(os.environ.get(DIAGNOSTIC_REASON_ENV) or "",
                                   context=context, source=DIAGNOSTIC_REASON_ENV,
                                   what="a diagnostic (read-only) instance")
    raise RegistryConfigurationError(
        f"{EXECUTION_MODE_ENV}={raw!r} is not a mode. Use {EXECUTE!r} (the default: this process runs tools, "
        f"so it refuses to start without an authorization plane) or {DIAGNOSTIC!r} together with a written "
        f'{DIAGNOSTIC_REASON_ENV}="why this instance exists and cannot execute".')


def require_plane(*, context: str = "") -> str:
    """The one call a privileged service makes at start-up. Returns the mode; raises when it may not serve.

    A service that runs tools and has nothing to ask is not healthy-but-degraded: it is a service that will
    answer every action with ``missing_explicit_grant`` while its dashboard stays green. That state is a
    start-up fact, so it is decided at start-up — where the operator is still reading the log — and the only
    way past it without a plane is the mode in which execution does not exist, not a warning.

    Call this before standing anything up. It is deliberately not called by ``resolve_authorizer`` on behalf
    of every caller: a library does not know whether the process embedding it serves requests.
    """
    mode, reason = execution_mode(context=context)
    if mode == DIAGNOSTIC:
        return DIAGNOSTIC
    if default_authorizer_installed():
        return EXECUTE
    declared, _ = local_authority_from_env(context=context)
    if declared:
        return EXECUTE
    where = f" ({context})" if context else ""
    raise RegistryConfigurationError(
        f"a privileged execution service is refusing to start{where}: no authorization plane is installed and "
        f"no local authority is declared, so every tool call it would serve is denied "
        f"(missing_explicit_grant) while its health check reads green. Do one of three things: install the "
        f"plane before this service starts (context_runtime.tools.set_default_authorizer(...)); declare local "
        f'authority with {LOCAL_AUTHORITY_ENV}="why this process may act as the operator"; or run it as the '
        f"read-only instance it actually is with "
        f'{EXECUTION_MODE_ENV}={DIAGNOSTIC} {DIAGNOSTIC_REASON_ENV}="what this instance is for". A warning '
        f"here was tried and rejected — a fleet that denies everything looks exactly like a healthy one.")


def resolve_authorizer(authorizer: Optional[Authorizer] = None, *, local_trust: bool = False,
                       local_trust_reason: str = "", context: str = "",
                       on_missing: str = "raise", privileged: bool = True) -> Optional[Authorizer]:
    """The authorizer one Context Runtime registry should be built with, or ``None`` to inherit the default.

    ``authorizer`` — the integrator's per-registry choice — wins outright, unexamined: it is code, and code may
    pass anything callable. Otherwise an installed process-wide plane is inherited (``None``). Otherwise local
    authority is used, but only when it was declared — ``local_trust=True`` with a written
    ``local_trust_reason``, or ``CR_LOCAL_AUTHORITY="<reason>"``. Otherwise this raises: a tenant that reaches
    this point with no answer is a misconfiguration, and the least dangerous place to say so is construction.

    ``on_missing`` decides what "no answer" means, and ``privileged`` decides who is allowed to ask for the
    second answer. ``"raise"`` (the default) makes it a start-up failure. ``"deny_later"`` returns ``None`` and
    lets the registry answer every call with its own ``missing_authorizer`` denial, and it is refused unless
    the caller also says ``privileged=False``: only a surface that cannot reach a tool may be built on the
    promise that the call path will say no, because "boots and denies" and "healthy" look identical from the
    outside — which is the failure this seam was migrated to remove. ``"defer_to_plane"`` also returns
    ``None``, and means something different: *this is a component, not a service* — the application embedding
    it installs a plane moments later and is the thing that must refuse to serve if one never arrives
    (``agentic_os.app_gate.validate_console``). A component may be built before its plane exists; a service
    may not serve without one. Neither mode grants local trust implicitly.
    """
    if authorizer is not None:
        if not callable(authorizer):
            raise RegistryConfigurationError(
                f"authorizer{f' for {context}' if context else ''} is {type(authorizer).__name__}, not a "
                f"callable (principal, spec, args) -> decision")
        return authorizer
    if default_authorizer_installed():
        return None                                   # inherit the plane; re-read at every call
    env_declared, env_reason = local_authority_from_env(context=context)
    if local_trust:
        reason, source = ((local_trust_reason, "local_trust_reason") if local_trust_reason
                          else (env_reason, LOCAL_AUTHORITY_ENV))
        return local_authority(reason, context=context, source=source)
    if env_declared:
        return local_authority(env_reason, context=context, source=LOCAL_AUTHORITY_ENV)
    if on_missing == "defer_to_plane":
        return None                                   # the embedding service owns readiness; see require_plane
    if on_missing == "deny_later":
        if privileged:
            raise RegistryConfigurationError(
                f"on_missing='deny_later' is refused{f' for {context}' if context else ''} for a registry that "
                f"can execute. It buys a process that starts, serves, and denies every tool call forever — "
                f"indistinguishable from a working one on a dashboard, and the exact state the fail-closed "
                f"migration was done to surface. Either build it with on_missing='raise' so the missing plane "
                f"stops the start-up, or say what this surface really is: privileged=False, meaning it has no "
                f"execution path for the denial to be about.")
        return None
    if on_missing != "raise":
        raise RegistryConfigurationError(
            f"on_missing={on_missing!r} is not one of 'raise' / 'deny_later' / 'defer_to_plane' "
            f"({context or 'caller'})")
    raise RegistryConfigurationError(
        f"no authorizer is configured{f' for {context}' if context else ''}, and Context Runtime will not "
        f"guess. Do one of: pass authorizer=... (a permissions plane, e.g. "
        f"agentic_os.permissions.make_authorizer(store, tool_actions=registry.classification())); install one "
        f"process-wide with context_runtime.tools.set_default_authorizer(...) before building the tenant; or "
        f"declare local execution explicitly with local_trust=True and a written local_trust_reason, or "
        f'{LOCAL_AUTHORITY_ENV}="why this process is trusted".')