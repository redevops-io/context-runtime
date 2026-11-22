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


def _refuse(reason: str, *, context: str, source: str) -> str:
    """A local-authority reason, checked. Raises rather than accepting an empty or bare-truthy declaration."""
    text = (reason or "").strip()
    where = f" ({context})" if context else ""
    if not text:
        raise RegistryConfigurationError(
            f"local authority was requested{where} without a reason. {source} takes a sentence describing "
            f"why this process may act with the operator's own authority — the reason is what makes an "
            f"allow-all choice reviewable in a manifest or a pull request.")
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


def resolve_authorizer(authorizer: Optional[Authorizer] = None, *, local_trust: bool = False,
                       local_trust_reason: str = "", context: str = "",
                       on_missing: str = "raise") -> Optional[Authorizer]:
    """The authorizer one Context Runtime registry should be built with, or ``None`` to inherit the default.

    ``authorizer`` — the integrator's per-registry choice — wins outright, unexamined: it is code, and code may
    pass anything callable. Otherwise an installed process-wide plane is inherited (``None``). Otherwise local
    authority is used, but only when it was declared — ``local_trust=True`` with a written
    ``local_trust_reason``, or ``CR_LOCAL_AUTHORITY="<reason>"``. Otherwise this raises: a tenant that reaches
    this point with no answer is a misconfiguration, and the least dangerous place to say so is construction.

    ``on_missing`` decides what "no answer" means. ``"raise"`` (the default, and right for a tenant that owns
    its registry) makes it a start-up failure. ``"deny_later"`` returns ``None`` and lets the registry answer
    every call with its own ``missing_authorizer`` denial — what a shared, request-facing surface such as
    ``AgentConsole`` wants, because there the missing plane is a per-request fact the operator can still fix,
    and the denial is already asserted in downstream tests. Neither mode grants local trust implicitly.
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
    if on_missing == "deny_later":
        return None
    if on_missing != "raise":
        raise RegistryConfigurationError(
            f"on_missing={on_missing!r} is not one of 'raise' / 'deny_later' ({context or 'caller'})")
    raise RegistryConfigurationError(
        f"no authorizer is configured{f' for {context}' if context else ''}, and Context Runtime will not "
        f"guess. Do one of: pass authorizer=... (a permissions plane, e.g. "
        f"agentic_os.permissions.make_authorizer(store, tool_actions=registry.classification())); install one "
        f"process-wide with context_runtime.tools.set_default_authorizer(...) before building the tenant; or "
        f"declare local execution explicitly with local_trust=True and a written local_trust_reason, or "
        f'{LOCAL_AUTHORITY_ENV}="why this process is trusted".')