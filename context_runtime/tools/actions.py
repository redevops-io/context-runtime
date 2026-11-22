"""What risk class a Context Runtime tool is registered with — and who is allowed to set it.

:mod:`context_runtime.tools.authz` answers who authorizes a call. This answers the other question a
registration has to settle: which capability the call will be checked for. Both are answered in one
place so that the integration tenants cannot each invent an answer, and both refuse to guess.

The rule is the one the whole classification design rests on: **the class describes the effect of the
code that runs.** It is set by whoever read that code, and it is never inherited from the name the tool
is filed under, from a remote server's ``readOnlyHint``, or from a previous occupant of the slot. The
seam this module guards is the last of those three: every integration tenant ships a local simulation of
one tool — it builds a string and returns it, so it is a ``read`` — and lets an integrator replace it
with ``*_tool_factory``. A replacement that posts to a vendor, stores a row or launches a scan is not a
read, and inheriting the class of the function it replaced would hand that call to anyone holding a read
grant.

So: nothing injected, the reviewed class of the shipped implementation stands. Something injected, the
integrator declares the class of their own function, and construction refuses without it — the same
posture as :func:`~context_runtime.tools.authz.resolve_authorizer` and as ``mount_mcp(actions=...)``,
where an unmapped MCP tool falls to ``write`` rather than to the server's own claim.
"""
from __future__ import annotations

from typing import Optional

from .base import ACTIONS, RegistryConfigurationError


def resolve_action(*, shipped: str, injected: Optional[object] = None, declared: Optional[str] = None,
                   param: str = "", tool: str = "", context: str = "") -> str:
    """The risk class of the function that is actually installed, which is not always the shipped one's.

    ``shipped`` is the class this file reviewed for the implementation it ships; ``injected`` is the
    caller's replacement function, or ``None`` when they took the shipped one; ``declared`` is the class
    the caller says their own function deserves, via the tenant's ``*_action`` keyword.

    Nothing injected returns ``shipped`` untouched. Something injected returns ``declared`` — and raises
    :class:`~context_runtime.tools.base.RegistryConfigurationError` when it is missing, because "the
    simulation was a read" is not evidence about a function this repository has never seen. A value
    outside :data:`~context_runtime.tools.base.ACTIONS` raises the same way, on either side: a typo is not
    a risk class, and a wrong class is worse than no class, because an unclassified tool is matched by no
    grant and so is denied at every call, while a wrongly-classified one runs on someone else's permission.
    """
    where = f"{tool} ({context})" if tool and context else (tool or context or "this tool")
    param = param or f"{tool.rsplit('.', 1)[-1] if tool else 'tool'}_action"
    if shipped not in ACTIONS:
        raise RegistryConfigurationError(
            f"{where}: the shipped risk class {shipped!r} is not one of the vocabulary {ACTIONS} — a bug "
            f"in this integration, not a value a caller can supply.")
    if injected is None:
        return shipped
    if declared is None:
        raise RegistryConfigurationError(
            f"{where}: a function was injected in place of the implementation shipped here, and a risk "
            f"class is not inherited from the tool it replaces. Pass {param}=... — one of {ACTIONS} — for "
            f"what the injected function actually does. If it fetches, posts, stores or launches anything, "
            f"that is not {shipped!r}.")
    if declared not in ACTIONS:
        raise RegistryConfigurationError(
            f"{where}: {param}={declared!r} is not one of the vocabulary {ACTIONS}.")
    return declared