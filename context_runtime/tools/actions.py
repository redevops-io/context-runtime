"""What risk class a Context Runtime tool is registered with, and what it takes to claim one.

:mod:`context_runtime.tools.authz` answers who authorizes a call. This answers the other question a
registration has to settle: which capability the call will be checked for. Both live in one place so the
integration tenants cannot each invent an answer, and both refuse to guess.

**The rule.** The class describes the effect of the code that runs. It is set by whoever read that code, and
it is never inherited from the name the tool is filed under, from a remote server's ``readOnlyHint``, or from
a previous occupant of the slot. The seam this module guards is the last of those three: every integration
tenant ships a local simulation of one tool — it builds a string and returns it, so it is a ``read`` — and
lets an integrator replace it with ``*_tool_factory``. A replacement that posts to a vendor, stores a row or
launches a scan is not a read, and inheriting the class of the function it replaced would hand that call to
anyone holding a read grant.

**Why a declaration is not enough on its own.** "Trust me, mine really is a read" is the same failure wearing
a different hat: the code asking to run grades its own risk again, which is the thing this design refuses.
So an injected implementation starts at :data:`INJECTED_FLOOR` — anything swapped in from outside this
repository is assumed to leave the process until shown otherwise — and the only way below the floor is a
*capability contract*: a named constructor whose argument must be a sentence about why the function is inert,
which is what makes the claim reviewable in a manifest or a pull request rather than a bare flag. That claim
is then checked against the function's own source: :func:`write_sinks` looks for the sink vocabulary the
platform's own read-classification sweep (agentic-os FR-04) uses, and a function reaching one cannot be
registered below the class that sink implies, whatever its author declared.

**What this does not pretend to be.** The sweep reads the callable that was handed over — nested definitions
included — not whatever that callable goes on to call. A factory that delegates to a client library is
invisible to it, which is exactly why the floor is ``write`` and not ``read``: the check exists to catch the
honest mistake and to make a false claim expensive, not to stand in for a review.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional

from .base import ACTIONS, RegistryConfigurationError

#: The least class an injected implementation may be registered at without a capability contract. A function
#: this repository did not write gets the benefit of the doubt in the safe direction: assumed consequential.
INJECTED_FLOOR = "write"

#: The sink vocabulary of the platform's sweep over tools classified ``read`` (agentic-os FR-04), copied
#: literally so Context Runtime stays installable without the platform. Attribute form: a subprocess, a file
#: written, an HTTP verb that is not a fetch.
SINK_ATTRS = frozenset({
    "system", "popen", "Popen", "check_output", "check_call", "call", "run", "spawn", "write_text",
    "write_bytes", "writelines", "unlink", "rmtree", "copyfile", "makedirs", "mkdir", "touch",
    "post", "put", "delete", "patch", "urlopen", "send", "submit",
})

#: Verbs that mean different things on different objects: ``os.remove`` deletes a file, ``rows.remove(x)``
#: is a list and ``df.rename`` is a dataframe. They count as sinks only when what they are called on names a
#: filesystem — the alternative is a rule that flags ordinary code until people stop reading the classes.
FS_QUALIFIED_ATTRS = frozenset({"remove", "rename", "move", "rmdir", "truncate", "removedirs"})

#: The receivers that make a :data:`FS_QUALIFIED_ATTRS` verb a sink.
FS_RECEIVERS = frozenset({"os", "os_path", "shutil", "pathlib", "Path", "path", "fs", "filesystem"})

#: Bare-name form of the same vocabulary, again as FR-04 spells it.
SINK_NAMES = frozenset({"open", "_run", "run", "run_cmd", "Popen", "system"})

#: Context Runtime's addition, because a registry here is as likely to reach a database as a shell: the
#: transactional verbs that end with something having changed.
SINK_TX_ATTRS = frozenset({"commit", "save", "insert", "upsert", "store"})

#: The subset that forces the ``delete`` class rather than merely the floor: filesystem destruction with no
#: innocent reading. ``remove`` and ``delete`` stay in the sink set — enough to refuse a ``read`` claim — but
#: they are not here, because ``items.remove(x)`` and ``session.delete(row)`` are ordinary code that would
#: otherwise be pushed into asking for a delete grant it does not need. Over-declaring is allowed; forcing it
#: on innocent code is what makes people stop reading the classes.
DESTRUCTIVE_ATTRS = frozenset({"unlink", "rmtree", "rmdir"})

#: Everything reachable as an attribute, ambiguous verbs included — they are qualified by receiver below.
_ALL_ATTR_SINKS = SINK_ATTRS | SINK_TX_ATTRS | DESTRUCTIVE_ATTRS | FS_QUALIFIED_ATTRS

#: Values that occupy the place of a reason and carry none.
_BARE_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on", "enable", "enabled", "pure", "read",
                          "pure read", "pure_read", "safe", "n/a", "none", "no reason", "todo"})


@dataclass(frozen=True)
class PureRead:
    """A caller's contract that an injected implementation is inert — the only claim below the floor.

    Built by :func:`pure_read`; the field is the sentence that makes the claim reviewable. It is a type
    rather than a boolean so that lowering a tool's class cannot happen by accident, by a default, or by a
    value copied out of an example.
    """

    reason: str


def pure_read(reason: str) -> PureRead:
    """``read``, claimed deliberately: ``reason`` has to say why the injected implementation touches nothing."""
    text = (reason or "").strip()
    if not text:
        raise RegistryConfigurationError(
            "pure_read() needs a sentence explaining why the injected implementation is a read — what it "
            "calls, and why that leaves nothing behind. The reason is what a reviewer reads.")
    if text.lower() in _BARE_TRUTHY:
        raise RegistryConfigurationError(
            f"pure_read({text!r}) is not a reason. Name what the function calls and why that is inert, e.g. "
            f'pure_read("formats its arguments into a string; no I/O — reviewed 2026-11-22").')
    return PureRead(reason=text)


def _source_of(fn: object) -> Optional[str]:
    """``fn``'s own source, or ``None`` when it cannot be read (built-in, compiled at run time)."""
    try:
        return textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None


def _receiver_name(node: ast.AST) -> str:
    """The leftmost name of a dotted call target: ``os.path.remove`` -> ``os``, ``rows.remove`` -> ``rows``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def write_sinks(fn: Callable[[dict], object]) -> tuple[str, ...]:
    """The sinks visible in ``fn``'s own source, nested definitions included; ``()`` if none or unreadable."""
    source = _source_of(fn)
    if source is None:
        return ()
    try:
        tree = ast.parse(source)
    except SyntaxError:                                  # getsource returned something unparseable
        return ()
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in SINK_ATTRS or attr in SINK_TX_ATTRS or attr in DESTRUCTIVE_ATTRS:
                found.add(attr)
            elif attr in FS_QUALIFIED_ATTRS and _receiver_name(func.value) in FS_RECEIVERS:
                found.add(f"{_receiver_name(func.value)}.{attr}")
        elif isinstance(func, ast.Name) and func.id in (SINK_NAMES | SINK_ATTRS | SINK_TX_ATTRS):
            found.add(func.id)
    return tuple(sorted(found))


def resolve_action(*, shipped: str, injected: Optional[object] = None, declared: Optional[object] = None,
                   param: str = "", tool: str = "", context: str = "") -> str:
    """The risk class of the function that is actually installed, which is not always the shipped one's.

    ``shipped`` is the class reviewed here for the implementation this file ships; ``injected`` is the
    caller's replacement, or ``None`` when they took the shipped one; ``declared`` is what the caller asked
    for through the tenant's ``*_action`` keyword — a class at or above :data:`INJECTED_FLOOR`, or
    :func:`pure_read` to claim less.

    Nothing injected returns ``shipped`` untouched, so a shipped pure implementation keeps the ``read`` it was
    audited at. Something injected is where the discipline lives:

    * **no declaration** → refused. A class is not inherited from the tool that occupied the slot.
    * **``"read"`` as a plain word** → refused. Below the floor is a claim that has to be made deliberately,
      in the type built for it.
    * **``pure_read(...)``** → refused unless the reason is a sentence, the source can be read, and
      :func:`write_sinks` finds nothing in it. Evidence beats a declaration: nobody registers a below-the-floor
      class over a visible ``post``, and nobody gets one for a function whose source cannot be read.
    * **a destructive sink under ``delete``** → refused, for the same reason.
    * **over-declaring** is always allowed: asking for more authority than a call needs can only ever deny it.
    """
    where = f"{tool} ({context})" if tool and context else (tool or context or "this tool")
    param = param or f"{tool.rsplit('.', 1)[-1] if tool else 'tool'}_action"
    if shipped not in ACTIONS:
        raise RegistryConfigurationError(
            f"{where}: the shipped risk class {shipped!r} is not one of the vocabulary {ACTIONS} — a bug in "
            f"this integration, not a value a caller can supply.")
    if injected is None:
        return shipped

    claim = declared
    if isinstance(claim, PureRead):
        claimed = "read"
    elif isinstance(claim, str):
        if claim == "read":
            raise RegistryConfigurationError(
                f"{where}: {param}='read' is below the floor for a function injected into this slot. "
                f"Something replaced the implementation this file audited, so it is registered as "
                f"'{INJECTED_FLOOR}' unless its author contracts that it is inert: {param}="
                f'pure_read("what it calls, and why that leaves nothing behind").')
        claimed = claim
    elif claim is None:
        raise RegistryConfigurationError(
            f"{where}: a function was injected in place of the implementation shipped here, and a risk class "
            f"is not inherited from the tool it replaces. Pass {param}=... — '{INJECTED_FLOOR}' or 'delete', "
            f"or {param}=pure_read(...) for an implementation you have read and can argue is inert.")
    else:
        raise RegistryConfigurationError(
            f"{where}: {param}={claim!r} is neither a risk class in {ACTIONS} nor a pure_read(...) contract.")

    if claimed not in ACTIONS:
        raise RegistryConfigurationError(
            f"{where}: {param}={claimed!r} is not one of the vocabulary {ACTIONS}.")

    sinks = write_sinks(injected)
    if claimed == "read":
        if _source_of(injected) is None:
            raise RegistryConfigurationError(
                f"{where}: {param}=pure_read(...) cannot be honoured for a function whose source is not "
                f"readable (a built-in or something compiled at run time): there is nothing to check the "
                f"claim against. Register it as '{INJECTED_FLOOR}'.")
        if sinks:
            raise RegistryConfigurationError(
                f"{where}: {param}=pure_read(...) claims this injected function is inert, and its own source "
                f"disagrees — it reaches {list(sinks)}. A call that leaves the process or changes state is "
                f"not a read: register it as '{INJECTED_FLOOR}' (or 'delete'), whatever it is named.")
    destructive = tuple(sorted(s for s in sinks if s in DESTRUCTIVE_ATTRS))
    if destructive and ACTIONS.index(claimed) < ACTIONS.index("delete"):
        raise RegistryConfigurationError(
            f"{where}: {param}={claimed!r} understates what this injected function does — it reaches "
            f"{list(destructive)}, which is the 'delete' class. Over-declare rather than under-declare.")
    return claimed