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

**Delegation is refused by default; it is never classified by default.** Handing a tenant a callable from
outside this repository hands it a capability the platform has not read, so the answer to "what class is
that?" is *nobody knows, so it does not run* — not a class anyone assumes. There is deliberately no floor
here: a constant like ``write`` looks like a conservative default and behaves like a grant, because the code
swapping itself in picks it, in passing, while writing an adapter. Every class a delegated callable is
registered with has to name where it came from, and :func:`delegated` is the only way to say so: the class,
the authority that authorized this delegation, what the callable hands its arguments to, and the review that
connects the two. The refusal messages carry that sentence rather than a class.

**The one claim that is verified instead of authorized.** :func:`pure_read` asks for *less* authority than a
delegation would, so it is not bought with provenance — it is proved, against the callable's own source:
:func:`write_sinks` searches the sink vocabulary the platform's read-classification sweep (agentic-os FR-04)
uses, and :func:`opaque_callees` refuses a claim the source cannot be read far enough to check. That is also
why :func:`delegated` will not carry a ``read``: three sentences from the code asking to run are not evidence
that the code is inert, and text this repository cannot check must never buy a weaker grant than the one the
reviewed implementation held.

**What this does not pretend to be.** The sweep reads the callable that was handed over — nested definitions
included. What it cannot see it refuses, and what it can see it only ever uses to *raise* a class or refuse a
claim: over-declaring stays legal, because registering a read as a ``write`` costs a call and registering a
write as a read costs a policy boundary.
"""
from __future__ import annotations

import ast
import inspect
import builtins
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional

from .base import ACTIONS, RegistryConfigurationError

#: The least class a delegated implementation may be *authorized* at. This is not a default and not a floor:
#: nothing is registered at this class because a callable was handed over, and a bare string naming it is
#: refused (see :func:`resolve_action`). It is the bottom of the range :func:`delegated` accepts, so the only
#: thing it decides is that an authorized delegation asks for at least ``write``.
MIN_DELEGATED_CLASS = "write"

#: Roots a ``pure_read`` claim may be checked against. The list is an allow-list on purpose: an effect this
#: module cannot read is an effect it refuses to grade, so a name missing from here costs the caller a
#: :func:`delegated` contract and a wrong entry in here costs a policy boundary. Parameters, locals and names
#: defined inside the callable never need to appear — they are visible in the source being checked.
INERT_ROOTS = frozenset({
    "json", "re", "math", "string", "textwrap", "datetime", "collections", "itertools", "typing",
    "dataclasses", "hashlib", "base64", "urllib_parse", "ToolResult", "PureRead",
})

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

#: How many characters a sentence about a delegation has to clear to be treated as one. Below this, what
#: someone typed is a label ("SEC-41", "checked", "vendor") and a later reader has nothing to check.
_MIN_SENTENCE = 24

#: Names that need no provenance because calling them cannot reach outside the process. Derived, not typed
#: out, so the list cannot drift from the interpreter's own answer.
_BUILTINS = frozenset(dir(builtins))


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


@dataclass(frozen=True)
class Delegated:
    """An authorized delegation: code from outside this repository will run, and someone named who said so.

    Four fields because four questions go unanswered when a class is guessed — what capability is being
    asked for (:attr:`action`), who authorized a callable this repository did not write to be runnable at all
    (:attr:`authority`), what the callable actually hands its arguments to (:attr:`via`), and which review
    connects the two (:attr:`review`). A record that answers them is greppable in a manifest and reviewable in
    a pull request; a bare class string answers none of them, which is why it is refused.
    """

    action: str
    authority: str
    via: str
    review: str


def _sentence(value: object, field: str, *, param: str, example: str) -> str:
    """``value`` as a checked sentence. Empty, bare-truthy and one-word answers are refusals, not typos."""
    text = (value or "").strip() if isinstance(value, str) else ""
    if not text or text.lower() in _BARE_TRUTHY:
        raise RegistryConfigurationError(
            f"{param}: {field}={value!r} does not say anything. {field} is the part a reviewer reads — "
            f"e.g. {example}.")
    if len(text) < _MIN_SENTENCE or " " not in text.strip():
        raise RegistryConfigurationError(
            f"{param}: {field}={text!r} is a label, not a sentence. Give {field} enough words to be checked "
            f"later — e.g. {example}.")
    return text


def delegated(*, action: str, authority: str, via: str, review: str) -> Delegated:
    """Authorize a delegated implementation at ``action`` — the only way one gets registered at all.

    ``authority`` names who authorized a callable from outside this repository to run at that class (a grant,
    a policy-plane rule, a signed-off ticket — not the caller's own intent). ``via`` names what the callable
    hands its arguments to, in the terms a reader can go and check. ``review`` is the sentence that says
    somebody read the code and when. All three are required, in sentences, because the delegation the platform
    is afraid of is the one that was wired up by someone who meant well and wrote down nothing.

    ``read`` is not accepted here on purpose: a class this weak has to be proved against the source with
    :func:`pure_read`, never asserted in prose by the code asking to run.
    """
    if action not in ACTIONS:
        raise RegistryConfigurationError(
            f"delegated(action={action!r}) is not one of the vocabulary {ACTIONS}.")
    if ACTIONS.index(action) < ACTIONS.index(MIN_DELEGATED_CLASS):
        raise RegistryConfigurationError(
            f"delegated(action={action!r}) is refused: a delegation cannot be authorized into the '{action}' "
            f"class, because prose from the code asking to run is not evidence that it is inert. "
            f"{action.capitalize()} is the only class below '{MIN_DELEGATED_CLASS}' and it is proved against "
            f"the callable's source with pure_read(...) — which refuses the claim the moment it cannot see "
            f"enough of the code to check it.")
    return Delegated(
        action=action,
        authority=_sentence(authority, "authority", param="delegated()",
                            example='authority="grant agentic-os:fleet/edge-sentinel block_ip (ticket SEC-41)"'),
        via=_sentence(via, "via", param="delegated()",
                      example='via="sentinel_client.Client.block_ip, an HTTPS PUT to the Sentinel API"'),
        review=_sentence(review, "review", param="delegated()",
                         example='review="read the client on 2026-11-22; it writes to one vendor API and nothing else"'),
    )


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


def opaque_callees(fn: Callable[[dict], object]) -> tuple[str, ...]:
    """What ``fn`` calls that this module cannot see inside — the delegation a sink sweep is blind to.

    A call is visible when its root is a parameter or a local of ``fn``, a name bound inside its own body, a
    builtin, or an entry of :data:`INERT_ROOTS`. Anything else — a client built at module scope, an object
    closed over by a factory, a method on an injected session — is where the effect of the call goes, and this
    file has no way to know what it is. An unreadable callable is reported as one opaque callee rather than as
    zero, because "could not look" and "found nothing" have to stay different answers.

    Used to refuse a ``pure_read`` claim, never to grant one: an opaque callee is exactly what
    :func:`delegated` exists to authorize.
    """
    source = _source_of(fn)
    if source is None:
        return ("<unreadable source>",)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ("<unparseable source>",)
    code = getattr(fn, "__code__", None) or getattr(getattr(fn, "__func__", None), "__code__", None)
    bound = set(getattr(code, "co_varnames", ()))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                bound.update(a.arg for a in (list(getattr(args, "posonlyargs", [])) + list(args.args)
                                             + list(args.kwonlyargs)))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    invisible = _BUILTINS | INERT_ROOTS | bound
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        root = (_receiver_name(node.func) if isinstance(node.func, ast.Attribute)
                else node.func.id if isinstance(node.func, ast.Name) else "")
        if root and root not in invisible:
            found.add(root)
    return tuple(sorted(found))


def resolve_action(*, shipped: str, injected: Optional[object] = None, declared: Optional[object] = None,
                   param: str = "", tool: str = "", context: str = "") -> str:
    """The risk class of the function that is actually installed, which is not always the shipped one's.

    ``shipped`` is the class reviewed here for the implementation this file ships; ``injected`` is the
    caller's replacement, or ``None`` when they took the shipped one; ``declared`` is what the caller passed
    through the tenant's ``*_action`` keyword — :func:`delegated` to authorize the delegation, or
    :func:`pure_read` to prove the callable inert instead.

    Nothing injected returns ``shipped`` untouched, so a shipped pure implementation keeps the ``read`` it was
    audited at. Something injected is a delegation, and delegation is refused by default:

    * **nothing passed** → refused. A class is not inherited from the tool that occupied the slot, and it is
      not defaulted either — the default is the thing that gets copied into the next deployment.
    * **a class as a plain word** (``"write"``, ``"delete"``, ``"read"``) → refused. A word is not an
      authorization; naming a class is the answer, and the question was who gave it.
    * **``delegated(...)``** → accepted for ``write``/``delete`` when the authority, the callee and the review
      are all sentences, and never for less than the class the wrapper's own source implies.
    * **``pure_read(...)``** → accepted only when the source can be read, :func:`write_sinks` finds nothing in
      it, and :func:`opaque_callees` finds nothing it hands its arguments to. A claim this repository can
      check is worth more than a claim it cannot.

    Over-declaring stays legal throughout: a call registered above its true class is denied, and a call
    registered below it is a policy boundary that has already been crossed.
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
    # From here the code that would run was written outside this repository. Nothing is assumed about it —
    # least of all its risk class, which is the one thing a caller must not get to supply silently.
    if claim is None:
        raise RegistryConfigurationError(
            f"{where}: a function was injected in place of the implementation shipped here, and nothing says "
            f"who authorized it. A risk class is not inherited from the tool it replaces, and there is no "
            f"default one either — the class a caller picks in passing while writing an adapter is a grant "
            f"that nobody reviewed. Pass {param}=delegated(action=..., authority=..., via=..., review=...), "
            f"or {param}=pure_read(...) for an implementation whose own source proves it inert.")
    if isinstance(claim, PureRead):
        if _source_of(injected) is None:
            raise RegistryConfigurationError(
                f"{where}: {param}=pure_read(...) cannot be honoured for a function whose source is not "
                f"readable (a built-in or something compiled at run time): there is nothing to check the "
                f"claim against. Authorize it with {param}=delegated(...) instead.")
        sinks = write_sinks(injected)
        if sinks:
            raise RegistryConfigurationError(
                f"{where}: {param}=pure_read(...) claims this injected function is inert, and its own source "
                f"disagrees — it reaches {list(sinks)}. A call that leaves the process or changes state is "
                f"not a read, whatever it is named: authorize it with "
                f"{param}=delegated(action='{MIN_DELEGATED_CLASS}', ...).")
        hidden = opaque_callees(injected)
        if hidden:
            raise RegistryConfigurationError(
                f"{where}: {param}=pure_read(...) cannot be honoured, because the claim cannot be checked: "
                f"the function hands its arguments to {list(hidden)}, which is not defined in the code this "
                f"check can read. That is delegation, and delegation is refused by default — either what it "
                f"calls is provably inert (add it to INERT_ROOTS with the review that says so) or the "
                f"delegation needs {param}=delegated(action=..., authority=..., via=..., review=...).")
        return "read"
    if isinstance(claim, Delegated):
        if claim.action not in ACTIONS:
            raise RegistryConfigurationError(
                f"{where}: {param} carries a delegation for action={claim.action!r}, which is not one of the "
                f"vocabulary {ACTIONS}.")
        if ACTIONS.index(claim.action) < ACTIONS.index(MIN_DELEGATED_CLASS):
            raise RegistryConfigurationError(
                f"{where}: {param} delegates at {claim.action!r}. A delegation is never authorized below "
                f"'{MIN_DELEGATED_CLASS}': prose buys a strong class, and only source evidence buys a weak "
                f"one — use {param}=pure_read(...) if this call really is a read.")
        # Built by hand instead of by delegated(): the sentences are the authorization, so they are checked
        # here too rather than trusted because the object is the right shape.
        _sentence(claim.authority, "authority", param=param, example='authority="grant SEC-41 ..."')
        _sentence(claim.via, "via", param=param, example='via="sentinel_client.Client.block_ip"')
        _sentence(claim.review, "review", param=param, example='review="read the client on 2026-11-22 ..."')
        destructive = tuple(sorted(s for s in write_sinks(injected) if s in DESTRUCTIVE_ATTRS))
        if destructive and ACTIONS.index(claim.action) < ACTIONS.index("delete"):
            raise RegistryConfigurationError(
                f"{where}: {param} delegates at {claim.action!r}, but the wrapper's own source reaches "
                f"{list(destructive)}, which is the 'delete' class. A delegation may over-ask and never "
                f"under-state: raise it to 'delete' or stop reaching for the filesystem.")
        return claim.action
    if isinstance(claim, str):
        raise RegistryConfigurationError(
            f"{where}: {param}={claim!r} names a risk class and no authority. Delegated code is refused by "
            f"default, so the class has to come from someone who can be named — a conservative-sounding word "
            f"typed next to the factory is the same unnamed grant as a permissive one. Use "
            f"{param}=delegated(action={claim!r}, authority=..., via=..., review=...) — or "
            f"{param}=pure_read(...) for a call that is a read and can show it.")
    raise RegistryConfigurationError(
        f"{where}: {param}={claim!r} is neither a delegated(...) authorization nor a pure_read(...) claim.")