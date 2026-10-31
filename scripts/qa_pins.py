#!/usr/bin/env python3
"""Check and prepare the move of the QA-related dependency pins from commits to release tags (stdlib only, offline).

Declarations covered
    runtime-contracts   pyproject.toml  [qa] extra and [context-state] extra; .github/workflows/tests.yml (qa job)
    agent-harness       pyproject.toml  base dependency; Dockerfile and deploy/proxmox-demo/Dockerfile.contextruntime
                        (commit or tag, and the sha256 of the source archive)

Commands
    check [--release]   verify that every declaration agrees with scripts/qa_pins.toml and that TEMP-PIN markers are
                        present exactly while a commit pin is temporary. With --release, fail while any pin is still a
                        temporary commit pin or any marker remains.
    rewrite ...         compute the move to release tags. A dry run by default: it prints a diff and changes nothing.
                        Writing needs --write and --confirm-tags-exist, and it only ever moves all pins together.
    verify-tags ...     ask the remote repositories whether the tags exist (network; git ls-remote).

This tool never creates a tag, a release or a commit, and `rewrite` without --write never touches a file. The procedure is
in docs/qa_planning.md.
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "scripts/qa_pins.toml"
PYPROJECT = "pyproject.toml"
WORKFLOW = ".github/workflows/tests.yml"
DOCKERFILES = ("Dockerfile", "deploy/proxmox-demo/Dockerfile.contextruntime")
MARKER = "TEMP-PIN"

_CONTRACTS_URL = r"git\+https://github\.com/redevops-io/runtime-contracts\.git@"
_HARNESS_URL = r"git\+https://github\.com/redevops-io/agent-harness\.git@"
RE_QA = re.compile(rf'^(qa = \["runtime-contracts @ {_CONTRACTS_URL})([^"\s]+)("\])([ \t]*# {MARKER})?[ \t]*$', re.M)
RE_CONTEXT_STATE = re.compile(rf'^(context-state = \["runtime-contracts @ {_CONTRACTS_URL})([^"\s]+)("\])', re.M)
RE_HARNESS_DEP = re.compile(rf'("agent-harness @ {_HARNESS_URL})([^"\s]+)(")')
RE_ARG_REF = re.compile(r"^ARG AGENT_HARNESS_(?:COMMIT|REF)=(\S+)$", re.M)
RE_ARG_SHA = re.compile(r"^ARG AGENT_HARNESS_SHA256=([0-9a-f]{64})$", re.M)
RE_WORKFLOW_PIN = re.compile(r"^([ \t]*RUNTIME_CONTRACTS_PIN: )(\S+)([ \t]*)$", re.M)
RE_WORKFLOW_MARKER = re.compile(rf"^[ \t]*# {MARKER}[ \t]*\n", re.M)
RE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
RE_TAG = re.compile(r"^v\d+\.\d+\.\d+$")
RE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PinError(Exception):
    pass


def _read(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8")
    except OSError as exc:
        raise PinError(f"cannot read {rel}: {exc}") from exc


def load_manifest(root: Path) -> dict:
    try:
        data = tomllib.loads(_read(root, MANIFEST))
        for name in ("runtime-contracts", "agent-harness"):
            sec = data[name]
            for key in ("repository", "temp_pin", "commit", "tag"):
                sec[key]
            if not RE_COMMIT.match(sec["commit"]):
                raise PinError(f"{name}: commit must be 40 lowercase hex digits")
        if not RE_SHA256.match(data["agent-harness"]["archive_sha256"]):
            raise PinError("agent-harness: archive_sha256 must be 64 lowercase hex digits")
        return data
    except (KeyError, tomllib.TOMLDecodeError) as exc:
        raise PinError(f"{MANIFEST} is not valid: {exc!r}") from exc


def _declared(sec: dict) -> str:
    """What the declarations should carry now: the tag once it is set, otherwise the commit."""
    return sec["tag"] or sec["commit"]


def check(root: Path, *, release: bool = False) -> list[str]:
    """Problems found; an empty list means every declaration agrees with the manifest."""
    problems: list[str] = []
    try:
        m = load_manifest(root)
        pyproject, workflow = _read(root, PYPROJECT), _read(root, WORKFLOW)
        dockerfiles = {rel: _read(root, rel) for rel in DOCKERFILES}
    except PinError as exc:
        return [str(exc)]
    contracts, harness = m["runtime-contracts"], m["agent-harness"]
    for name, sec in (("runtime-contracts", contracts), ("agent-harness", harness)):
        if sec["tag"] and not RE_TAG.match(sec["tag"]):
            problems.append(f"{name}: tag {sec['tag']!r} is not of the form vN.N.N")
        if sec["tag"] and sec["temp_pin"]:
            problems.append(f"{name}: a tag is set but temp_pin is still true")
        if not sec["tag"] and not sec["temp_pin"]:
            problems.append(f"{name}: temp_pin is false but no tag is set")

    # runtime-contracts
    want = _declared(contracts)
    qa = RE_QA.search(pyproject)
    if not qa:
        problems.append(f"{PYPROJECT}: the qa extra is not declared in the expected form")
    else:
        if qa.group(2) != want:
            problems.append(f"{PYPROJECT}: qa extra pins {qa.group(2)}, the manifest says {want}")
        if bool(qa.group(4)) != bool(contracts["temp_pin"]):
            problems.append(f"{PYPROJECT}: the {MARKER} marker on the qa extra "
                            f"{'is missing' if contracts['temp_pin'] else 'must be removed'}")
    cs = RE_CONTEXT_STATE.search(pyproject)
    if not cs:
        problems.append(f"{PYPROJECT}: the context-state extra is not declared in the expected form")
    elif contracts["tag"] and cs.group(2) != contracts["tag"]:
        problems.append(f"{PYPROJECT}: context-state extra pins {cs.group(2)}; the qa and context-state extras must "
                        f"resolve to one contracts tag ({contracts['tag']})")
    wf = RE_WORKFLOW_PIN.search(workflow)
    if not wf:
        problems.append(f"{WORKFLOW}: RUNTIME_CONTRACTS_PIN is not set")
    else:
        if wf.group(2) != want:
            problems.append(f"{WORKFLOW}: RUNTIME_CONTRACTS_PIN is {wf.group(2)}, the manifest says {want}")
        if bool(RE_WORKFLOW_MARKER.search(workflow)) != bool(contracts["temp_pin"]):
            problems.append(f"{WORKFLOW}: the {MARKER} marker "
                            f"{'is missing' if contracts['temp_pin'] else 'must be removed'}")

    # agent-harness
    want_h = _declared(harness)
    dep = RE_HARNESS_DEP.search(pyproject)
    if not dep:
        problems.append(f"{PYPROJECT}: the agent-harness dependency is not declared in the expected form")
    elif dep.group(2) != want_h:
        problems.append(f"{PYPROJECT}: agent-harness pins {dep.group(2)}, the manifest says {want_h}")
    for rel, text in dockerfiles.items():
        ref, sha = RE_ARG_REF.search(text), RE_ARG_SHA.search(text)
        if not ref or not sha:
            problems.append(f"{rel}: the agent-harness ARG lines are not in the expected form")
            continue
        if ref.group(1) != want_h:
            problems.append(f"{rel}: agent-harness ref is {ref.group(1)}, the manifest says {want_h}")
        if sha.group(1) != harness["archive_sha256"]:
            problems.append(f"{rel}: agent-harness archive sha256 differs from the manifest")

    if release:
        for name, sec in (("runtime-contracts", contracts), ("agent-harness", harness)):
            if sec["temp_pin"] or not sec["tag"]:
                problems.append(f"{name}: still a temporary commit pin; a release needs a tag")
        for rel, text in ((PYPROJECT, pyproject), (WORKFLOW, workflow)):
            if MARKER in text:
                problems.append(f"{rel}: a {MARKER} marker remains")
    return problems


def _sub_once(pattern: re.Pattern, repl, text: str, where: str) -> str:
    new, n = pattern.subn(repl, text)
    if n != 1:
        raise PinError(f"{where}: expected exactly one match for {pattern.pattern[:60]!r}, found {n}")
    return new


def compute_rewrite(root: Path, *, contracts_tag: str, harness_tag: str, harness_sha256: str) -> dict[str, tuple[str, str]]:
    """The new content of every file that changes, as ``{relative path: (old, new)}``. Reads only."""
    if not RE_TAG.match(contracts_tag or ""):
        raise PinError(f"--contracts-tag {contracts_tag!r} must look like vN.N.N")
    if not RE_TAG.match(harness_tag or ""):
        raise PinError(f"--harness-tag {harness_tag!r} must look like vN.N.N")
    if not RE_SHA256.match(harness_sha256 or ""):
        raise PinError("--harness-sha256 must be the 64-hex sha256 of the harness tag's source archive")
    load_manifest(root)
    out: dict[str, tuple[str, str]] = {}

    old = new = _read(root, PYPROJECT)
    new = _sub_once(RE_QA, lambda m: f"{m.group(1)}{contracts_tag}{m.group(3)}", new, PYPROJECT + " [qa]")
    new = _sub_once(RE_CONTEXT_STATE, lambda m: f"{m.group(1)}{contracts_tag}{m.group(3)}", new,
                    PYPROJECT + " [context-state]")
    new = _sub_once(RE_HARNESS_DEP, lambda m: f"{m.group(1)}{harness_tag}{m.group(3)}", new, PYPROJECT + " agent-harness")
    out[PYPROJECT] = (old, new)

    old = new = _read(root, WORKFLOW)
    new = _sub_once(RE_WORKFLOW_PIN, lambda m: f"{m.group(1)}{contracts_tag}{m.group(3)}", new, WORKFLOW)
    new = RE_WORKFLOW_MARKER.sub("", new)
    out[WORKFLOW] = (old, new)

    for rel in DOCKERFILES:
        old = new = _read(root, rel)
        new = _sub_once(RE_ARG_REF, f"ARG AGENT_HARNESS_REF={harness_tag}", new, rel)
        new = new.replace("${AGENT_HARNESS_COMMIT}", "${AGENT_HARNESS_REF}")
        new = new.replace("/archive/${AGENT_HARNESS_REF}.tar.gz", "/archive/refs/tags/${AGENT_HARNESS_REF}.tar.gz")
        new = _sub_once(RE_ARG_SHA, f"ARG AGENT_HARNESS_SHA256={harness_sha256}", new, rel)
        out[rel] = (old, new)

    old = _read(root, MANIFEST)
    new = _manifest_after(old, contracts_tag, harness_tag, harness_sha256)
    out[MANIFEST] = (old, new)
    return out


def _manifest_after(text: str, contracts_tag: str, harness_tag: str, harness_sha256: str) -> str:
    parts = re.split(r"(?m)^(?=\[)", text)
    out = []
    for part in parts:
        if part.startswith("[runtime-contracts]"):
            part = _sub_once(re.compile(r'(?m)^tag = ""$'), f'tag = "{contracts_tag}"', part, MANIFEST)
            part = _sub_once(re.compile(r"(?m)^temp_pin = true$"), "temp_pin = false", part, MANIFEST)
        elif part.startswith("[agent-harness]"):
            part = _sub_once(re.compile(r'(?m)^tag = ""$'), f'tag = "{harness_tag}"', part, MANIFEST)
            part = _sub_once(re.compile(r"(?m)^temp_pin = true$"), "temp_pin = false", part, MANIFEST)
            part = _sub_once(re.compile(r'(?m)^archive_sha256 = "[0-9a-f]{64}"$'),
                             f'archive_sha256 = "{harness_sha256}"', part, MANIFEST)
        out.append(part)
    return "".join(out)


FOLLOW_UPS = (
    "update tests/test_container.py: the Dockerfile ARG is now AGENT_HARNESS_REF (a tag) and pyproject pins the same tag",
    "update the pin comments in pyproject.toml, Dockerfile and deploy/proxmox-demo/Dockerfile.contextruntime",
    "run: python scripts/qa_pins.py check --release",
    "build the image locally and record the result (the unit-test job does not run Docker)",
    "write the release notes and bump the version in pyproject.toml and context_runtime/__init__.py (separate change)",
)


def verify_tags(contracts_repo: str, contracts_tag: str, harness_repo: str, harness_tag: str, timeout: int = 60) -> list[str]:
    """Ask the remotes whether each tag exists. Needs the network; reads only."""
    problems = []
    for repo, tag in ((contracts_repo, contracts_tag), (harness_repo, harness_tag)):
        try:
            out = subprocess.run(["git", "ls-remote", "--tags", repo, f"refs/tags/{tag}"], capture_output=True,
                                 text=True, timeout=timeout, env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin"})
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"{repo}: could not ask the remote ({type(exc).__name__})")
            continue
        if out.returncode != 0 or f"refs/tags/{tag}" not in out.stdout:
            problems.append(f"{repo}: tag {tag} was not found")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this repository)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--release", action="store_true")
    r = sub.add_parser("rewrite")
    r.add_argument("--contracts-tag", required=True)
    r.add_argument("--harness-tag", required=True)
    r.add_argument("--harness-sha256", required=True)
    r.add_argument("--write", action="store_true")
    r.add_argument("--confirm-tags-exist", action="store_true")
    v = sub.add_parser("verify-tags")
    v.add_argument("--contracts-tag", required=True)
    v.add_argument("--harness-tag", required=True)
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    try:
        if args.cmd == "check":
            problems = check(root, release=args.release)
            for p in problems:
                print(f"qa-pins: {p}", file=sys.stderr)
            print(f"qa-pins: {'FAILED' if problems else 'ok'} ({len(problems)} problem(s))")
            return 1 if problems else 0
        if args.cmd == "verify-tags":
            m = load_manifest(root)
            problems = verify_tags(m["runtime-contracts"]["repository"], args.contracts_tag,
                                   m["agent-harness"]["repository"], args.harness_tag)
            for p in problems:
                print(f"qa-pins: {p}", file=sys.stderr)
            return 1 if problems else 0
        changes = compute_rewrite(root, contracts_tag=args.contracts_tag, harness_tag=args.harness_tag,
                                  harness_sha256=args.harness_sha256)
        for rel, (old, new) in changes.items():
            sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
        if not args.write:
            print("qa-pins: dry run; no file was changed")
            return 0
        if not args.confirm_tags_exist:
            raise PinError("--write needs --confirm-tags-exist: say that both tags exist (verify-tags checks the remotes)")
        for rel, (_, new) in changes.items():
            (root / rel).write_text(new, encoding="utf-8")
        print("qa-pins: files rewritten. Follow-ups:")
        for item in FOLLOW_UPS:
            print(f"  - {item}")
        return 0
    except PinError as exc:
        print(f"qa-pins: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
