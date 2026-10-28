"""Regression guard: future changes must not restore unrestricted environment inheritance for MCP subprocesses.

tests/test_mcp_env.py checks the behaviour of the allowlist. This file guards against the ways it tends to be undone later:
widening the allowlist, defaulting ``inherit_env`` to on, a launch path that bypasses ``stdio_env``, a new subprocess spawn
site, or ``os.environ`` creeping back into the launcher. Only synthetic secrets are used.
"""
from __future__ import annotations

import ast
import inspect
import json
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx", reason="needs the optional dependency 'httpx' (pip install -e '.[test]')")

import context_runtime
from context_runtime.tools import mcp as mcp_module
from context_runtime.tools.mcp import INHERIT_ENV_VAR, MCPClient, SAFE_ENV_PREFIXES, SAFE_ENV_VARS, stdio_env

PACKAGE = Path(context_runtime.__file__).resolve().parent
MCP_FILE = Path(mcp_module.__file__).resolve()

#: The exact allowlist as reviewed. Adding a name here is a security decision: it needs a reason in the PR.
REVIEWED_SAFE_ENV_VARS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TZ", "TERM", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
    "PYTHONPATH", "VIRTUAL_ENV", "PYTHONUTF8", "PYTHONIOENCODING",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NO_PROXY",
})
REVIEWED_PREFIXES = ("LC_",)

#: Every place the package starts a subprocess, by file and call name. Anything new must be classified: a launcher of MCP
#: servers must go through MCPClient.stdio (which applies stdio_env); anything else is added here with a justification.
KNOWN_SPAWN_SITES = {
    "tools/mcp.py": {"Popen"},                    # the MCP stdio launcher; env comes from stdio_env
    "ingest/multimodal.py": {"run"},              # media tooling, not an MCP server
    "adapters/store_video.py": {"run"},           # ffmpeg frame extraction, not an MCP server
    "integrations/supply_chain.py": {"run"},      # scanner CLI; inherits the parent environment by design (not MCP)
    "integrations/runtime_scan_engine.py": {"run"},  # scanner CLI; inherits the parent environment by design (not MCP)
}
SPAWN_NAMES = {"Popen", "run", "check_output", "check_call", "call", "create_subprocess_exec", "create_subprocess_shell",
               "stdio_client", "StdioServerParameters", "spawn", "spawnv", "spawnve", "execv", "execve", "system", "popen"}

SECRET_LIKE = [
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AZURE_CLIENT_SECRET", "GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "XAI_API_KEY", "KAGGLE_API_TOKEN", "HF_TOKEN", "DATABASE_URL", "POSTGRES_PASSWORD", "REDIS_URL",
    "SLACK_BOT_TOKEN", "STRIPE_SECRET_KEY", "SENTRY_DSN", "CLOUDFLARE_API_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS",
    "NPM_TOKEN", "PYPI_TOKEN", "APPROVAL_SECRET", "SSH_AUTH_SOCK", "DOCKER_CONFIG", "KUBECONFIG", "CR_TEST_API_KEY",
]
SYNTHETIC = "synthetic-value-not-a-secret"

ENV_SERVER = r'''import json, os, sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    rid, meth = req.get("id"), req.get("method")
    if rid is None:
        continue
    if meth == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "env", "version": "0"}}
    elif meth == "tools/list":
        result = {"tools": [{"name": "env", "description": "report the environment", "inputSchema": {"type": "object"}}]}
    else:
        result = {"content": [{"type": "text", "text": json.dumps(dict(os.environ))}]}
    print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}), flush=True)
'''


@pytest.fixture(autouse=True)
def _escape_hatch_off(monkeypatch):
    """The process-wide opt-out must never be on by accident in these tests."""
    monkeypatch.delenv(INHERIT_ENV_VAR, raising=False)


# ---- the defaults and the allowlist are pinned ---------------------------------------------------------------------------

def test_inheritance_is_off_by_default_in_every_signature():
    assert inspect.signature(stdio_env).parameters["inherit_env"].default is False
    assert inspect.signature(MCPClient.stdio).parameters["inherit_env"].default is None  # None means: ask the process-wide variable
    assert INHERIT_ENV_VAR == "CONTEXT_RUNTIME_MCP_INHERIT_ENV"


def test_allowlist_is_exactly_the_reviewed_set():
    assert SAFE_ENV_VARS == REVIEWED_SAFE_ENV_VARS, (
        "the stdio MCP environment allowlist changed: widening it is a security decision, so update REVIEWED_SAFE_ENV_VARS "
        f"in this test together with the reason. Added: {sorted(SAFE_ENV_VARS - REVIEWED_SAFE_ENV_VARS)}; "
        f"removed: {sorted(REVIEWED_SAFE_ENV_VARS - SAFE_ENV_VARS)}")
    assert tuple(SAFE_ENV_PREFIXES) == REVIEWED_PREFIXES


# ---- behaviour over many environments --------------------------------------------------------------------------------------

def _random_parent(rng: random.Random) -> dict[str, str]:
    prefixes = ["AWS_", "AZURE_", "GITHUB_", "OPENAI_", "ANTHROPIC_", "XAI_", "KAGGLE_", "GOOGLE_", "DATABASE_", "POSTGRES_", "CR_", ""]
    suffixes = ["KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIALS", "DSN", "URL", "CONNECTION_STRING", "CERT"]
    parent = {"PATH": "/usr/bin", "HOME": "/home/x", "LC_ALL": "C", "TZ": "UTC"}
    for _ in range(rng.randint(5, 40)):
        parent[rng.choice(prefixes) + rng.choice(suffixes) + str(rng.randint(0, 9))] = SYNTHETIC
    for name in rng.sample(SECRET_LIKE, rng.randint(1, len(SECRET_LIKE))):
        parent[name] = SYNTHETIC
    return parent


def test_default_child_env_is_a_subset_of_the_allowlist_for_many_parents():
    rng = random.Random(20261027)  # fixed seed, so a failure reproduces
    for _ in range(300):
        parent = _random_parent(rng)
        out = stdio_env(parent)
        for name in out:
            upper = name.upper()
            assert upper in REVIEWED_SAFE_ENV_VARS or upper.startswith(REVIEWED_PREFIXES), f"{name} reached the child by default"
        assert SYNTHETIC not in out.values()


def test_only_the_explicit_mapping_adds_beyond_the_allowlist():
    rng = random.Random(7)
    for _ in range(100):
        parent = _random_parent(rng)
        explicit = {"MY_SERVER_TOKEN": "explicit-value", "AWS_SECRET_ACCESS_KEY": "explicit-aws"}
        out = stdio_env(parent, explicit=explicit)
        beyond = {k for k in out if k.upper() not in REVIEWED_SAFE_ENV_VARS and not k.upper().startswith(REVIEWED_PREFIXES)}
        assert beyond == set(explicit)
        assert out["AWS_SECRET_ACCESS_KEY"] == "explicit-aws"  # explicit wins, never the parent's value


def test_opting_in_to_inheritance_still_works_and_is_the_only_way():
    parent = {"PATH": "/usr/bin", "GITHUB_TOKEN": SYNTHETIC}
    assert stdio_env(parent) == {"PATH": "/usr/bin"}
    assert stdio_env(parent, inherit_env=True) == parent


def test_spawned_child_gets_none_of_many_secret_like_variables(tmp_path, monkeypatch):
    server = tmp_path / "env_server.py"
    server.write_text(ENV_SERVER)
    for name in SECRET_LIKE:
        monkeypatch.setenv(name, SYNTHETIC)
    client = MCPClient.stdio([sys.executable, str(server)])
    try:
        client.initialize()
        seen = json.loads(client.call_tool("env")["content"][0]["text"])
    finally:
        client.close()
    leaked = sorted(name for name in SECRET_LIKE if name in seen)
    assert not leaked, f"reached the stdio MCP child by default: {leaked}"
    assert SYNTHETIC not in seen.values()


# ---- the launch path cannot bypass stdio_env ----------------------------------------------------------------------------------

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _enclosing_function(tree: ast.Module) -> dict[int, str]:
    owner: dict[int, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                owner[id(node)] = fn.name
    return owner


def _is_os_environ(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id == "os"


def test_popen_in_the_mcp_launcher_always_passes_an_env_built_by_stdio_env():
    tree = _parse(MCP_FILE)
    popens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "Popen"]
    assert popens, "the MCP launcher no longer calls Popen: update this guard"
    for call in popens:
        env = next((kw.value for kw in call.keywords if kw.arg == "env"), None)
        assert env is not None, "Popen without env= inherits the whole parent environment"
        assert not _is_os_environ(env) and not (isinstance(env, ast.Constant) and env.value is None)
    # the variable passed as env= must be assigned from stdio_env(...) in the same module
    assigned = {t.id for a in ast.walk(tree) if isinstance(a, ast.Assign) and isinstance(a.value, ast.Call)
                and getattr(a.value.func, "id", "") == "stdio_env" for t in a.targets if isinstance(t, ast.Name)}
    for call in popens:
        env = next(kw.value for kw in call.keywords if kw.arg == "env")
        assert isinstance(env, ast.Name) and env.id in assigned, "Popen env= must come from stdio_env(...)"


def test_os_environ_is_only_used_where_it_is_reviewed_in_the_mcp_module():
    tree = _parse(MCP_FILE)
    owner = _enclosing_function(tree)
    used_in = sorted({owner.get(id(n), "<module>") for n in ast.walk(tree) if _is_os_environ(n)})
    assert used_in == ["stdio", "stdio_env"], used_in  # the default parent and the process-wide opt-out lookup, nothing else
    for n in ast.walk(tree):  # no wholesale copies of the environment anywhere in the module
        if isinstance(n, ast.Call) and n.args and _is_os_environ(n.args[0]) and getattr(n.func, "id", "") == "dict":
            raise AssertionError("dict(os.environ) in the MCP module")
        if isinstance(n, ast.Call) and _is_os_environ(getattr(n.func, "value", None)) and n.func.attr == "copy":
            raise AssertionError("os.environ.copy() in the MCP module")
        if isinstance(n, ast.Dict) and any(k is None and _is_os_environ(v) for k, v in zip(n.keys, n.values)):
            raise AssertionError("**os.environ in the MCP module")


# ---- a new subprocess spawn site has to be classified -------------------------------------------------------------------------

def _spawn_sites() -> dict[str, set[str]]:
    sites: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                base = getattr(getattr(node.func, "value", None), "id", "")
                if name in SPAWN_NAMES and (base in {"subprocess", "asyncio", "os", ""} or name in {"stdio_client", "StdioServerParameters"}):
                    if name in {"run", "call", "system", "popen", "spawn"} and base not in {"subprocess", "os"}:
                        continue  # a method called run() on something else
                    sites.setdefault(path.relative_to(PACKAGE).as_posix(), set()).add(name)
    return sites


def test_every_subprocess_spawn_site_is_known():
    found = _spawn_sites()
    assert found == KNOWN_SPAWN_SITES, (
        "the set of places that start subprocesses changed. If the new site starts an MCP server, route it through "
        "MCPClient.stdio (which applies stdio_env); otherwise add it to KNOWN_SPAWN_SITES with a justification. "
        f"New or changed: {sorted(set(found) ^ set(KNOWN_SPAWN_SITES))} / {[(k, found.get(k)) for k in found if found.get(k) != KNOWN_SPAWN_SITES.get(k)]}")
