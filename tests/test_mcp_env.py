"""stdio MCP servers get an allowlisted environment, not the whole parent environment.

A tiny stdio server reports the environment it was started with. Only synthetic secrets are used.
"""
from __future__ import annotations

import json
import logging
import sys

import pytest

pytest.importorskip("httpx", reason="needs the optional dependency 'httpx' (pip install -e '.[test]')")

from context_runtime.tools.mcp import INHERIT_ENV_VAR, MCPClient, SAFE_ENV_VARS, stdio_env

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

SECRET = "synthetic-secret-value-not-real"


@pytest.fixture()
def server_path(tmp_path):
    path = tmp_path / "env_server.py"
    path.write_text(ENV_SERVER)
    return str(path)


def child_env(server_path, **kwargs) -> dict[str, str]:
    client = MCPClient.stdio([sys.executable, server_path], **kwargs)
    try:
        client.initialize()
        result = client.call_tool("env")
        return json.loads(result["content"][0]["text"])
    finally:
        client.close()


@pytest.fixture()
def parent_env(monkeypatch):
    monkeypatch.delenv(INHERIT_ENV_VAR, raising=False)
    monkeypatch.setenv("CR_TEST_API_KEY", SECRET)
    monkeypatch.setenv("CR_TEST_PASSWORD", SECRET)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("TZ", "UTC")


def test_default_child_does_not_receive_parent_secrets(server_path, parent_env):
    env = child_env(server_path)
    assert SECRET not in json.dumps(env)
    for name in ("CR_TEST_API_KEY", "CR_TEST_PASSWORD", "AWS_SECRET_ACCESS_KEY"):
        assert name not in env
    assert "PATH" in env and env.get("LC_ALL") == "C" and env.get("TZ") == "UTC"


def test_explicit_env_mapping_reaches_the_child(server_path, parent_env):
    env = child_env(server_path, env={"CR_TEST_TOKEN": "explicit-value", "CR_TEST_NUMBER": 7})
    assert env["CR_TEST_TOKEN"] == "explicit-value" and env["CR_TEST_NUMBER"] == "7"
    assert "CR_TEST_API_KEY" not in env


def test_explicit_mapping_wins_over_an_allowlisted_parent_value(server_path, parent_env):
    assert child_env(server_path, env={"TZ": "Asia/Tokyo"})["TZ"] == "Asia/Tokyo"


def test_inherit_env_restores_the_old_behaviour_and_warns(server_path, parent_env, caplog):
    with caplog.at_level(logging.WARNING, logger="context_runtime.tools.mcp"):
        env = child_env(server_path, inherit_env=True)
    assert env["CR_TEST_API_KEY"] == SECRET
    assert any("full parent environment" in r.getMessage() for r in caplog.records)


def test_process_wide_escape_hatch_and_per_call_override(server_path, parent_env, monkeypatch):
    monkeypatch.setenv(INHERIT_ENV_VAR, "1")
    assert child_env(server_path)["CR_TEST_API_KEY"] == SECRET            # old behaviour through the variable
    assert "CR_TEST_API_KEY" not in child_env(server_path, inherit_env=False)  # an explicit False wins


def test_no_warning_by_default(server_path, parent_env, caplog):
    with caplog.at_level(logging.WARNING, logger="context_runtime.tools.mcp"):
        child_env(server_path)
    assert not [r for r in caplog.records if "full parent environment" in r.getMessage()]


def test_proxy_variables_pass_only_without_embedded_credentials():
    parent = {"HTTP_PROXY": "http://proxy.local:3128", "https_proxy": "http://user:pw@proxy.local:3128",
              "NO_PROXY": "localhost", "OPENAI_API_KEY": SECRET, "PATH": "/bin"}
    env = stdio_env(parent)
    assert env == {"HTTP_PROXY": "http://proxy.local:3128", "NO_PROXY": "localhost", "PATH": "/bin"}


def test_stdio_env_is_case_insensitive_and_keeps_lc_prefix():
    env = stdio_env({"path": "/bin", "Lc_Time": "C", "MY_TOKEN": SECRET, "PYTHONPATH": "/src"})
    assert env == {"path": "/bin", "Lc_Time": "C", "PYTHONPATH": "/src"}


def test_allowlist_names_no_credentials():
    assert not [n for n in SAFE_ENV_VARS if any(w in n for w in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))]


def test_shipped_web_search_server_starts_under_the_allowlist(parent_env):
    client = MCPClient.stdio([sys.executable, "-m", "context_runtime.tools.mcp_servers.web_search"])
    try:
        assert "web_search" in [t["name"] for t in client.list_tools()]
    finally:
        client.close()


def test_deploy_scan_settings_are_passed_explicitly_and_secrets_are_not(monkeypatch):
    import context_runtime.integrations.runtime_scan as rs

    seen = {}

    class FakeClient:
        @classmethod
        def stdio(cls, command, env=None, **kw):
            seen["env"] = env
            return object()

    monkeypatch.setattr("context_runtime.tools.mcp.MCPClient", FakeClient, raising=True)
    monkeypatch.setattr("context_runtime.tools.mcp.mount_mcp", lambda *a, **k: ["scan.scan_start"], raising=True)
    monkeypatch.setenv("SCAN_ALLOWLIST", "example.test")
    monkeypatch.setenv("CR_TEST_API_KEY", SECRET)
    rs.mount_deploy_scan(object())
    assert seen["env"]["SCAN_ALLOWLIST"] == "example.test"
    assert "CR_TEST_API_KEY" not in seen["env"]


def test_every_scan_setting_the_engine_reads_is_in_the_explicit_list():
    import re
    from pathlib import Path

    import context_runtime.integrations.runtime_scan as rs
    import context_runtime.integrations.runtime_scan_engine as engine

    names = set(re.findall(r'os\.getenv\("(SCAN_[A-Z_]+)"', Path(engine.__file__).read_text()))
    assert names, "no SCAN_ settings found: the pattern needs updating"
    assert names <= set(rs.DEPLOY_SCAN_ENV), sorted(names - set(rs.DEPLOY_SCAN_ENV))
