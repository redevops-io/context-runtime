"""Platform services: directory resolution, atomic writes, inter-process locks (runs on every OS)."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from context_runtime import platform_services as ps
from context_runtime.policy.store import RuleStore

POSIX = os.name == "posix"


# ───────────────────────── directory resolution (pure: host passed in) ─────────────────────────

HOME = Path("/home/u") if POSIX else Path("C:/Users/u")


def test_state_dir_environment_override_wins_everywhere():
    for plat in ("linux", "darwin", "win32"):
        env = {"CONTEXT_RUNTIME_HOME": "/srv/cr", "LOCALAPPDATA": "C:/x"}
        assert ps.state_dir(env=env, platform=plat, home=HOME, container=False) == Path("/srv/cr")
    # the older variable still works, the newer one wins
    assert ps.state_dir(env={"AGENTIC_OS_HOME": "/old"}, platform="linux", home=HOME, container=False) == Path("/old")
    both = {"AGENTIC_OS_HOME": "/old", "CONTEXT_RUNTIME_HOME": "/new"}
    assert ps.state_dir(env=both, platform="linux", home=HOME, container=False) == Path("/new")


def test_state_dir_by_host_convention(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)          # no legacy ./.context-runtime here
    assert ps.state_dir(env={}, platform="darwin", home=HOME, container=False) == \
        HOME / "Library" / "Application Support" / "context-runtime"
    assert ps.state_dir(env={"LOCALAPPDATA": "C:/L"}, platform="win32", home=HOME, container=False) == \
        Path("C:/L") / "context-runtime"
    assert ps.state_dir(env={}, platform="win32", home=HOME, container=False) == \
        HOME / "AppData" / "Local" / "context-runtime"
    assert ps.state_dir(env={}, platform="linux", home=HOME, container=False) == \
        HOME / ".local" / "share" / "context-runtime"
    assert ps.state_dir(env={"XDG_DATA_HOME": str(tmp_path / "xdg")}, platform="linux", home=HOME,
                        container=False) == tmp_path / "xdg" / "context-runtime"


def test_config_and_cache_dirs_by_host_convention():
    assert ps.cache_dir(env={}, platform="darwin", home=HOME) == HOME / "Library" / "Caches" / "context-runtime"
    assert ps.config_dir(env={}, platform="darwin", home=HOME) == \
        HOME / "Library" / "Application Support" / "context-runtime"
    assert ps.cache_dir(env={"LOCALAPPDATA": "C:/L"}, platform="win32", home=HOME) == \
        Path("C:/L") / "context-runtime" / "cache"
    assert ps.config_dir(env={"LOCALAPPDATA": "C:/L"}, platform="win32", home=HOME) == \
        Path("C:/L") / "context-runtime" / "config"
    assert ps.cache_dir(env={}, platform="linux", home=HOME) == HOME / ".cache" / "context-runtime"
    assert ps.config_dir(env={}, platform="linux", home=HOME) == HOME / ".config" / "context-runtime"
    assert ps.cache_dir(env={"CONTEXT_RUNTIME_CACHE_DIR": "/c"}, platform="darwin", home=HOME) == Path("/c")
    assert ps.config_dir(env={"CONTEXT_RUNTIME_CONFIG_DIR": "/cfg"}, platform="win32", home=HOME) == Path("/cfg")


def test_relative_xdg_values_are_ignored():
    env = {"XDG_CACHE_HOME": "relative/dir"}
    assert ps.cache_dir(env=env, platform="linux", home=HOME) == HOME / ".cache" / "context-runtime"


def test_container_keeps_its_historical_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ps.state_dir(env={}, platform="linux", home=HOME, container=True) == Path(".context-runtime")
    # an item with a fixed container location keeps it ...
    assert ps.state_path("policy", container_default="/data/context-runtime/policy", env={}, platform="linux",
                         home=HOME, container=True) == Path("/data/context-runtime/policy")
    # ... unless the operator pointed the home somewhere
    assert ps.state_path("policy", container_default="/data/context-runtime/policy",
                         env={"CONTEXT_RUNTIME_HOME": "/h"}, platform="linux", home=HOME,
                         container=True) == Path("/h/policy")
    assert ps.config_file(env={}, platform="linux", home=HOME, container=True) == Path("config.yaml")
    assert ps.config_file(env={"CONTEXT_RUNTIME_CONFIG": "/etc/cr.yaml"}, platform="linux", home=HOME,
                          container=True) == Path("/etc/cr.yaml")


def test_outside_a_container_no_item_uses_a_data_root_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = ps.state_path("policy", container_default="/data/context-runtime/policy", env={}, platform="darwin",
                      home=HOME, container=False)
    assert p == HOME / "Library" / "Application Support" / "context-runtime" / "policy"


def test_container_detection_can_be_forced():
    assert ps.in_container({"CONTEXT_RUNTIME_CONTAINER": "1"}, "darwin")
    assert not ps.in_container({"CONTEXT_RUNTIME_CONTAINER": "0", "KUBERNETES_SERVICE_HOST": "10.0.0.1"}, "linux")
    assert ps.in_container({"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, "linux")
    assert not ps.in_container({"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, "win32")


def test_existing_working_directory_state_is_kept_not_orphaned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".context-runtime").mkdir()
    got = ps.state_dir(env={}, platform="linux", home=tmp_path / "home", container=False)
    assert got == Path(".context-runtime")
    # once the platform directory exists it takes over
    (tmp_path / "home" / ".local" / "share" / "context-runtime").mkdir(parents=True)
    assert ps.state_dir(env={}, platform="linux", home=tmp_path / "home", container=False) == \
        tmp_path / "home" / ".local" / "share" / "context-runtime"


def test_config_file_legacy_working_directory_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("router: {}\n", encoding="utf-8")
    assert ps.config_file(env={}, platform="linux", home=tmp_path / "h", container=False) == Path("config.yaml")
    cfg = tmp_path / "h" / ".config" / "context-runtime"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("router: {}\n", encoding="utf-8")
    assert ps.config_file(env={}, platform="linux", home=tmp_path / "h", container=False) == cfg / "config.yaml"


def test_a_fresh_home_with_no_legacy_state_yields_no_cwd_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for plat in ("linux", "darwin", "win32"):
        assert ps.state_dir(env={}, platform=plat, home=tmp_path / "h", container=False).is_absolute()


# ───────────────────────── ensure_private_dir ─────────────────────────

@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_ensure_private_dir_creates_0700_and_leaves_existing_alone(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir(mode=0o755)
    existing.chmod(0o755)
    leaf = ps.ensure_private_dir(existing / "a" / "b")
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o700
    assert stat.S_IMODE((existing / "a").stat().st_mode) == 0o700
    assert stat.S_IMODE(existing.stat().st_mode) == 0o755


def test_ensure_private_dir_is_idempotent(tmp_path):
    a = ps.ensure_private_dir(tmp_path / "x" / "y")
    assert ps.ensure_private_dir(a) == a and a.is_dir()


# ───────────────────────── atomic_write ─────────────────────────

def test_atomic_write_creates_parents_and_round_trips(tmp_path):
    target = tmp_path / "deep" / "dir" / "data.json"
    ps.atomic_write(target, '{"k": "ünï"}')
    assert target.read_text(encoding="utf-8") == '{"k": "ünï"}'
    ps.atomic_write(target, b"\x00\x01")
    assert target.read_bytes() == b"\x00\x01"


def test_atomic_write_leaves_no_temporary_files(tmp_path):
    for i in range(5):
        ps.atomic_write(tmp_path / "f.txt", f"v{i}")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.txt"]


def test_atomic_write_failure_keeps_the_old_content_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "f.txt"
    ps.atomic_write(target, "old")

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(ps, "_replace_with_retry", boom)
    with pytest.raises(OSError):
        ps.atomic_write(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_atomic_write_temp_names_are_unique(tmp_path, monkeypatch):
    seen: list[str] = []
    real = ps._replace_with_retry

    def spy(src, dst, **k):
        seen.append(src)
        return real(src, dst, **k)

    monkeypatch.setattr(ps, "_replace_with_retry", spy)
    for _ in range(4):
        ps.atomic_write(tmp_path / "f.txt", "x")
    assert len(set(seen)) == 4 and all(Path(s).parent == tmp_path for s in seen)


@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_atomic_write_permissions_new_file_private_existing_file_preserved(tmp_path):
    new = tmp_path / "new"
    ps.atomic_write(new, "x")
    assert stat.S_IMODE(new.stat().st_mode) == 0o600
    shared = tmp_path / "shared"
    shared.write_text("x", encoding="utf-8")
    shared.chmod(0o640)
    ps.atomic_write(shared, "y")
    assert stat.S_IMODE(shared.stat().st_mode) == 0o640


def test_replace_retries_on_windows_permission_error_then_succeeds(monkeypatch):
    monkeypatch.setattr(ps.os, "name", "nt")
    calls, sleeps = [], []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 4:
            raise PermissionError(13, "in use")

    ps._replace_with_retry("a", "b", sleep=sleeps.append, replace=flaky)
    assert len(calls) == 4 and len(sleeps) == 3 and sleeps == sorted(sleeps)
    assert max(sleeps) <= ps.REPLACE_BACKOFF_MAX


def test_replace_gives_up_after_the_bounded_retries(monkeypatch):
    monkeypatch.setattr(ps.os, "name", "nt")
    calls, sleeps = [], []

    def stuck(src, dst):
        calls.append(1)
        raise PermissionError(13, "in use")

    with pytest.raises(PermissionError):
        ps._replace_with_retry("a", "b", retries=3, sleep=sleeps.append, replace=stuck)
    assert len(calls) == 4 and len(sleeps) == 3


def test_replace_does_not_retry_on_posix(monkeypatch):
    monkeypatch.setattr(ps.os, "name", "posix")
    calls = []

    def denied(src, dst):
        calls.append(1)
        raise PermissionError(13, "denied")

    with pytest.raises(PermissionError):
        ps._replace_with_retry("a", "b", sleep=lambda s: None, replace=denied)
    assert len(calls) == 1


def test_atomic_write_waits_for_a_reader_that_briefly_holds_the_target_open(tmp_path):
    """On Windows a file open elsewhere cannot be replaced until it is closed (PermissionError); the bounded
    retry rides out a short hold such as an antivirus scan or a reader. On POSIX the replace is immediate."""
    import threading
    target = tmp_path / "f.txt"
    ps.atomic_write(target, "one")
    reader = open(target, encoding="utf-8")  # noqa: SIM115 - held open on purpose
    timer = threading.Timer(0.3, reader.close)
    timer.start()
    try:
        ps.atomic_write(target, "two")
    finally:
        timer.join()
        reader.close()
    assert target.read_text(encoding="utf-8") == "two"


# ───────────────────────── file_lock ─────────────────────────

def test_file_lock_is_exclusive_and_times_out(tmp_path):
    lock = tmp_path / "x.lock"
    with ps.file_lock(lock):
        t0 = time.monotonic()
        with pytest.raises(ps.LockTimeout):
            with ps.file_lock(lock, timeout=0.2):
                pass
        assert time.monotonic() - t0 >= 0.19
    with ps.file_lock(lock, timeout=1):     # released afterwards
        pass


def test_file_lock_is_released_when_the_block_raises(tmp_path):
    lock = tmp_path / "x.lock"
    with pytest.raises(RuntimeError):
        with ps.file_lock(lock):
            raise RuntimeError("boom")
    with ps.file_lock(lock, timeout=1):
        pass


_WORKER = textwrap.dedent("""
    import sys
    from context_runtime.platform_services import file_lock, atomic_write, lock_path_for
    path, n = sys.argv[1], int(sys.argv[2])
    for _ in range(n):
        with file_lock(lock_path_for(path), timeout=60):
            try:
                cur = int(open(path, encoding="utf-8").read())
            except FileNotFoundError:
                cur = 0
            atomic_write(path, str(cur + 1))
""")


def test_file_lock_serialises_read_modify_write_across_processes(tmp_path):
    counter = tmp_path / "counter.txt"
    procs = [subprocess.Popen([sys.executable, "-c", _WORKER, str(counter), "25"]) for _ in range(4)]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0, 0]
    assert counter.read_text(encoding="utf-8") == "100"          # no lost update
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_locked_append_never_interleaves_lines_from_concurrent_processes(tmp_path):
    log = tmp_path / "log.jsonl"
    code = textwrap.dedent("""
        import sys
        from context_runtime.platform_services import locked_append
        for i in range(40):
            locked_append(sys.argv[1], sys.argv[2] + ":" + "x" * 500 + "\\n")
    """)
    procs = [subprocess.Popen([sys.executable, "-c", code, str(log), tag]) for tag in "ABC"]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 120 and all(ln[2:] == "x" * 500 and ln[0] in "ABC" for ln in lines)


# ───────────────────────── stores built on the helpers ─────────────────────────

_RULE_WORKER = textwrap.dedent("""
    import sys
    from context_runtime.policy.store import RuleStore
    store, tag = RuleStore(sys.argv[1]), sys.argv[2]
    for i in range(15):
        store.add("global", "rule", f"{tag}-{i}")
""")


def test_rule_store_keeps_every_rule_added_by_concurrent_processes(tmp_path):
    procs = [subprocess.Popen([sys.executable, "-c", _RULE_WORKER, str(tmp_path), tag]) for tag in "ABC"]
    assert [p.wait(timeout=120) for p in procs] == [0, 0, 0]
    texts = {r.text for r in RuleStore(str(tmp_path)).list(scope="global")}
    assert texts == {f"{t}-{i}" for t in "ABC" for i in range(15)}


def test_policy_default_directory_comes_from_the_resolver(tmp_path, monkeypatch):
    monkeypatch.delenv("POLICY_DIR", raising=False)
    monkeypatch.setenv("CONTEXT_RUNTIME_HOME", str(tmp_path / "home"))
    assert Path(RuleStore().dir) == tmp_path / "home" / "policy"
    monkeypatch.setenv("POLICY_DIR", str(tmp_path / "explicit"))
    assert Path(RuleStore().dir) == tmp_path / "explicit"


def test_market_radar_defaults_come_from_the_resolver(tmp_path, monkeypatch):
    from context_runtime.integrations.job_leads import OutreachLedger, PitchTemplate, ProfileStore
    for v in ("LEAD_PROFILE_DIR", "PITCH_TEMPLATE_PATH", "OUTREACH_LEDGER_PATH", "OUTREACH_OUTBOX"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("CONTEXT_RUNTIME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_RUNTIME_CONTAINER", "0")
    assert Path(ProfileStore().dir) == tmp_path / "home" / "market-radar" / "profiles"
    assert Path(PitchTemplate().path) == tmp_path / "home" / "market-radar" / "pitch_template.txt"
    assert Path(OutreachLedger().path) == tmp_path / "home" / "market-radar" / "outreach_ledger.jsonl"


def test_market_radar_container_defaults_are_unchanged(monkeypatch):
    from context_runtime.integrations.job_leads import ProfileStore
    for v in ("LEAD_PROFILE_DIR", "CONTEXT_RUNTIME_HOME", "AGENTIC_OS_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("CONTEXT_RUNTIME_CONTAINER", "1")
    assert Path(ProfileStore().dir) == Path("/data/market-radar/profiles")


def test_control_plane_state_defaults_to_no_working_directory_path(tmp_path, monkeypatch):
    from context_runtime.control_plane.fleet import Fleet
    from context_runtime.control_plane.registry import Registry
    monkeypatch.delenv("CONTEXT_RUNTIME_HOME", raising=False)
    monkeypatch.delenv("AGENTIC_OS_HOME", raising=False)
    monkeypatch.setenv("CONTEXT_RUNTIME_CONTAINER", "0")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "h" / "AppData" / "Local"))
    monkeypatch.chdir(tmp_path)
    fleet = Fleet(Registry.load())
    assert fleet.home.is_absolute() and str(tmp_path / "h") in str(fleet.home)
    assert not (tmp_path / ".context-runtime").exists()
