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
    # ... and keeps it when CONTEXT_RUNTIME_HOME is set (the compose files set it to /data): that variable never
    # moved the policy directory and must not start to (rules already stored there would be orphaned)
    assert ps.state_path("policy", container_default="/data/context-runtime/policy",
                         env={"CONTEXT_RUNTIME_HOME": "/data"}, platform="linux", home=HOME,
                         container=True) == Path("/data/context-runtime/policy")
    assert ps.state_dir(env={"CONTEXT_RUNTIME_HOME": "/data"}, platform="linux", home=HOME,
                        container=True) == Path("/data")
    assert ps.config_file(env={}, platform="linux", home=HOME, container=True) == Path("config.yaml")
    assert ps.config_file(env={"CONTEXT_RUNTIME_CONFIG": "/etc/cr.yaml"}, platform="linux", home=HOME,
                          container=True) == Path("/etc/cr.yaml")


def test_outside_a_container_no_item_uses_a_data_root_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = ps.state_path("policy", container_default="/data/context-runtime/policy", env={}, platform="darwin",
                      home=HOME, container=False)
    assert p == HOME / "Library" / "Application Support" / "context-runtime" / "policy"


def test_container_detection_real_branches(monkeypatch):
    """No forcing: the environment variables, marker files and cgroup text a real container shows."""
    monkeypatch.setattr(ps.os.path, "exists", lambda p: False)
    monkeypatch.setattr(ps, "_cgroup_text", lambda: "")
    assert not ps.in_container({}, "linux")
    for env in ({"ECS_CONTAINER_METADATA_URI": "http://169.254.170.2/v3"},
                {"ECS_CONTAINER_METADATA_URI_V4": "http://169.254.170.2/v4"},
                {"AWS_EXECUTION_ENV": "AWS_ECS_FARGATE"}, {"K_SERVICE": "svc"},
                {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
        assert ps.in_container(env, "linux"), env
    assert not ps.in_container({"AWS_EXECUTION_ENV": "AWS_Lambda_python3.12"}, "linux")     # not claimed
    for marker in ("/.dockerenv", "/run/.containerenv"):
        monkeypatch.setattr(ps.os.path, "exists", lambda p, m=marker: p == m)
        assert ps.in_container({}, "linux"), marker
    monkeypatch.setattr(ps.os.path, "exists", lambda p: False)
    for cg in ("12:memory:/docker/abc", "0::/kubepods/burstable/pod1", "1:name=systemd:/ecs/task/abc"):
        monkeypatch.setattr(ps, "_cgroup_text", lambda cg=cg: cg)
        assert ps.in_container({}, "linux"), cg
    monkeypatch.setattr(ps, "_cgroup_text", lambda: "0::/user.slice/user-1000.slice/session-3.scope")
    assert not ps.in_container({}, "linux")
    # the explicit override beats every signal in both directions
    assert not ps.in_container({"CONTEXT_RUNTIME_CONTAINER": "0", "K_SERVICE": "svc"}, "linux")
    assert ps.in_container({"CONTEXT_RUNTIME_CONTAINER": "1"}, "linux")
    assert not ps.in_container({"K_SERVICE": "svc"}, "darwin")


def test_cgroup_text_is_empty_when_unreadable(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no /proc")))
    assert ps._cgroup_text() == ""


def test_container_detection_can_be_forced():
    assert ps.in_container({"CONTEXT_RUNTIME_CONTAINER": "1"}, "darwin")
    assert not ps.in_container({"CONTEXT_RUNTIME_CONTAINER": "0", "KUBERNETES_SERVICE_HOST": "10.0.0.1"}, "linux")
    assert ps.in_container({"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, "linux")
    assert not ps.in_container({"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, "win32")


def test_existing_working_directory_state_is_kept_not_orphaned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".context-runtime").mkdir()
    got = ps.state_dir(env={}, platform="linux", home=tmp_path / "home", container=False)
    assert got == Path(".context-runtime")       # empty legacy directory and no platform directory: legacy
    # an EMPTY legacy directory no longer wins once the platform directory exists
    platform_dir = tmp_path / "home" / ".local" / "share" / "context-runtime"
    platform_dir.mkdir(parents=True)
    assert ps.state_dir(env={}, platform="linux", home=tmp_path / "home", container=False) == platform_dir


def test_a_non_empty_legacy_directory_beats_an_existing_platform_directory_with_one_warning(
        tmp_path, monkeypatch, caplog):
    """Never start with empty state: pending approvals and policies in ./.context-runtime must not vanish
    because another run created the platform directory."""
    import logging
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ps, "_warned", set())
    legacy = tmp_path / ".context-runtime"
    legacy.mkdir()
    (legacy / "approvals.jsonl").write_text("{}\n", encoding="utf-8")
    platform_dir = tmp_path / "home" / ".local" / "share" / "context-runtime"
    platform_dir.mkdir(parents=True)
    with caplog.at_level(logging.WARNING, logger="context_runtime.platform"):
        for _ in range(3):
            got = ps.state_dir(env={}, platform="linux", home=tmp_path / "home", container=False)
    assert got == Path(".context-runtime")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(legacy.resolve()) in warnings[0] and str(platform_dir) in warnings[0]
    assert "copy its contents" in warnings[0]                       # says how to migrate
    # an explicit variable still decides
    assert ps.state_dir(env={"CONTEXT_RUNTIME_HOME": str(platform_dir)}, platform="linux", home=tmp_path / "home",
                        container=False) == platform_dir


def test_a_working_directory_config_file_beats_an_existing_platform_one_with_a_warning(tmp_path, monkeypatch, caplog):
    import logging
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ps, "_warned", set())
    (tmp_path / "config.yaml").write_text("router: {}\n", encoding="utf-8")
    cfg = tmp_path / "h" / ".config" / "context-runtime"
    cfg.mkdir(parents=True)
    (cfg / "config.yaml").write_text("router: {tiers: []}\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="context_runtime.platform"):
        got = ps.config_file(env={}, platform="linux", home=tmp_path / "h", container=False)
    assert got == Path("config.yaml")
    msg = [r.getMessage() for r in caplog.records][0]
    assert str(cfg / "config.yaml") in msg and "NOT used" in msg


def test_config_file_legacy_working_directory_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("router: {}\n", encoding="utf-8")
    assert ps.config_file(env={}, platform="linux", home=tmp_path / "h", container=False) == Path("config.yaml")
    # with no working-directory file the platform one is used
    (tmp_path / "config.yaml").unlink()
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


def test_compose_scenario_policy_and_market_radar_stay_under_data_when_the_home_is_data(monkeypatch):
    """deploy/proxmox-demo/chat-demo.compose.yml: a container with CONTEXT_RUNTIME_HOME=/data and no other
    path variable. Policy rules and market-radar files live where they always did."""
    from context_runtime.integrations.job_leads import OutreachLedger, PitchTemplate, ProfileStore
    for v in ("POLICY_DIR", "LEAD_PROFILE_DIR", "PITCH_TEMPLATE_PATH", "OUTREACH_LEDGER_PATH", "OUTREACH_OUTBOX",
              "AGENTIC_OS_HOME"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("CONTEXT_RUNTIME_CONTAINER", "1")
    monkeypatch.setenv("CONTEXT_RUNTIME_HOME", "/data")
    assert Path(RuleStore().dir) == Path("/data/context-runtime/policy")
    assert Path(ProfileStore().dir) == Path("/data/market-radar/profiles")
    assert Path(PitchTemplate().path) == Path("/data/market-radar/pitch_template.txt")
    assert Path(OutreachLedger().path) == Path("/data/market-radar/outreach_ledger.jsonl")
    # an item's own variable still wins
    monkeypatch.setenv("POLICY_DIR", "/elsewhere")
    assert Path(RuleStore().dir) == Path("/elsewhere")


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


# ───────────────────────── atomic_write: Linux file semantics ─────────────────────────

@pytest.mark.skipif(not POSIX, reason="symlinks and permission bits")
def test_atomic_write_writes_through_a_symlink(tmp_path):
    real = tmp_path / "real" / "state.json"
    real.parent.mkdir()
    real.write_text("old", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    ps.atomic_write(link, "new")
    assert link.is_symlink() and link.read_text(encoding="utf-8") == "new"      # the link survived
    assert real.read_text(encoding="utf-8") == "new"                           # and its target was replaced
    assert sorted(p.name for p in real.parent.iterdir()) == ["state.json"]    # no temp file left


@pytest.mark.skipif(not POSIX, reason="symlinks")
def test_atomic_write_through_a_dangling_symlink_creates_the_target(tmp_path):
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "later.json")
    ps.atomic_write(link, "x")
    assert link.is_symlink() and (tmp_path / "later.json").read_text(encoding="utf-8") == "x"


@pytest.mark.skipif(not POSIX or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="needs an unprivileged POSIX user")
def test_atomic_write_refuses_a_read_only_target_like_write_text(tmp_path):
    f = tmp_path / "ro.txt"
    f.write_text("keep", encoding="utf-8")
    f.chmod(0o444)
    with pytest.raises(PermissionError):
        ps.atomic_write(f, "overwrite")
    assert f.read_text(encoding="utf-8") == "keep" and [p.name for p in tmp_path.iterdir()] == ["ro.txt"]
    with pytest.raises(PermissionError):
        f.write_text("overwrite", encoding="utf-8")                              # what it did before


def test_atomic_write_handles_target_names_near_the_name_limit(tmp_path):
    for n in (200, 230, 250):
        name = "n" * n + ".txt"
        try:
            (tmp_path / name).write_text("probe", encoding="utf-8")
        except OSError:
            continue                                                              # this file system's limit is lower
        (tmp_path / name).unlink()
        ps.atomic_write(tmp_path / name, f"v{n}")
        assert (tmp_path / name).read_text(encoding="utf-8") == f"v{n}"
        assert [p.name for p in tmp_path.iterdir()] == [name]
        (tmp_path / name).unlink()


def test_temp_and_lock_names_fit_the_limit_for_multibyte_names():
    name = "é" * 120                                                              # 240 bytes
    assert len(ps._fit_name(name, 14, 255).encode("utf-8")) <= 255 - 14
    assert len(ps.lock_path_for(Path("/x") / ("é" * 125)).name.encode("utf-8")) <= 255
    assert ps.lock_path_for(Path("/x/a.json")).name == "a.json.lock"


@pytest.mark.skipif(not POSIX, reason="POSIX permission bits")
def test_shared_setting_gives_the_umask_default_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTEXT_RUNTIME_SHARED", "1")
    old = os.umask(0o022)
    try:
        d = ps.ensure_private_dir(tmp_path / "shared" / "sub")
        ps.atomic_write(d / "f.json", "x")
        ps.locked_append(d / "log.jsonl", "line\n")
        with ps.file_lock(ps.lock_path_for(d / "f.json")):
            pass
        assert stat.S_IMODE(d.stat().st_mode) == 0o755
        assert stat.S_IMODE((d / "f.json").stat().st_mode) == 0o644
        assert stat.S_IMODE((d / "log.jsonl").stat().st_mode) == 0o644
        assert stat.S_IMODE((d / "f.json.lock").stat().st_mode) == 0o644
        monkeypatch.delenv("CONTEXT_RUNTIME_SHARED")
        ps.atomic_write(d / "private.json", "x")
        with ps.file_lock(ps.lock_path_for(d / "private.json")):
            pass
        assert stat.S_IMODE((d / "private.json").stat().st_mode) == 0o600
        assert stat.S_IMODE((d / "private.json.lock").stat().st_mode) == 0o600
    finally:
        os.umask(old)


# ───────────────────────── file_lock where the file system cannot lock ─────────────────────────

def _no_flock(monkeypatch, code):
    import errno as _errno

    class _Fcntl:
        LOCK_EX, LOCK_NB, LOCK_UN = 2, 4, 8

        @staticmethod
        def flock(fd, op):
            if op & 8:
                return
            raise OSError(getattr(_errno, code), "locking not supported")

    monkeypatch.setattr(ps, "fcntl", _Fcntl)
    monkeypatch.setattr(ps, "_warned", set())


@pytest.mark.parametrize("code", ["ENOLCK", "ENOTSUP", "EOPNOTSUPP"])
def test_best_effort_lock_degrades_to_an_in_process_lock_with_one_warning(tmp_path, monkeypatch, caplog, code):
    import logging
    _no_flock(monkeypatch, code)
    monkeypatch.delenv("CONTEXT_RUNTIME_LOCKING", raising=False)
    lock = tmp_path / "x.lock"
    with caplog.at_level(logging.WARNING, logger="context_runtime.platform"):
        for _ in range(3):
            with ps.file_lock(lock):
                pass
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(lock) in warnings[0] and "other processes are NOT excluded" in warnings[0]


def test_degraded_lock_still_excludes_threads_of_this_process(tmp_path, monkeypatch):
    _no_flock(monkeypatch, "ENOLCK")
    lock = tmp_path / "x.lock"
    with ps.file_lock(lock):
        with pytest.raises(ps.LockTimeout):
            with ps.file_lock(lock, timeout=0.2):
                pass
    with ps.file_lock(lock, timeout=1):
        pass


def test_strict_locking_raises_instead_of_degrading(tmp_path, monkeypatch):
    _no_flock(monkeypatch, "ENOLCK")
    monkeypatch.setenv("CONTEXT_RUNTIME_LOCKING", "strict")
    with pytest.raises(OSError) as e:
        with ps.file_lock(tmp_path / "x.lock"):
            pass
    assert not isinstance(e.value, ps.LockTimeout)
    monkeypatch.setenv("CONTEXT_RUNTIME_LOCKING", "best-effort")
    with ps.file_lock(tmp_path / "x.lock", governance=True):                     # the variable beats the default
        pass


def test_governance_stores_are_strict_by_default_and_others_best_effort(tmp_path, monkeypatch):
    _no_flock(monkeypatch, "ENOLCK")
    monkeypatch.delenv("CONTEXT_RUNTIME_LOCKING", raising=False)
    assert ps.locking_strict(governance=True) and not ps.locking_strict(governance=False)
    with pytest.raises(OSError):
        with ps.file_lock(tmp_path / "gov.lock", governance=True):
            pass
    with ps.file_lock(tmp_path / "other.lock"):
        pass
    store = RuleStore(str(tmp_path / "rules"))                                   # policy rules are governance data
    with pytest.raises(OSError):
        store.add("global", "rule", "x")
    monkeypatch.setenv("CONTEXT_RUNTIME_LOCKING", "best-effort")
    store.add("global", "rule", "x")
    assert [r.text for r in store.list(scope="global")] == ["x"]


def test_an_unrelated_oserror_from_flock_is_not_swallowed(tmp_path, monkeypatch):
    _no_flock(monkeypatch, "EBADF")
    with pytest.raises(OSError):
        with ps.file_lock(tmp_path / "x.lock"):
            pass
