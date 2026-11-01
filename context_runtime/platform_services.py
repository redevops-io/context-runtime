"""Platform services: where state lives, atomic file replacement, and inter-process file locks.

One small module, standard library only, so every store resolves its location and persists its data the
same way on Linux, macOS and Windows:

* :func:`state_dir`, :func:`config_dir`, :func:`cache_dir` -- the per-user directories. An environment
  override always wins; then the host convention (macOS ``~/Library/Application Support`` and
  ``~/Library/Caches``, Windows ``%LOCALAPPDATA%``, Linux XDG). Containers keep the layout they have always
  used (see :func:`in_container`), so a Docker deployment does not move.
* :func:`atomic_write` -- a unique temporary file in the target's directory, flushed and ``fsync``-ed, then
  ``os.replace``-d over the target (bounded retry on Windows, where a reader or a scanner holding the
  target open makes the replace fail with ``PermissionError``), then the directory is ``fsync``-ed on POSIX.
* :func:`file_lock` -- an exclusive inter-process lock (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on
  Windows) with a timeout. It is not re-entrant: do not take the same lock twice in one call stack.
* :func:`ensure_private_dir` -- create directories (mode 0700 on POSIX, user-profile ACLs on Windows).
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]
try:  # Windows
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore[assignment]

_log = logging.getLogger("context_runtime.platform")

APP_NAME = "context-runtime"
STATE_ENV = ("CONTEXT_RUNTIME_HOME", "AGENTIC_OS_HOME")
CONFIG_DIR_ENV = "CONTEXT_RUNTIME_CONFIG_DIR"
CONFIG_FILE_ENV = ("CONTEXT_RUNTIME_CONFIG", "AGENTIC_OS_CONFIG")
CACHE_ENV = "CONTEXT_RUNTIME_CACHE_DIR"
CONTAINER_ENV = "CONTEXT_RUNTIME_CONTAINER"

#: The directory name a CWD-relative deployment used before platform directories existed.
LEGACY_STATE_DIRNAME = ".context-runtime"


# ───────────────────────────────── host detection ─────────────────────────────────

def os_family(platform: str | None = None) -> str:
    """``"windows"``, ``"macos"`` or ``"linux"`` (any other POSIX counts as linux)."""
    p = sys.platform if platform is None else platform
    if p.startswith("win") or p == "cygwin":
        return "windows"
    if p == "darwin":
        return "macos"
    return "linux"


def in_container(env: Mapping[str, str] | None = None, platform: str | None = None) -> bool:
    """True on a Linux host that is a container: ``CONTEXT_RUNTIME_CONTAINER`` (1/0 forces the answer), the
    Docker/Podman marker files, or a Kubernetes pod. Containers keep their historical layout."""
    e = os.environ if env is None else env
    forced = e.get(CONTAINER_ENV, "").strip().lower()
    if forced in ("1", "true", "yes"):
        return True
    if forced in ("0", "false", "no"):
        return False
    if os_family(platform) != "linux":
        return False
    return bool(e.get("KUBERNETES_SERVICE_HOST")) or os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


_warned: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        _log.warning(message)


def _first_env(e: Mapping[str, str], names: tuple[str, ...] | str) -> str:
    for n in ((names,) if isinstance(names, str) else names):
        v = e.get(n, "").strip()
        if v:
            return v
    return ""


def _home(env: Mapping[str, str], home: Path | None) -> Path:
    if home is not None:
        return home
    h = env.get("HOME") or env.get("USERPROFILE")
    return Path(h) if h else Path.home()


# ───────────────────────────────── directories ─────────────────────────────────

def _platform_dir(kind: str, env: Mapping[str, str], family: str, home: Path) -> Path:
    """The per-user directory of ``kind`` (``state``/``config``/``cache``) by host convention."""
    if family == "macos":
        base = home / "Library" / ("Caches" if kind == "cache" else "Application Support")
        return base / APP_NAME
    if family == "windows":
        local = env.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        base = Path(local) / APP_NAME
        return base if kind == "state" else base / kind
    xdg = {"state": ("XDG_DATA_HOME", ".local/share"), "config": ("XDG_CONFIG_HOME", ".config"),
           "cache": ("XDG_CACHE_HOME", ".cache")}[kind]
    root = env.get(xdg[0], "").strip()
    return (Path(root) if root and os.path.isabs(root) else home / xdg[1]) / APP_NAME


def state_dir(*, env: Mapping[str, str] | None = None, platform: str | None = None,
              home: Path | None = None, container: bool | None = None) -> Path:
    """Where durable state (approvals, policies, learned stats) lives.

    Order: ``CONTEXT_RUNTIME_HOME`` / ``AGENTIC_OS_HOME`` > container layout (the working-directory-relative
    ``.context-runtime`` that containers have always used) > an existing legacy ``.context-runtime`` in the
    working directory (kept so data written by an older version is not orphaned) > the host convention.
    The keyword arguments exist for tests; production callers pass none.
    """
    e = os.environ if env is None else env
    explicit = _first_env(e, STATE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    if (in_container(e, platform) if container is None else container):
        return Path(LEGACY_STATE_DIRNAME)
    target = _platform_dir("state", e, os_family(platform), _home(e, home))
    legacy = Path(LEGACY_STATE_DIRNAME)
    if not target.exists() and legacy.is_dir():
        _warn_once(f"using the existing working-directory state {legacy.resolve()}; move it to {target} "
                   "or set CONTEXT_RUNTIME_HOME")
        return legacy
    return target


def state_path(*parts: str, container_default: str | None = None, env: Mapping[str, str] | None = None,
               platform: str | None = None, home: Path | None = None, container: bool | None = None) -> Path:
    """A path below :func:`state_dir`. ``container_default`` is the absolute location a container has
    always used for this item (for example ``/data/context-runtime/policy``); it applies only in a
    container and only when ``CONTEXT_RUNTIME_HOME`` is not set."""
    e = os.environ if env is None else env
    is_container = in_container(e, platform) if container is None else container
    if container_default and is_container and not _first_env(e, STATE_ENV):
        return Path(container_default)
    return state_dir(env=e, platform=platform, home=home, container=is_container).joinpath(*parts)


def config_dir(*, env: Mapping[str, str] | None = None, platform: str | None = None,
               home: Path | None = None) -> Path:
    """Where user configuration lives: ``CONTEXT_RUNTIME_CONFIG_DIR`` > the host convention."""
    e = os.environ if env is None else env
    explicit = e.get(CONFIG_DIR_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _platform_dir("config", e, os_family(platform), _home(e, home))


def config_file(name: str = "config.yaml", *, env: Mapping[str, str] | None = None,
                platform: str | None = None, home: Path | None = None,
                container: bool | None = None) -> Path:
    """The configuration file: ``CONTEXT_RUNTIME_CONFIG`` / ``AGENTIC_OS_CONFIG`` > container layout (the
    working-directory ``name`` containers have always used) > an existing working-directory file with no
    file in the config directory (legacy) > ``config_dir() / name``."""
    e = os.environ if env is None else env
    explicit = _first_env(e, CONFIG_FILE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    if in_container(e, platform) if container is None else container:
        return Path(name)
    target = config_dir(env=e, platform=platform, home=home) / name
    if not target.exists() and Path(name).is_file():
        _warn_once(f"using the working-directory {Path(name).resolve()}; move it to {target}")
        return Path(name)
    return target


def cache_dir(*, env: Mapping[str, str] | None = None, platform: str | None = None,
              home: Path | None = None) -> Path:
    """Disposable caches: ``CONTEXT_RUNTIME_CACHE_DIR`` > the host convention."""
    e = os.environ if env is None else env
    explicit = e.get(CACHE_ENV, "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return _platform_dir("cache", e, os_family(platform), _home(e, home))


def ensure_private_dir(path: str | os.PathLike[str]) -> Path:
    """``mkdir -p`` where every directory this call creates is 0700 on POSIX (existing directories are
    never changed). On Windows the new directory inherits the user-profile ACL, which is the intent for
    ``%LOCALAPPDATA%``."""
    p = Path(path)
    missing: list[Path] = []
    cur = p
    while not cur.exists():
        missing.append(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    for d in reversed(missing):
        try:
            d.mkdir(mode=0o700) if os.name == "posix" else d.mkdir()
        except FileExistsError:
            continue
        if os.name == "posix":
            with contextlib.suppress(OSError):
                os.chmod(d, 0o700)   # mkdir's mode is masked by the umask
    return p


# ───────────────────────────────── atomic write ─────────────────────────────────

#: Windows reports a target that is open elsewhere (a reader, an antivirus scan) as PermissionError (or
#: WinError 32/33/5). Retry briefly; a persistent failure is re-raised.
REPLACE_RETRIES = 12
REPLACE_BACKOFF = 0.01          # seconds, doubled each retry, capped
REPLACE_BACKOFF_MAX = 0.25


def _replace_with_retry(src: str, dst: str, *, retries: int = REPLACE_RETRIES, sleep=time.sleep,
                        replace=os.replace) -> None:
    delay = REPLACE_BACKOFF
    for attempt in range(retries + 1):
        try:
            replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or attempt == retries:
                raise
            sleep(delay)
            delay = min(delay * 2, REPLACE_BACKOFF_MAX)


def _fsync_dir(directory: Path) -> None:
    """Make the rename itself durable. Directories cannot be opened on Windows; skip there."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as e:  # some file systems (and network mounts) refuse fsync on a directory
        if e.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF, errno.EACCES):
            raise
    finally:
        os.close(fd)


def atomic_write(path: str | os.PathLike[str], data: str | bytes, *, encoding: str = "utf-8",
                 mode: int = 0o600) -> None:
    """Replace ``path`` with ``data`` so a reader sees either the old file or the new one, never a partial.

    The temporary file has a unique name (two writers never share it) and lives in the target's directory
    (the rename cannot cross a file system). A new file is created ``mode`` (0600 by default, POSIX); an
    existing target keeps its permissions. The parent directory is created if missing.
    """
    target = Path(path)
    parent = target.parent
    ensure_private_dir(parent)
    payload = data.encode(encoding) if isinstance(data, str) else data
    try:
        keep = os.stat(target).st_mode & 0o777
    except OSError:
        keep = mode
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name == "posix":
            os.chmod(tmp, keep)
        _replace_with_retry(tmp, str(target))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(parent)


def locked_append(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8",
                  timeout: float = 10.0, mode: int = 0o600) -> None:
    """Append ``text`` to ``path`` under :func:`file_lock` as one write, flushed and ``fsync``-ed, so lines
    from concurrent processes are never interleaved."""
    target = Path(path)
    ensure_private_dir(target.parent)
    with file_lock(lock_path_for(target), timeout=timeout):
        existed = target.exists()
        with open(target, "ab") as fh:
            fh.write(text.encode(encoding))
            fh.flush()
            os.fsync(fh.fileno())
        if not existed and os.name == "posix":
            with contextlib.suppress(OSError):
                os.chmod(target, mode)


# ───────────────────────────────── file lock ─────────────────────────────────

class LockTimeout(TimeoutError):
    """The lock could not be taken within the timeout."""


def lock_path_for(path: str | os.PathLike[str]) -> Path:
    """The sidecar lock file for a data file: ``<name>.lock`` next to it (never the data file itself, which
    is replaced by rename)."""
    p = Path(path)
    return p.with_name(p.name + ".lock")


def _try_lock(fh) -> bool:
    if fcntl is not None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
    if msvcrt is not None:  # pragma: no cover - exercised on Windows
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    raise RuntimeError("no file-locking primitive available on this platform")


def _unlock(fh) -> None:
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - exercised on Windows
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def file_lock(path: str | os.PathLike[str], *, timeout: float = 10.0, poll: float = 0.02) -> Iterator[None]:
    """Hold an exclusive inter-process lock on ``path`` (created if needed, mode 0600) for the block.

    Polls until ``timeout`` seconds have passed, then raises :class:`LockTimeout`. The OS releases the lock
    if the holder dies. Not re-entrant (a second acquisition of the same path in one process waits for the
    first and so deadlocks until the timeout); keep critical sections short.
    """
    lock_path = Path(path)
    ensure_private_dir(lock_path.parent)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fh = os.fdopen(fd, "r+b", buffering=0)
    try:
        deadline = time.monotonic() + max(0.0, timeout)
        while not _try_lock(fh):
            if time.monotonic() >= deadline:
                raise LockTimeout(f"could not lock {lock_path} within {timeout:g}s")
            time.sleep(poll)
        try:
            yield
        finally:
            _unlock(fh)
    finally:
        fh.close()
