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
import hashlib
import logging
import os
import sys
import tempfile
import threading
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
    """True on a Linux host that is a container. Containers keep their historical layout.

    ``CONTEXT_RUNTIME_CONTAINER=1`` or ``0`` forces the answer and is the explicit override when detection is
    wrong. Otherwise a container is recognised by any of: the Docker or Podman marker files, a Kubernetes pod
    (``KUBERNETES_SERVICE_HOST``), Amazon ECS / Fargate (``ECS_CONTAINER_METADATA_URI`` or ``..._V4``,
    ``AWS_EXECUTION_ENV`` starting with ``AWS_ECS``), Google Cloud Run (``K_SERVICE``), or a container runtime
    named in ``/proc/1/cgroup``.
    """
    e = os.environ if env is None else env
    forced = e.get(CONTAINER_ENV, "").strip().lower()
    if forced in ("1", "true", "yes"):
        return True
    if forced in ("0", "false", "no"):
        return False
    if os_family(platform) != "linux":
        return False
    if any(e.get(k) for k in ("KUBERNETES_SERVICE_HOST", "ECS_CONTAINER_METADATA_URI",
                              "ECS_CONTAINER_METADATA_URI_V4", "K_SERVICE")):
        return True
    if e.get("AWS_EXECUTION_ENV", "").startswith("AWS_ECS"):
        return True
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    cg = _cgroup_text()
    return any(marker in cg for marker in ("docker", "kubepods", "containerd", "libpod", "ecs", "crio"))


def _cgroup_text() -> str:
    """The text of ``/proc/1/cgroup`` (empty when unreadable); a function so tests can substitute it."""
    try:
        with open("/proc/1/cgroup", encoding="utf-8", errors="replace") as fh:
            return fh.read(65536)
    except OSError:
        return ""


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
    ``.context-runtime`` that containers have always used) > a non-empty legacy ``.context-runtime`` in the
    working directory > the host convention. The legacy directory is used whenever it holds anything, even if
    the platform directory also exists, so a start never silently switches to empty state; one warning names
    both paths and says how to migrate. The keyword arguments exist for tests; production callers pass none.
    """
    e = os.environ if env is None else env
    explicit = _first_env(e, STATE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    if (in_container(e, platform) if container is None else container):
        return Path(LEGACY_STATE_DIRNAME)
    target = _platform_dir("state", e, os_family(platform), _home(e, home))
    legacy = Path(LEGACY_STATE_DIRNAME)
    if _non_empty_dir(legacy) or (legacy.is_dir() and not target.exists()):
        both = f" (the platform directory {target} also exists and is NOT used)" if target.exists() else ""
        _warn_once(f"using the existing working-directory state {legacy.resolve()}{both}; to move to the platform "
                   f"location, copy its contents to {target} and remove {legacy.resolve()}, or set "
                   "CONTEXT_RUNTIME_HOME to pick one explicitly")
        return legacy
    return target


def _non_empty_dir(path: Path) -> bool:
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def state_path(*parts: str, container_default: str | None = None, env: Mapping[str, str] | None = None,
               platform: str | None = None, home: Path | None = None, container: bool | None = None) -> Path:
    """A path below :func:`state_dir`. ``container_default`` is the absolute location a container has
    always used for this item (for example ``/data/context-runtime/policy``); in a container it always applies,
    including when ``CONTEXT_RUNTIME_HOME`` is set (as ``/data`` in the compose files): that variable never
    moved these items and must not start to. An item's own variable (``POLICY_DIR`` ...) is handled by its
    caller and wins over everything."""
    e = os.environ if env is None else env
    is_container = in_container(e, platform) if container is None else container
    if container_default and is_container:
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
    working-directory ``name`` containers have always used) > an existing working-directory file (legacy; used
    even when the config directory also has one, with a warning naming both) > ``config_dir() / name``."""
    e = os.environ if env is None else env
    explicit = _first_env(e, CONFIG_FILE_ENV)
    if explicit:
        return Path(explicit).expanduser()
    if in_container(e, platform) if container is None else container:
        return Path(name)
    target = config_dir(env=e, platform=platform, home=home) / name
    if Path(name).is_file():
        both = f" (the file {target} also exists and is NOT used)" if target.exists() else ""
        _warn_once(f"using the working-directory {Path(name).resolve()}{both}; to move to the platform location, "
                   f"copy it to {target} and remove it here, or set CONTEXT_RUNTIME_CONFIG to pick one explicitly")
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
    """``mkdir -p`` where every directory this call creates is 0700 on POSIX (``0777 & ~umask`` with
    ``CONTEXT_RUNTIME_SHARED=1``); existing directories are never changed. On Windows the new directory inherits the user-profile ACL, which is the intent for
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
            d.mkdir(mode=default_dir_mode()) if os.name == "posix" else d.mkdir()
        except FileExistsError:
            continue
        if os.name == "posix":
            with contextlib.suppress(OSError):
                os.chmod(d, default_dir_mode())   # mkdir's mode is masked by the umask
    return p


# ───────────────────────────────── permissions ─────────────────────────────────

SHARED_ENV = "CONTEXT_RUNTIME_SHARED"


def shared_directory() -> bool:
    """True when the operator declared that state is shared with other users or containers
    (``CONTEXT_RUNTIME_SHARED=1``). Files and lock files then get the process umask's default permissions
    instead of the private 0600 / 0700 used otherwise."""
    return os.environ.get(SHARED_ENV, "").strip().lower() in ("1", "true", "yes")


def _umask() -> int:
    old = os.umask(0)
    os.umask(old)
    return old


def default_file_mode() -> int:
    """Mode for a new state file: 0600, or ``0666 & ~umask`` when :func:`shared_directory`."""
    return (0o666 & ~_umask()) if shared_directory() else 0o600


def default_dir_mode() -> int:
    """Mode for a new state directory: 0700, or ``0777 & ~umask`` when :func:`shared_directory`."""
    return (0o777 & ~_umask()) if shared_directory() else 0o700


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
    """Make the rename itself durable. Directories cannot be opened on Windows; skip there. The data is
    already in place when this runs, so a failure to sync the directory is logged, not raised."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as e:
        _log.debug("fsync of directory %s failed (%s); the write itself succeeded", directory, e)
    finally:
        os.close(fd)


def _name_max(directory: Path) -> int:
    try:
        return int(os.pathconf(directory, "PC_NAME_MAX"))
    except (AttributeError, ValueError, OSError):
        return 255


def _fit_name(stem: str, extra_bytes: int, limit: int) -> str:
    """``stem`` shortened (on a character boundary) so that ``stem`` plus ``extra_bytes`` fits ``limit`` bytes."""
    room = max(8, limit - extra_bytes)
    raw = stem.encode("utf-8")
    if len(raw) <= room:
        return stem
    return raw[:room].decode("utf-8", errors="ignore")


def atomic_write(path: str | os.PathLike[str], data: str | bytes, *, encoding: str = "utf-8",
                 mode: int | None = None) -> None:
    """Replace ``path`` with ``data`` so a reader sees either the old file or the new one, never a partial.

    * The temporary file has a unique name (two writers never share it), is derived from the target name but
      shortened so it never exceeds the file system's name limit, and lives in the target's directory (the
      rename cannot cross a file system).
    * A symlink at ``path`` is written through (the link survives and its target is replaced), as
      ``Path.write_text`` did.
    * An existing file keeps its permissions; a read-only existing file raises ``PermissionError`` (POSIX,
      for a user who could not write it), as ``write_text`` did. A new file is created with ``mode``
      (default :func:`default_file_mode`: 0600, or the umask default when ``CONTEXT_RUNTIME_SHARED=1``).
    * Missing parent directories are created with :func:`ensure_private_dir`.
    """
    target = Path(os.path.realpath(path))
    parent = target.parent
    ensure_private_dir(parent)
    payload = data.encode(encoding) if isinstance(data, str) else data
    try:
        st = os.stat(target)
    except OSError:
        keep = default_file_mode() if mode is None else mode
    else:
        keep = st.st_mode & 0o777
        if os.name == "posix" and not os.access(target, os.W_OK):
            raise PermissionError(errno.EACCES, "target is read-only", str(target))
    prefix = "." + _fit_name(target.name, 4 + 8 + 2, _name_max(parent) - 1) + "."
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=".tmp")
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
                  timeout: float = 10.0, mode: int | None = None, durable: bool = True) -> None:
    """Append ``text`` to ``path`` under :func:`file_lock` as one write, so lines from concurrent threads and
    processes are never lost or interleaved.

    A bare ``open(path, "a")`` is not enough on Windows: the C runtime implements append as "seek to the end, then
    write", so two writers that seek before either writes overwrite each other's line (observed: lines lost in
    nearly every barrier-released round of three threads on a Windows runner; none on Linux or macOS). The lock
    makes the seek and the write one critical section on every platform. ``text`` is written as given (end it with
    ``"\n"`` for a line-oriented log) in a single ``write`` call, flushed, and ``fsync``-ed unless
    ``durable=False`` (for diagnostic logs where losing the last line in a crash is acceptable).
    """
    target = Path(path)
    ensure_private_dir(target.parent)
    with file_lock(lock_path_for(target), timeout=timeout, poll=0.005):
        existed = target.exists()
        with open(target, "ab") as fh:
            fh.write(text.encode(encoding))
            fh.flush()
            if durable:
                os.fsync(fh.fileno())
        if not existed and os.name == "posix":
            with contextlib.suppress(OSError):
                os.chmod(target, default_file_mode() if mode is None else mode)


# ───────────────────────────────── file lock ─────────────────────────────────

class LockTimeout(TimeoutError):
    """The lock could not be taken within the timeout."""


LOCKING_ENV = "CONTEXT_RUNTIME_LOCKING"          # strict | best-effort
#: errno values that mean "this file system cannot do advisory locks" (NFS without lockd, some FUSE, 9p, SMB).
_NO_LOCK_ERRNOS = {getattr(errno, n) for n in ("ENOLCK", "ENOTSUP", "EOPNOTSUPP", "ENOSYS", "EINVAL")
                   if hasattr(errno, n)}


def lock_path_for(path: str | os.PathLike[str]) -> Path:
    """The sidecar lock file for a data file: ``<name>.lock`` next to it (never the data file itself, which
    is replaced by rename). A very long name is shortened with a digest so the lock name fits."""
    p = Path(path)
    name = p.name
    if len(name.encode("utf-8")) + 5 > 255:
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = _fit_name(name, 5 + 9, 255) + "~" + digest
    return p.with_name(name + ".lock")


def locking_strict(governance: bool = False, env: Mapping[str, str] | None = None) -> bool:
    """Whether a lock that the file system cannot provide is an error. ``CONTEXT_RUNTIME_LOCKING=strict`` or
    ``best-effort`` decides; unset, stores that hold governance data (``governance=True``) are strict and the
    others best-effort."""
    e = os.environ if env is None else env
    v = e.get(LOCKING_ENV, "").strip().lower()
    if v in ("strict", "best-effort", "besteffort", "best_effort"):
        return v == "strict"
    return governance


_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()


def _thread_lock_for(path: Path) -> threading.Lock:
    key = os.path.abspath(path)
    with _thread_locks_guard:
        return _thread_locks.setdefault(key, threading.Lock())


def _try_lock(fh) -> bool:
    """True when the lock was taken, False when somebody else holds it. Raises ``OSError`` for anything else."""
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
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EDEADLK, getattr(errno, "EDEADLOCK", errno.EDEADLK)):
                return False       # lock violation: held by another handle
            raise
    raise RuntimeError("no file-locking primitive available on this platform")


def _unlock(fh) -> None:
    if fcntl is not None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - exercised on Windows
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


@contextlib.contextmanager
def file_lock(path: str | os.PathLike[str], *, timeout: float = 10.0, poll: float = 0.02,
              governance: bool = False, strict: bool | None = None) -> Iterator[None]:
    """Hold an exclusive inter-process lock on ``path`` (created if needed) for the block.

    Polls until ``timeout`` seconds have passed, then raises :class:`LockTimeout`. The OS releases the lock
    if the holder dies. Not re-entrant (a second acquisition of the same path in one process waits for the
    first and so deadlocks until the timeout); keep critical sections short.

    On a file system that cannot lock (``ENOLCK``, ``ENOTSUP``, ``EOPNOTSUPP``, ``ENOSYS``, ``EINVAL``), the
    behaviour follows ``CONTEXT_RUNTIME_LOCKING`` (or ``strict=``): ``strict`` re-raises the ``OSError``;
    ``best-effort`` logs ONE warning naming the path and falls back to a lock that serialises threads of this
    process only (other processes are NOT excluded). Unset, ``governance=True`` stores are strict and the
    rest best-effort. The lock file is 0600 (or the umask default with ``CONTEXT_RUNTIME_SHARED=1``).
    """
    lock_path = Path(path)
    ensure_private_dir(lock_path.parent)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, default_file_mode())
    fh = os.fdopen(fd, "r+b", buffering=0)
    thread_lock = None
    try:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                got = _try_lock(fh)
            except OSError as e:
                if e.errno not in _NO_LOCK_ERRNOS:
                    raise
                if locking_strict(governance) if strict is None else strict:
                    raise
                _warn_once(f"file locking is not supported at {lock_path} ({errno.errorcode.get(e.errno, e.errno)}); "
                           "serialising threads of this process only, other processes are NOT excluded. "
                           f"Set {LOCKING_ENV}=strict to fail instead.")
                thread_lock = _thread_lock_for(lock_path)
                if not thread_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                    thread_lock = None
                    raise LockTimeout(f"could not lock {lock_path} within {timeout:g}s") from None
                break
            if got:
                break
            if time.monotonic() >= deadline:
                raise LockTimeout(f"could not lock {lock_path} within {timeout:g}s")
            time.sleep(poll)
        try:
            yield
        finally:
            if thread_lock is not None:
                thread_lock.release()
            else:
                _unlock(fh)
    finally:
        fh.close()
