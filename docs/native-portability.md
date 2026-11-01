# Native Windows and macOS: what changed and what is verified

This page describes the portability changes in the core library (state locations, atomic writes, file locks,
deterministic corpus enumeration, the DuckDB `fts` extension) and states, test by test, what is verified on
which operating system. It makes no claim of full parity: anything not listed as verified is not verified.

## Breaking changes

### Windows reserved device names are refused as corpus identifiers (all operating systems)

A corpus reference whose path segment is a Windows device name is refused on **every** operating system, so that an
identifier maps to the same, valid file name everywhere. The match is case-insensitive and applies with or without
an extension (`nul.txt` and `com1.tar.gz` are refused too).

Refused names: `con`, `prn`, `aux`, `nul`, `conin$`, `conout$`, `com0` to `com9`, `lpt0` to `lpt9` (the superscript
digits `com¹ com² com³ lpt¹ lpt² lpt³` as well).

Error: `POST /librechat/ingest` answers `400` with `{"detail": "invalid corpus reference"}` (the same answer as any
other invalid reference). Before, `con`, `nul` and the others were accepted on Linux and macOS.

Related: the policy store maps a scope name that is a device name to `<name>~` for its file name (`con` is stored
under `con~/app.jsonl`; `con` and `con_` remain two different files), and shortens a file-name component over 128
characters to `...~<digest>`. Rules that were saved under a scope named like a device name, or under a name longer
than 128 characters, before this change are not found under the new file name; rename the directory or file by hand
if you have such a scope. Nothing is migrated automatically.

### A default `Archive()` is temporary

`Archive()` without a `directory` now lives in a `TemporaryDirectory` that is deleted when the archive is
garbage-collected or the interpreter exits. Before, the `mkdtemp` directory and the archived tool outputs survived
the process. Pass an explicit `directory` to keep them; an explicit directory is never deleted.

### Other behaviour changes

- Outside a container, state and configuration default to the platform locations below instead of the working directory.
- State files and lock files are created `0600` and directories `0700` (POSIX); see "Atomic writes and locks".
- `DuckDBStore` does not download `fts` unless asked, and its fallback is approximate (below).
- A single file passed as the root of a text store's `index()` is now indexed (it returned nothing before); NFD file
  names get NFC labels and chunk ids; files that share a name still get the same chunk id and one warning says so.
- Windows junctions and other directory reparse points are not entered by corpus scans and are refused by ingestion
  (this includes cloud-sync placeholder directories, if they carry the reparse attribute: not verified).

## Where state, configuration and caches live

`context_runtime/platform_services.py` resolves the locations. An environment variable always wins:

| purpose | variables | macOS | Windows | Linux |
|---|---|---|---|---|
| state | `CONTEXT_RUNTIME_HOME`, `AGENTIC_OS_HOME` | `~/Library/Application Support/context-runtime` | `%LOCALAPPDATA%\context-runtime` | `$XDG_DATA_HOME` or `~/.local/share/context-runtime` |
| config directory | `CONTEXT_RUNTIME_CONFIG_DIR` | `~/Library/Application Support/context-runtime` | `%LOCALAPPDATA%\context-runtime\config` | `$XDG_CONFIG_HOME` or `~/.config/context-runtime` |
| config file | `CONTEXT_RUNTIME_CONFIG`, `AGENTIC_OS_CONFIG` | `config.yaml` in the config directory | same | same |
| cache | `CONTEXT_RUNTIME_CACHE_DIR` | `~/Library/Caches/context-runtime` | `%LOCALAPPDATA%\context-runtime\cache` | `$XDG_CACHE_HOME` or `~/.cache/context-runtime` |

In a container the earlier layout is kept: working-directory `.context-runtime` and `config.yaml`,
`/data/context-runtime/policy`, `/data/market-radar/...`. The fixed `/data/...` locations apply **even when
`CONTEXT_RUNTIME_HOME` is set** (the compose files set it to `/data`): that variable never moved them. An item's own
variable (`POLICY_DIR`, `LEAD_PROFILE_DIR`, `PITCH_TEMPLATE_PATH`, `OUTREACH_LEDGER_PATH`, `OUTREACH_OUTBOX`) still wins.

A container is recognised by `CONTEXT_RUNTIME_CONTAINER=1` (`0` forces "not a container": this is the explicit
override when detection is wrong), `/.dockerenv`, `/run/.containerenv`, `KUBERNETES_SERVICE_HOST`, Amazon ECS / Fargate
(`ECS_CONTAINER_METADATA_URI`, `ECS_CONTAINER_METADATA_URI_V4`, `AWS_EXECUTION_ENV=AWS_ECS*`), Google Cloud Run
(`K_SERVICE`), or a container runtime named in `/proc/1/cgroup`.

Outside a container a non-empty `.context-runtime` (or a `config.yaml`) in the working directory is still used, even
when the platform location also exists; one warning names both paths and says how to migrate. Nothing is moved or merged
automatically, and a start never silently switches to an empty location.

## Atomic writes and locks

`atomic_write` writes a uniquely named temporary file in the target's directory (its name is shortened so it never
exceeds the file system's name limit), flushes and `fsync`s it, replaces the target with `os.replace` (on Windows,
retrying a bounded number of times when the target is briefly open elsewhere) and `fsync`s the directory on POSIX.
A symlink at the target is written through (the link survives); an existing file keeps its permissions and a
read-only target is refused as `Path.write_text` refused it. **Deliberate change:** new state files, lock files
and log files are `0600` and new directories `0700` on POSIX (before: the umask, typically `0644` / `0755`). Set
`CONTEXT_RUNTIME_SHARED=1` for a directory that other users or containers must read: files, lock files and
directories then get the umask defaults.

`file_lock` is an exclusive inter-process lock (`fcntl.flock` or `msvcrt.locking`) with a timeout; it is not
re-entrant. On a file system that cannot lock (`ENOLCK`, `ENOTSUP`, `EOPNOTSUPP`: some NFS, FUSE, 9p and SMB mounts),
`CONTEXT_RUNTIME_LOCKING=best-effort` logs **one** warning naming the path and falls back to a lock between the
threads of the process only (other processes are not excluded); `CONTEXT_RUNTIME_LOCKING=strict` raises instead.
Unset, the policy store (governance data) is strict and the other stores best-effort.

## DuckDB `fts` extension (explicit installation)

`DuckDBStore` runs BM25 inside DuckDB when the `fts` extension can be loaded. By default it **never downloads
anything**. If the extension cannot be loaded it logs one warning line
(`duckdb: the fts extension is not available (...); using the APPROXIMATE in-memory lexical fallback (not
score-compatible with the engine) ...`) and searches the same rows with the in-memory store's BM25
(`DuckDBStore.fts_available` is `False`). Recency queries are unaffected.

The fallback is **not** the engine's BM25 and not score-compatible with it. Measured differences, pinned by tests:
the engine drops English stop words and digit tokens and stems with Porter, the fallback does neither (`the` and
`404` match in the fallback and not in the engine; `errors` matches `error` in the engine and not in the fallback),
and the score scales differ. For queries without those cases the best-ranked chunk agreed on the small corpus tested;
do not compare scores across the two paths or apply one score threshold to both.

Order in which an extension is looked for:

1. A file: `DuckDBStore(fts_extension="/path/fts.duckdb_extension")` or `CONTEXT_RUNTIME_DUCKDB_FTS_EXTENSION`. If
   a file is configured and fails to load, nothing else is tried.
2. A directory: `CONTEXT_RUNTIME_DUCKDB_EXTENSION_DIR` points at a tree laid out like DuckDB's own
   (`<dir>/<duckdb version>/<platform>/fts.duckdb_extension`, for example `<dir>/v2.0.0/osx_arm64/...`,
   `windows_amd64`, `linux_amd64`).
3. An extension DuckDB already has in its default location (`~/.duckdb/extensions/...`).
4. A one-time download, **only** when asked for: `DuckDBStore(install_fts=True)` or
   `CONTEXT_RUNTIME_DUCKDB_INSTALL_FTS=1`.

Preparing an offline machine:

1. On a machine with network access, with the same DuckDB version and platform, run
   `python -c "import duckdb; duckdb.connect().execute('INSTALL fts')"`. The file lands in
   `~/.duckdb/extensions/<version>/<platform>/fts.duckdb_extension` (Windows: `%USERPROFILE%\.duckdb\...`).
2. Copy that file to the offline machine and set `CONTEXT_RUNTIME_DUCKDB_FTS_EXTENSION` to its path; or copy the
   `<version>/<platform>` directories under one folder and set `CONTEXT_RUNTIME_DUCKDB_EXTENSION_DIR` to that folder.
3. Check `DuckDBStore().fts_available`. The DuckDB version in the path must equal the installed `duckdb` package.

Signed or hardened bundles: the extension is a native library signed by DuckDB, not by this project. A bundle
that is notarised or run under code-integrity policies has to account for it (or rely on the in-memory fallback).

## Windows access control

On Windows `ensure_private_dir` and `atomic_write` set **no ACL**. A new directory or file inherits its parent's.
Under `%LOCALAPPDATA%` that is the user-profile ACL (the user, SYSTEM, Administrators). A location chosen with
`CONTEXT_RUNTIME_HOME` is as private as that location's ACL, no more. `tests/test_windows_acl.py` reads the
effective DACL on the Windows runner and asserts that the code adds no explicit entry and grants nothing to
Everyone, Users or Authenticated Users beyond the parent's grants; it also asserts that the default user-profile
location grants nothing to those groups. Mode bits (`0700`, `0600`) are POSIX only.

## What is verified, and where

Each item names the test; the pull request that introduced it lists the CI runs observed per operating system.

| item | test | runs on |
|---|---|---|
| Explicit UTF-8, no stray state in the working directory | `test_encoding_explicit.py`, `conftest.py` | Linux, Windows, macOS |
| Atomic write, retry, lock exclusion across processes | `test_platform_services.py` | Linux, Windows, macOS |
| Deterministic enumeration, byte-identical goldens | `test_corpus_files.py`, `tests/golden/` | Linux, Windows, macOS |
| Junction refusal (real junctions) | `test_corpus_files.py` | Windows only |
| Windows ACLs | `test_windows_acl.py` | Windows only |
| DuckDB fts unavailable paths, provisioning, parity | `test_duckdb_fts_offline.py` | Linux, Windows, macOS |
| Vector ranking parity (semantic with injected vectors, turbovec, a real ONNX model) | `test_vector_parity.py` | Linux, Windows, macOS where the optional packages install |
| Caddy demo proxy | `test_demo_proxy.py` | skipped, with the reason `UNVERIFIED on this platform: caddy unavailable`, where caddy is absent |

Every test run prints a skip summary at the end; skips whose reason starts with `UNVERIFIED` are behaviour that run
did not verify.
