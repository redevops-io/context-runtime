# Native Windows and macOS: what changed and what is verified

This page describes the portability changes in the core library (state locations, atomic writes, file locks,
deterministic corpus enumeration, the DuckDB `fts` extension) and states, test by test, what is verified on
which operating system. It makes no claim of full parity: anything not listed as verified is not verified.

## Breaking changes

### Windows reserved device names are refused as corpus identifiers (all operating systems)

A corpus reference whose path segment is a Windows device name is refused on **every** operating system, so that an
identifier maps to the same, valid file name everywhere. The match is case-insensitive and applies with or without
an extension (`nul.txt` and `com1.tar.gz` are refused too).

Refused names: `con`, `prn`, `aux`, `nul`, `conin$`, `conout$`, `com1` to `com9`, `lpt1` to `lpt9` (the superscript
digits `com¹ com² com³ lpt¹ lpt² lpt³` as well).

Error: `POST /librechat/ingest` answers `400` with `{"detail": "invalid corpus reference"}` (the same answer as any
other invalid reference). Before, `con`, `nul` and the others were accepted on Linux and macOS.

Related: the policy store maps a scope name that is a device name to `<name>_` for its file name (`con` is stored
under `con_/app.jsonl`), and caps a file-name component at 128 characters (a short digest is appended when a name
is shortened). Rules that were saved under a scope named like a device name before this change are not found under
the new file name; rename the directory or file by hand if you have such a scope. Nothing is migrated automatically.

## Where state, configuration and caches live

`context_runtime/platform_services.py` resolves the locations. An environment variable always wins:

| purpose | variables | macOS | Windows | Linux |
|---|---|---|---|---|
| state | `CONTEXT_RUNTIME_HOME`, `AGENTIC_OS_HOME` | `~/Library/Application Support/context-runtime` | `%LOCALAPPDATA%\context-runtime` | `$XDG_DATA_HOME` or `~/.local/share/context-runtime` |
| config directory | `CONTEXT_RUNTIME_CONFIG_DIR` | `~/Library/Application Support/context-runtime` | `%LOCALAPPDATA%\context-runtime\config` | `$XDG_CONFIG_HOME` or `~/.config/context-runtime` |
| config file | `CONTEXT_RUNTIME_CONFIG`, `AGENTIC_OS_CONFIG` | `config.yaml` in the config directory | same | same |
| cache | `CONTEXT_RUNTIME_CACHE_DIR` | `~/Library/Caches/context-runtime` | `%LOCALAPPDATA%\context-runtime\cache` | `$XDG_CACHE_HOME` or `~/.cache/context-runtime` |

In a container (`/.dockerenv`, `/run/.containerenv`, a Kubernetes pod, or `CONTEXT_RUNTIME_CONTAINER=1`) the earlier
layout is kept (working-directory `.context-runtime` and `config.yaml`, `/data/context-runtime/policy`,
`/data/market-radar/...`), each still overridable by its own variable. Outside a container, an existing
working-directory `.context-runtime` or `config.yaml` is still used, with a log line, until the platform location
exists. Directories created by `ensure_private_dir` are `0700` on POSIX.

## Atomic writes and locks

`atomic_write` writes a uniquely named temporary file in the target's directory, flushes and `fsync`s it, replaces
the target with `os.replace` (on Windows, retrying a bounded number of times when the target is briefly open
elsewhere) and `fsync`s the directory on POSIX. `file_lock` is an exclusive inter-process lock (`fcntl.flock` or
`msvcrt.locking`) with a timeout; it is not re-entrant.

## DuckDB `fts` extension (explicit installation)

`DuckDBStore` runs BM25 inside DuckDB when the `fts` extension can be loaded. By default it **never downloads
anything**. If the extension cannot be loaded it logs one warning line
(`duckdb: the fts extension is not available (...); using the in-memory BM25 path ...`) and serves BM25 from the same rows
with the in-memory implementation (`DuckDBStore.fts_available` is `False`). Recency queries are unaffected. Scores
of the two implementations differ (different tokenisers and stemming); the top-ranked chunks agree on unambiguous
queries (tested).

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
