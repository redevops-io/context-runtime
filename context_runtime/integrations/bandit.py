"""The shared learning core for Context Runtime app integrations (the fleet pattern).

Every tenant (sidekick, redevops-rag, …) makes a discrete choice keyed by intent
bucket and gets a measurable reward back. That is a contextual bandit. This module is
that bandit, generic over any *arm* object exposing a ``.key: str``. App-specific
arms and reward functions live in each tenant's module; the learning is shared here.

This is the v0.1-achievable stand-in for the v0.3 River contextual bandit — same
select/update/reward seam, so swapping in River later is a drop-in.
"""
from __future__ import annotations

import logging
from typing import Protocol

_log = logging.getLogger("context_runtime.learning")


class Arm(Protocol):
    @property
    def key(self) -> str: ...


class EpsilonGreedyBandit:
    """Contextual ε-greedy over arms, keyed by a context string (the intent bucket).

    Optimistic initialization makes every unseen arm look maximal, so each is tried at
    least once before the policy commits — cheap exploration without tuning a schedule.
    Deterministic xorshift rng (no global ``random``) keeps runs reproducible/testable.
    """

    def __init__(self, arms: tuple, epsilon: float = 0.15, optimistic: float = 1.0,
                 seed: int = 0x9E3779B9, persist_path: str | None = None, discount: float = 0.0,
                 ledger_max_bytes: int = 20_000_000):
        import threading
        self.arms = arms
        self.epsilon = epsilon
        self.optimistic = optimistic
        # discount ∈ (0,1] = constant step size (exponential recency weighting) so stale evidence
        # fades and the policy tracks a drifting best arm — the Whitepaper-v3 non-stationarity property.
        # 0 (default) = sample-average (1/n), the stationary estimator; behavior is unchanged.
        self.discount = discount
        self.stats: dict[str, dict[str, list[float]]] = {}   # ctx → arm.key → [n, mean]
        self._rng = seed & 0xFFFFFFFF
        self.persist_path = persist_path   # learned policy survives restarts if set
        # Guards stats + _rng. FastAPI runs sync endpoints in a threadpool (~40 threads)
        # over one shared bandit; without this, concurrent select/update on the shared
        # stats dict lose updates or raise "dict changed size during iteration".
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()   # serializes disk writes, off the stats lock
        self.frozen = False                  # this instance's kill switch; CR_LEARNING_FROZEN=1 stops every instance
        self.refused_while_frozen = 0
        self.ledger_max_bytes = ledger_max_bytes
        self.ledger_errors = 0
        self._ledger_lock = threading.Lock()
        self._seq = 0
        self._persist_blocked = False        # set when an unreadable policy file could not be moved aside: never overwrite it
        self._set_aside_reason = ""
        if persist_path:
            self._load()
            self._open_ledger()

    # ── the kill switch (learning and exploration stop; what was already learned still decides) ──
    @property
    def is_frozen(self) -> bool:
        import os
        return self.frozen or os.environ.get("CR_LEARNING_FROZEN", "").strip().lower() in ("1", "true", "yes", "on")

    def _rand(self) -> float:
        x = self._rng
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._rng = x & 0xFFFFFFFF
        return self._rng / 0x100000000

    def _ctx(self, ctx: str) -> dict[str, list[float]]:
        # backfill any arms added since this context was first seen / persisted, so select,
        # update, and value() never KeyError on a new arm against an old (persisted) context.
        d = self.stats.setdefault(ctx, {})
        for a in self.arms:
            d.setdefault(a.key, [0.0, self.optimistic])
        return d

    def select(self, ctx: str):
        frozen = self.is_frozen
        with self._lock:
            arms = self._ctx(ctx)
            if not frozen and self._rand() < self.epsilon:
                return self.arms[int(self._rand() * len(self.arms)) % len(self.arms)]
            # among the DECLARED arms only: a persisted file may still name an arm that has since been removed (or never
            # existed). Ranking it made select() raise as soon as its mean was the highest.
            declared = {a.key for a in self.arms}
            best = max((k for k in arms if k in declared), key=lambda k: arms[k][1])
            return next(a for a in self.arms if a.key == best)

    def update(self, ctx: str, arm, reward: float, provenance: dict | None = None) -> None:
        """Record one observation. ``provenance`` (source, run_id, principal, plan_hash), or the enclosing
        ``learning.ledger.provenance(...)`` block, says who or what produced the reward."""
        import json
        import math

        from ..learning.ledger import clean_provenance
        # One NaN or infinite reward used to make the arm's mean NaN for good (every later update keeps it NaN), the
        # snapshot written to disk carried it across restarts, and ``max`` over values that include NaN picks an arm by
        # dictionary order. A reward that is not a finite number is a caller's defect: refuse it, learn nothing.
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise ValueError("a bandit reward must be a finite number")
        who = clean_provenance(provenance)      # refused before anything is learned
        if self.is_frozen:
            self.refused_while_frozen += 1
            return
        with self._lock:
            arms = self._ctx(ctx)
            n, mean = arms[arm.key]
            n += 1
            alpha = self.discount if self.discount > 0.0 else 1.0 / n
            arms[arm.key] = [n, mean + alpha * (reward - mean)]
            self._seq += 1
            entry = {"seq": self._seq, "op": "update", "ctx": ctx, "arm": arm.key, "reward": float(reward), "discount": self.discount,
                     "optimistic": self.optimistic, "provenance": who}
            # serialize a consistent snapshot under the lock; write it to disk outside so
            # the fsync/rename never blocks concurrent select/update.
            snapshot = json.dumps(self.stats) if self.persist_path else None
        if snapshot is not None:
            self._append(entry)
            self._write_snapshot(snapshot)

    # ── the ledger: where each learned value came from ──
    @property
    def ledger_path(self) -> str | None:
        return f"{self.persist_path}.ledger.jsonl" if self.persist_path else None

    def _open_ledger(self) -> None:
        import json
        import os

        from ..learning.ledger import LedgerIncomplete, read
        try:
            entries = read(self.ledger_path)
        except (LedgerIncomplete, OSError):
            _log.warning("the learning ledger %s cannot be read; it is kept, and rollback will refuse until it is moved aside",
                         self.ledger_path)
            try:
                with open(self.ledger_path, "rb") as fh:
                    self._seq = sum(1 for _ in fh)
            except OSError:
                pass
            return
        self._seq = entries[-1]["seq"] if entries else 0
        if self._set_aside_reason and entries and not self._persist_blocked:
            self._seq += 1                 # the policy restarted blank: the history says so, or it would no longer reproduce it
            self._append({"seq": self._seq, "op": "reset", "ctx": None,
                          "provenance": {"source": "system", "principal": f"unreadable policy file ({self._set_aside_reason})"}})
        if not os.path.exists(self.ledger_path) and any(v[0] > 0 for arms in self.stats.values() for v in arms.values()):
            self._seq += 1                 # a policy learned before there was a ledger: that is where its history starts
            self._append({"seq": self._seq, "op": "baseline", "stats": json.loads(json.dumps(self.stats))})

    def _append(self, entry: dict) -> None:
        """Append one entry. A ledger that cannot be written never stops the serving path: it is counted and logged, and
        because the ledger then no longer reproduces the policy, rollback refuses."""
        import json
        import os
        import time
        entry = {"ts": round(time.time(), 3), **entry}
        try:
            with self._ledger_lock:
                path = self.ledger_path
                if os.path.exists(path) and os.path.getsize(path) > self.ledger_max_bytes:
                    os.replace(path, f"{path}.1")                  # bounded: history restarts from a baseline
                    # a NEW number, taken with the snapshot: every observation the snapshot contains has a lower one, so a
                    # replay (in seq order) lets the baseline replace them and never counts one twice
                    with self._lock:
                        base = json.loads(json.dumps(self.stats))
                        self._seq += 1
                        seq = self._seq
                    self._write_line(path, {"ts": entry["ts"], "seq": seq, "op": "baseline", "stats": base})
                    return                                          # the baseline already contains this observation
                self._write_line(path, entry)
        except OSError:
            self.ledger_errors += 1
            if self.ledger_errors == 1:
                _log.warning("the learning ledger %s cannot be written; learning continues, rollback will refuse", self.ledger_path)

    @staticmethod
    def _write_line(path: str, entry: dict) -> None:
        import json

        from ..platform_services import locked_append
        # one locked write, so lines from concurrent threads and processes are never lost or interleaved
        locked_append(path, json.dumps(entry, separators=(",", ":")) + "\n", mode=0o600)

    # ── undo: reset, and rollback of specific observations ──
    def reset(self, ctx: str | None = None, *, by: str) -> None:
        """Forget what was learned for one context, or for all of them. ``by`` names who asked."""
        import json
        if not by:
            raise ValueError("a reset names who asked for it")
        with self._lock:
            if ctx is None:
                self.stats = {}
            else:
                self.stats.pop(ctx, None)
            self._seq += 1
            entry = {"seq": self._seq, "op": "reset", "ctx": ctx, "provenance": {"source": "operator", "principal": str(by)[:200]}}
            snapshot = json.dumps(self.stats) if self.persist_path else None
        if snapshot is not None:
            self._append(entry)
            self._write_snapshot(snapshot)

    def rollback(self, exclude, *, by: str) -> int:
        """Take back the observations for which ``exclude(entry)`` is true and rebuild the policy without them.
        Returns how many were taken back. Raises ``LedgerIncomplete`` if the ledger does not reproduce the live policy."""
        import json

        from ..learning.ledger import LedgerIncomplete, read, rebuild, same_learning
        if not by:
            raise ValueError("a rollback names who asked for it")
        if not self.persist_path:
            raise LedgerIncomplete("a policy that is not persisted has no ledger")
        with self._ledger_lock, self._lock:
            entries = read(self.ledger_path)
            if not same_learning(rebuild(entries), self.stats):
                raise LedgerIncomplete("the ledger does not reproduce the live policy, so it cannot say what it would be without these observations")
            already = {s for e in entries if e["op"] == "rollback" for s in e.get("excluded", ())}
            chosen = [e["seq"] for e in entries if e["op"] == "update" and e["seq"] not in already and exclude(e)]
            if not chosen:
                return 0
            rebuilt = rebuild(entries, exclude=chosen)
            self._seq += 1
            self._write_line(self.ledger_path, {"seq": self._seq, "op": "rollback", "excluded": chosen,
                                                "provenance": {"source": "operator", "principal": str(by)[:200]}})
            self.stats = rebuilt
            snapshot = json.dumps(self.stats)
        self._write_snapshot(snapshot)
        return len(chosen)

    def status(self) -> dict:
        """What an operator needs to see about this learner."""
        with self._lock:
            observed = sum(int(v[0]) for arms in self.stats.values() for v in arms.values())
            return {"frozen": self.frozen or self.is_frozen, "refused_while_frozen": self.refused_while_frozen,
                    "contexts": len(self.stats), "observations": observed, "persisted": bool(self.persist_path),
                    "ledger": self.ledger_path, "ledger_errors": self.ledger_errors, "persist_blocked": self._persist_blocked}

    # ── persistence (so learning survives restarts) ──
    def _load(self) -> None:
        import json
        import os
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                import math

                def clean(v) -> list:
                    # a value that is not a finite number (written before non-finite rewards were refused) is forgotten:
                    # the arm starts again as unseen instead of staying NaN
                    # (the wrong SHAPE is something else: a damaged or foreign file, which is set aside below)
                    if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
                        raise ValueError("a policy entry is not [observations, mean]")
                    n, mean = v
                    if not all(math.isfinite(x) for x in (n, mean)):
                        _log.warning("a learned value in %s was not a finite number; that arm starts again as unseen", self.persist_path)
                        return [0.0, self.optimistic]
                    return [n, mean]
                loaded = {ctx: {k: clean(v) for k, v in arms.items()} for ctx, arms in raw.items()}
                with self._lock:
                    self.stats = loaded
            except Exception as exc:      # noqa: BLE001
                self._set_aside(type(exc).__name__)

    def _set_aside(self, why: str) -> None:
        """A policy file that cannot be read is KEPT under another name (it is the evidence, and may be repairable) and
        said so once. It used to be replaced by a blank policy with no trace."""
        import os
        import time
        self._set_aside_reason = why
        kept = f"{self.persist_path}.corrupt-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
        try:
            os.replace(self.persist_path, kept)
            _log.warning("the learned policy %s could not be read (%s); it was kept as %s and learning starts blank",
                         self.persist_path, why, kept)
        except OSError:
            self._persist_blocked = True
            _log.warning("the learned policy %s could not be read (%s) and could not be moved aside; it will NOT be overwritten",
                         self.persist_path, why)

    def save(self) -> None:
        import json
        with self._lock:
            snapshot = json.dumps(self.stats) if self.persist_path else None
        if snapshot is not None:
            self._write_snapshot(snapshot)

    def _write_snapshot(self, data: str) -> None:
        """Atomically write an already-serialized stats snapshot. Serialized by its own
        lock (not the stats lock) so persistence I/O stays off the learning hot path."""
        from ..platform_services import atomic_write
        if self._persist_blocked:
            return
        with self._save_lock:
            atomic_write(self.persist_path, data)

    def policy(self) -> dict[str, str]:
        """Current best arm key per context — the learned policy, for inspection."""
        with self._lock:
            return {ctx: max(a, key=lambda k: a[k][1]) for ctx, a in self.stats.items()}

    def value(self, ctx: str, arm_key: str) -> tuple[int, float]:
        with self._lock:
            a = self._ctx(ctx)[arm_key]
            return int(a[0]), a[1]
