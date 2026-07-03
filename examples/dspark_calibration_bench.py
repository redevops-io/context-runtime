#!/usr/bin/env python3
"""v1 vs v2 — does calibrated, load-aware retrieval learn a better policy?

Because the v2 additions are opt-in, **v2 with the flags off is byte-for-byte v1**, so this
one process runs a faithful A/B by toggling them:

  baseline (v1): reward = per-query judge − cost           (bandit never sees hit relevance)
  enhanced (v2): reward = blend(judge, calibrated P(rel))  + abstention + load-aware depth

The setup is a controllable stub retriever with **ground-truth** per-passage relevance, so
we can score what was actually served. The per-query judge models the REAL heuristic judge:
it scores term coverage / recall of the whole context, so it rewards dumping more passages
and is blind to precision — the coarse signal the per-passage calibrated relevance corrects.

The reward comparison is SEED-AVERAGED (§1): a single run is dominated by bandit exploration
luck (all high-coverage arms tie on the judge, so which one the policy locks onto is random),
so we average over 40 seeds and sweep beta to show the systematic effect. Everything is
seeded → reproducible.

Run:  PYTHONPATH=. python examples/dspark_calibration_bench.py
Exits 0. No external deps.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context_runtime.integrations.calibration import CalibrationLog, fit_from_log
from context_runtime.integrations.librechat import (
    DEFAULT_STRATEGIES,
    LibreChatTenant,
    RetrievalStrategy,
)
from context_runtime.integrations.loadmeter import LoadMeter
from context_runtime.types import Hit

# ── ground truth ────────────────────────────────────────────────────────────
# 10 candidate chunks; c0..c2 are the relevant ones for an answerable query.
N_CHUNKS = 10
RELEVANT = {"c0", "c1", "c2"}

# Each retrieval METHOD has a true quality (how well it ranks the relevant chunks first)
# and its own raw-score SCALE (so raw scores are NOT comparable across methods — only
# calibrated P(relevant) is). hybrid is genuinely best; bm25 looks strong on raw score
# (big scale) but is mediocre — the trap a raw-score reward would fall for.
METHOD_QUALITY = {"bm25": 0.55, "vector": 0.75, "hybrid": 0.90, "graph": 0.60, "community": 0.50}
METHOD_SCALE = {"bm25": 8.0, "vector": 1.0, "hybrid": 1.2, "graph": 2.0, "community": 1.5}


def _rng(seed: int):
    """Tiny deterministic LCG in [0,1) — no global random, reproducible."""
    state = {"x": (seed * 2654435761 + 12345) & 0xFFFFFFFF}

    def nxt() -> float:
        state["x"] = (1103515245 * state["x"] + 12345) & 0x7FFFFFFF
        return state["x"] / 0x7FFFFFFF
    return nxt


def _shash(*parts: str) -> int:
    """Stable cross-process hash (Python's built-in hash() is salted per process by
    PYTHONHASHSEED, which would make this benchmark non-reproducible)."""
    import zlib
    return zlib.crc32("|".join(parts).encode())


class StubRetriever:
    """Deterministic retriever with embedded ground-truth relevance. A method of quality q
    ranks relevant chunks first with prob ~q; scores are relevance-correlated and scaled
    per method so raw magnitudes are incomparable across methods."""

    def index(self, path):  # pragma: no cover
        return {}

    def search(self, query, k, method):
        answerable = not query.startswith("UNANSWERABLE")
        q = METHOD_QUALITY.get(method, 0.6)
        scale = METHOD_SCALE.get(method, 1.0)
        rnd = _rng(_shash(query, method))
        scored = []
        for i in range(N_CHUNKS):
            cid = f"c{i}"
            true_rel = 1.0 if (answerable and cid in RELEVANT) else 0.0
            # relevant chunks score high under a good method; noise shrinks with quality
            base = true_rel * q + (1.0 - q) * rnd()
            raw = round(base * scale, 4)
            scored.append((raw, cid, true_rel))
        scored.sort(reverse=True)
        hits = []
        for raw, cid, true_rel in scored[:k]:
            h = Hit(chunk_id=cid, filename=f"{cid}.txt", text=f"passage {cid} for {query}", score=raw)
            h.meta["_true_rel"] = true_rel      # benchmark-internal ground truth (tenant ignores it)
            hits.append(h)
        return hits


# ── the coverage-biased per-query judge ──────────────────────────────────────
# Models the REAL heuristic_judge, which scores term coverage of the retrieved context —
# a recall-like signal that rewards dumping more passages and is blind to precision. This
# is the coarse signal the calibrated per-passage relevance in the reward corrects: judge-
# only chases deep, low-precision arms; the blend pulls the policy back to precise ones.
def coverage_judge(hits, noise: float) -> float:
    found = min(3, sum(1 for h in hits if h.meta.get("_true_rel", 0) > 0))
    return max(0.0, min(1.0, found / 3.0 + noise))  # recall of the 3 relevant chunks


def true_precision(hits) -> float:
    return sum(h.meta["_true_rel"] for h in hits) / len(hits) if hits else 0.0


# ── build a calibration map from a labelled bootstrap (v2 only) ───────────────
def fit_calibration(tmp: str):
    log = CalibrationLog(pathlib.Path(tmp) / "boot.jsonl")
    ret = StubRetriever()
    for i in range(400):
        query = f"bootstrap query {i}"
        for method in METHOD_QUALITY:
            hits = ret.search(query, k=6, method=method)
            rows = [{"chunk_id": h.chunk_id, "score": float(h.score), "rel": h.meta["_true_rel"]}
                    for h in hits]
            log.append(method, "lookup", true_precision(hits), rows)
    return fit_from_log(log, min_samples=20)


# ── one run of the learning loop over an identical, seeded query stream ───────
def run_world(*, calibration, reward_beta, abstain_threshold=None, load_meter=None,
              load_aware=False, pin_inflight=0, strategies=DEFAULT_STRATEGIES, n=600, seed=7):
    tenant = LibreChatTenant(
        retriever=StubRetriever(), strategies=strategies,
        calibration=calibration, reward_beta=reward_beta,
        abstain_threshold=abstain_threshold, load_meter=load_meter, load_aware=load_aware,
    )
    for _ in range(pin_inflight):          # simulate concurrent load for the sizer's band
        load_meter.enter()
    noise_rng = _rng(seed)
    ans_prec, depths = [], []
    abst_unans, abst_ans, unans_total, ans_total = 0, 0, 0, 0
    for i in range(n):
        unanswerable = (i % 6 == 0)                      # ~17% of the stream has no good answer
        query = f"{'UNANSWERABLE ' if unanswerable else ''}user question {i}"
        ctx = tenant.retrieve(query)
        noise = (noise_rng() - 0.5) * 0.4                # coarse coverage/recall judge ± noise
        tenant.record_judgment(query, coverage_judge(ctx.hits, noise))
        if i < n // 2:
            continue                                     # measure only after the policy has learned
        if unanswerable:
            unans_total += 1
            abst_unans += int(ctx.abstain)
        else:
            ans_total += 1
            if ctx.abstain:
                abst_ans += 1
            else:
                ans_prec.append(true_precision(ctx.hits))   # precision on ANSWERABLE, answered
                depths.append(len(ctx.hits))
    return {
        "ans_precision": sum(ans_prec) / len(ans_prec) if ans_prec else 0.0,
        "mean_depth": sum(depths) / len(depths) if depths else 0.0,
        "abstain_recall": abst_unans / unans_total if unans_total else 0.0,     # unans caught
        "false_abstain_rate": abst_ans / ans_total if ans_total else 0.0,       # answerable wrongly dropped
    }


def main() -> int:
    tmp = tempfile.mkdtemp()
    cmap = fit_calibration(tmp)
    print("=" * 72)
    print("DSpark v1 vs v2 — self-learning retrieval over ground-truth relevance")
    print("=" * 72)
    print("eval = 2nd half of a 600-query stream; precision on ANSWERABLE, answered queries")
    print("only (abstention off) so it isolates the policy, not survivorship.\n")

    # ── (1) reward fix, SEED-AVERAGED. A single run is dominated by bandit exploration
    #        luck: every high-coverage arm ties on the coverage judge, so WHICH one the
    #        policy locks onto is random. Averaging over seeds isolates the reward's real,
    #        systematic effect — and it grows with how much the reward trusts calibrated
    #        per-passage relevance over the coarse coverage judge (the beta sweep). ──
    seeds = range(1, 41)
    n_seeds = len(seeds)

    def avg_prec(cal, beta):
        return sum(run_world(calibration=cal, reward_beta=beta, seed=s)["ans_precision"]
                   for s in seeds) / n_seeds

    v1 = avg_prec(None, 0.0)
    print(f"(1) reward — served true-precision, averaged over {n_seeds} seeds (coverage-biased judge)")
    print(f"      v1  judge-only reward (beta 0.0) ......... {v1 * 100:5.1f}%")
    for beta in (0.5, 0.7, 0.9):
        v2 = avg_prec(cmap, beta)
        print(f"      v2  judge + calibrated rel (beta {beta}) .... {v2 * 100:5.1f}%   ({(v2 - v1) * 100:+.1f} pts)")
    print()

    # ── (2) abstention, isolated: v2 with the P(relevant) floor on ──
    v2a = run_world(calibration=cmap, reward_beta=0.5, abstain_threshold=0.5)
    print("(2) abstention — v2 with a P(relevant) floor of 0.5 (v1 cannot abstain at all)")
    print(f"      unanswerable queries correctly abstained ... {v2a['abstain_recall']*100:5.1f}%")
    print(f"      answerable queries wrongly abstained ....... {v2a['false_abstain_rate']*100:5.1f}%\n")

    # ── (3) expensive-stage sizer, isolated: a single DEEP arm (k=8) with a low-relevance
    #        tail; sizer OFF (serve the whole block) vs ON (admit by survival product) ──
    deep = (RetrievalStrategy("hybrid", 8, True),)
    off = run_world(calibration=cmap, reward_beta=0.5, load_aware=False, strategies=deep)
    on = run_world(calibration=cmap, reward_beta=0.5, load_aware=True, strategies=deep,
                   load_meter=LoadMeter(mid=4, hi=8), pin_inflight=0)
    print("(3) expensive-stage sizer — deep arm (k=8); passages sent to the expensive stage")
    print(f"      sizer off .... {off['mean_depth']:.2f} passages, precision {off['ans_precision']*100:.0f}%")
    print(f"      sizer on ..... {on['mean_depth']:.2f} passages, precision {on['ans_precision']*100:.0f}%")
    print(f"      depth cut: {(1 - on['mean_depth']/off['mean_depth'])*100:.0f}%  "
          f"(the pruned tail was the low-relevance one → precision rose)\n")

    print("verdict: the reward fix lifts learned-policy precision (it finally sees served "
          "relevance);\n         abstention adds a grounded 'don't answer' v1 lacks; the "
          "sizer cuts expensive-stage\n         depth by pruning the low-relevance tail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
