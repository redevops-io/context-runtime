"""Build the real-data inputs for the accelerator arms (run on the dev box, which has the corpus + a real
encoder), then ship the resulting ``accel_inputs.npz`` to the GPU host for the crossover sweep.

  Arm H  : the full evidence table — (ref=page_id, revid, content_hash, ts_unix) for every content-namespace
           revision in the frozen strategywiki dump (~10^5 rows). No embedding needed.
  Arm I/J: real bge-small-en-v1.5 (384-d) embeddings of real revision text, L2-normalised, plus per-item
           token counts (from real text length) and a relevance score (cosine to a query) for the knapsack.

Everything is deterministic (ordered scan of the frozen dump, fixed model, fixed query) so the sweep on the
GPU host reproduces. Usage:

    PYTHONPATH=. python -m accelerators.prepare_inputs --embed 12000 --queries 300 --out /tmp/accel_inputs.npz
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from harness.corpus import iter_pages
from harness.evidence_corpus import HISTORY, CONTENT_NS

OUT_DEFAULT = Path("/tmp/accel_inputs.npz")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"      # 384-d, ONNX via fastembed (CPU, no torch)
QUERY = "strategy guide walkthrough for the game level and its objectives"   # fixed arm-J relevance query


def _ts_unix(iso: str) -> int:
    try:
        return int(datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return 0


def _sha_to_int(sha1: str) -> int:
    """Stable 63-bit int key for a MediaWiki base-36 content digest (content identity for groupby)."""
    if not sha1:
        return 0
    return int(hashlib.blake2b(sha1.encode(), digest_size=8).hexdigest(), 16) & ((1 << 63) - 1)


def build(embed_n: int, n_queries: int):
    h_ref, h_revid, h_sha, h_ts = [], [], [], []
    texts: list[str] = []                    # real revision texts to embed (deduped by content hash)
    seen_sha: set[str] = set()

    for page in iter_pages(HISTORY, include_text=True):
        if page.ns not in CONTENT_NS:
            continue
        for r in page.revisions:
            h_ref.append(page.page_id)
            h_revid.append(int(r.rev_id))
            h_sha.append(_sha_to_int(r.sha1))
            h_ts.append(_ts_unix(r.timestamp))
            if len(texts) < embed_n and r.text and 200 <= len(r.text) <= 8000 and r.sha1 not in seen_sha:
                seen_sha.add(r.sha1)
                texts.append(r.text)

    print(f"evidence table: {len(h_ref)} revisions across content namespaces")
    print(f"embedding corpus: {len(texts)} distinct-content revision texts")

    from fastembed import TextEmbedding
    model = TextEmbedding(EMBED_MODEL)
    emb = np.array(list(model.embed(texts)), dtype=np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)      # cosine == inner product
    q_emb = np.array(list(model.embed([QUERY])), dtype=np.float32)
    q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    # arm I queries: a deterministic held-out slice of the corpus (self appears in both backends, so it
    # does not bias GPU-vs-exact recall); arm J relevance: cosine of each item to the fixed topic query.
    queries = emb[:n_queries].copy()
    tokens = np.array([max(1, len(t) // 4) for t in texts], dtype=np.int32)   # ~4 chars/token
    relevance = ((emb @ q_emb[0]) + 1.0) * 50.0                                # -> [0,100], float32
    order = np.argsort(-relevance)                                             # most-relevant first
    return {
        "h_ref": np.array(h_ref, dtype=np.int32), "h_revid": np.array(h_revid, dtype=np.int64),
        "h_sha": np.array(h_sha, dtype=np.int64), "h_ts": np.array(h_ts, dtype=np.int64),
        "emb": emb, "q_emb": queries,
        "j_value": relevance[order].astype(np.float32), "j_tokens": tokens[order],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=12000, help="max revision texts to embed (arm I/J corpus)")
    ap.add_argument("--queries", type=int, default=300, help="arm-I query count")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    data = build(args.embed, args.queries)
    np.savez_compressed(args.out, **data)
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({mb:.1f} MB): "
          f"H={len(data['h_ref'])} rows, I/J corpus={data['emb'].shape}, queries={data['q_emb'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
