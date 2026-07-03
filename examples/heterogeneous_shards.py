"""Heterogeneous sharding — a personal mix of FINANCIAL + MEDICAL records.

A real user's local files are a mix of very different data. Dump them into ONE flat
index and two things break:

  1. VOLUME ASYMMETRY — 5000 pages of 10-K filings drown ~15 medical notes. A medical
     query competes against thousands of financial chunks for a top-k slot, and the
     minority domain gets buried.
  2. VOCAB COLLISION — "discharge" (hospital vs debt), "statement" (patient vs financial),
     "balance" (fluid vs sheet), "chronic"/"acute" (condition vs distress), "liability".
     In a mixed index those terms' statistics blur across domains, so a medical
     "discharge summary" query can surface a 10-K's "discharge of liability" instead.

Sharding by source (one index per domain) + reciprocal-rank fusion fixes both: each
shard contributes its own top-k regardless of the other's size, and RRF combines by
RANK (not raw BM25 score), so a verbose 10-K page can't swamp a short clinical note.

This drives the point with REAL data (FinanceBench 10-K pages) + a small curated medical
corpus built to collide on vocabulary. It compares a naive single mixed index against a
ShardedRetriever, measuring whether the correct MEDICAL note is recalled.

    python examples/heterogeneous_shards.py [--financial-cap 1500]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.scheduler.parallel_fusion import ShardedRetriever

FINANCEBENCH_CORPUS = Path(__file__).resolve().parent.parent / ".financebench" / "corpus"


def make_store(docs: list[dict], name: str, backend: str):
    """Build a store on the chosen backend. The flat-mixed-index vs coverage-routed result
    holds on all three — in-memory (default), DuckDB fts, or Postgres tsvector."""
    if backend == "memory":
        return InMemoryStore(docs, source=name)
    if backend == "duckdb":
        from context_runtime.adapters.store_duckdb import DuckDBStore
        return DuckDBStore(docs, path=f"/tmp/cr_{name}.duckdb", source=name)
    if backend == "postgres":
        from context_runtime.adapters.store_postgres import PostgresStore
        dsn = os.environ.get("CR_PG_DSN")
        if not dsn:
            raise SystemExit("--backend postgres needs CR_PG_DSN "
                             "(e.g. postgresql://postgres:pw@127.0.0.1:5432/db)")
        return PostgresStore(docs, dsn=dsn, table=f"cr_{name}", source=name)
    raise SystemExit(f"unknown backend {backend!r}")

# ── the minority domain: curated clinical notes that deliberately collide on vocab ──
# Each note's id is the ground-truth answer for the probe that targets it.
MEDICAL_RECORDS = [
    ("med_discharge", "DISCHARGE SUMMARY: 68-year-old male admitted for community-acquired "
     "pneumonia. Treated with IV ceftriaxone. Condition improved; discharged home on oral "
     "antibiotics with follow-up in one week. Discharge medications reconciled."),
    ("med_cardiac_stress", "CARDIOLOGY: Patient underwent an exercise stress test on the "
     "treadmill. ST-segment depression noted at peak. Recommend coronary angiography. "
     "History of hypertension and hyperlipidemia."),
    ("med_arrest", "CODE NOTE: Patient suffered cardiac arrest on the floor. ACLS initiated, "
     "return of spontaneous circulation after two rounds of epinephrine. Transferred to ICU."),
    ("med_diabetes", "ENDOCRINOLOGY: Type 2 diabetes mellitus, chronic and poorly controlled. "
     "HbA1c 9.2 percent. Started on metformin and basal insulin. Diabetic foot exam normal."),
    ("med_fluid_balance", "NURSING NOTE: Strict intake and output ordered. Fluid balance is "
     "positive by 1.2 liters over 24 hours. Mild peripheral edema. Diuretic administration held."),
    ("med_med_admin", "MEDICATION ADMINISTRATION RECORD: Morphine 2 mg IV given for acute pain. "
     "Patient tolerated administration without respiratory depression. Reassessed in 30 minutes."),
    ("med_appendicitis", "SURGERY: Acute appendicitis confirmed on CT. Laparoscopic "
     "appendectomy performed without complication. Postoperative course uneventful."),
    ("med_statement", "PATIENT STATEMENT: The patient states the chest pain began this morning, "
     "radiating to the left arm, associated with shortness of breath and diaphoresis."),
    ("med_liability_consent", "INFORMED CONSENT: Risks, benefits, and alternatives discussed. "
     "Patient acknowledges the procedure carries a risk of bleeding and infection and consents."),
    ("med_renal", "NEPHROLOGY: Chronic kidney disease stage 3. Creatinine 1.9. Avoid nephrotoxic "
     "agents. Renal ultrasound shows no obstruction. Continue ACE inhibitor at reduced dose."),
    ("med_psych", "PSYCHIATRY: Patient reports acute stress reaction following a family loss. "
     "Denies suicidal ideation. Sleep is poor. Started on short course of a sleep aid."),
    ("med_ortho", "ORTHOPEDICS: Left femur fracture after a fall. Balance and gait assessment "
     "shows instability. Open reduction internal fixation scheduled. Weight-bearing restricted."),
    ("med_labs", "LABORATORY: Complete blood count shows hemoglobin 8.1, indicating anemia. "
     "Iron studies pending. Ferritin low. Consider GI workup for chronic blood loss."),
    ("med_respiratory", "PULMONOLOGY: Chronic obstructive pulmonary disease exacerbation. "
     "Wheezing on exam. Nebulized bronchodilators and steroids administered. Oxygen at 2 liters."),
    ("med_maternity", "OBSTETRICS: 32-week gestation, routine prenatal visit. Fundal height "
     "appropriate. Fetal heart tones 140. Glucose tolerance test scheduled for next visit."),
    ("med_derm", "DERMATOLOGY: Suspicious pigmented lesion on the back, asymmetric with irregular "
     "borders. Excisional biopsy performed. Awaiting pathology to rule out melanoma."),
]

# Probes: (query, ground-truth medical id, uses-collision-term?). Each targets a medical note
# whose vocabulary overlaps the FINANCIAL corpus, so a mixed index tends to bury it.
MEDICAL_PROBES = [
    ("hospital discharge summary medications on discharge home", "med_discharge", True),
    ("cardiac stress test treadmill ST depression", "med_cardiac_stress", True),
    ("patient statement chest pain radiating to the arm", "med_statement", True),
    ("fluid balance intake and output positive edema", "med_fluid_balance", True),
    ("medication administration morphine for acute pain", "med_med_admin", True),
    ("chronic kidney disease creatinine avoid nephrotoxic", "med_renal", True),
    ("acute appendicitis laparoscopic appendectomy", "med_appendicitis", False),
    ("balance and gait instability after a fall fracture", "med_ortho", True),
]

# A few financial probes (from real FinanceBench-style questions) to confirm sharding
# doesn't hurt the majority domain — the right answer term should still surface.
FINANCIAL_PROBES = [
    ("capital expenditure amount for the fiscal year", "capital expenditures"),
    ("total revenue net sales for the year", "net sales"),
    ("long term debt and financial liabilities on the balance sheet", "long-term debt"),
]


def load_financial_docs(cap: int) -> list[dict]:
    files = sorted(FINANCEBENCH_CORPUS.glob("*.txt"))[:cap]
    docs = []
    for fp in files:
        docs.append({"chunk_id": fp.stem, "filename": fp.name,
                     "text": fp.read_text(errors="ignore"), "created_at": None})
    return docs


def medical_docs() -> list[dict]:
    return [{"chunk_id": cid, "filename": "medical_records", "text": txt, "created_at": None}
            for cid, txt in MEDICAL_RECORDS]


def domain(hit) -> str:
    return "med" if hit.chunk_id.startswith("med_") else "fin"


def recall_at_k(hits, needed_id: str) -> bool:
    return any(h.chunk_id == needed_id for h in hits)


def noise(hits, wanted: str) -> int:
    """# of wrong-domain docs in the top-k (context pollution)."""
    return sum(1 for h in hits if domain(h) != wanted)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--financial-cap", type=int, default=3000,
                    help="how many FinanceBench pages to load (the majority domain)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--backend", choices=["memory", "duckdb", "postgres"], default="memory",
                    help="store backend for all shards (postgres needs CR_PG_DSN)")
    args = ap.parse_args()

    if not FINANCEBENCH_CORPUS.exists():
        raise SystemExit(f"FinanceBench corpus not found at {FINANCEBENCH_CORPUS} "
                         "(run deploy/financebench/build_corpus.py first)")

    fin = load_financial_docs(args.financial_cap)
    med = medical_docs()
    k = args.k
    print(f"corpus: {len(fin)} financial 10-K pages + {len(med)} medical notes  "
          f"(k={k})\n")

    # three retrieval strategies over the SAME heterogeneous data (on the chosen backend):
    print(f"backend: {args.backend}\n")
    mixed = make_store(fin + med, "mixed", args.backend)          # one flat index
    shards = [make_store(fin, "financial", args.backend), make_store(med, "medical", args.backend)]
    fused = ShardedRetriever(shards, engine="auto", router="fuse")      # fan-out + RRF
    routed = ShardedRetriever(shards, engine="auto", router="coverage")  # fan-out + route-by-coverage

    strategies = [("mixed flat index", mixed), ("sharded + RRF fuse", fused),
                  ("sharded + routed", routed)]

    # ── MEDICAL queries: is the right note recalled, and how much financial noise rides along? ──
    print(f"MEDICAL probes  (recall of the right note  |  cross-domain noise in top-{k})")
    print(f"  {'strategy':22s}  recall   avg-noise")
    for label, r in strategies:
        rec = sum(recall_at_k(r.search(q, k, "hybrid"), nid) for q, nid, _ in MEDICAL_PROBES)
        noi = sum(noise(r.search(q, k, "hybrid"), "med") for q, nid, _ in MEDICAL_PROBES)
        print(f"  {label:22s}  {rec}/{len(MEDICAL_PROBES)}      {noi / len(MEDICAL_PROBES):.2f} fin-docs/query")

    print(f"\n  per-probe top-{k} domains (M=mixed, F=fuse, R=routed):")
    for q, nid, collides in MEDICAL_PROBES:
        def dstr(r):
            return "".join("M" if domain(h) == "med" else "·" for h in r.search(q, k, "hybrid"))
        tag = " [collision]" if collides else ""
        print(f"    M:{dstr(mixed):5s} F:{dstr(fused):5s} R:{dstr(routed):5s}  {q}{tag}")

    # ── FINANCIAL queries: routing should send them the other way (no medical noise) ──
    print(f"\nFINANCIAL probes  (answer term present  |  medical noise in top-{k})")
    for label, r in [("sharded + RRF fuse", fused), ("sharded + routed", routed)]:
        got = sum(any(term.split()[0] in h.text.lower() for h in r.search(q, k, "hybrid"))
                  for q, term in FINANCIAL_PROBES)
        med_noise = sum(noise(r.search(q, k, "hybrid"), "fin") for q, term in FINANCIAL_PROBES)
        print(f"  {label:22s}  {got}/{len(FINANCIAL_PROBES)}      {med_noise} med-docs total")

    # verdict + guarantees
    fuse_noise = sum(noise(fused.search(q, k, "hybrid"), "med") for q, _, _ in MEDICAL_PROBES)
    route_noise = sum(noise(routed.search(q, k, "hybrid"), "med") for q, _, _ in MEDICAL_PROBES)
    route_rec = sum(recall_at_k(routed.search(q, k, "hybrid"), nid) for q, nid, _ in MEDICAL_PROBES)
    print(f"\nverdict: coverage routing cut cross-domain noise {fuse_noise} -> {route_noise} docs "
          f"across {len(MEDICAL_PROBES)} medical queries while keeping recall {route_rec}/{len(MEDICAL_PROBES)}.")
    print("(the routing threshold is what a Context Runtime bandit learns — see integrations/chat_memory.py)")
    assert route_noise <= fuse_noise, "routing should not increase cross-domain noise"
    assert route_rec == len(MEDICAL_PROBES), "routing must preserve recall"


if __name__ == "__main__":
    main()
