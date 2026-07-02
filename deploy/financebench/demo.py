"""FinanceBench × LibreQB demo: for each hard question, show what EACH retrieval method
surfaces (the Query Board) vs. the served strategy — proving no single method suffices."""
import json, sys, urllib.request

CP = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8092"
QUERIES = [
    # (label, question, ground-truth answer, evidence page)
    ("metrics/table", "What is the FY2022 unadjusted EBITDA less capex for PepsiCo?", "$9068M", 61),
    ("metrics/table", "What is PepsiCo's FY2022 unadjusted EBITDA % margin?", "16.5%", 61),
    ("novel/multi-hop", "Excluding M&A, which segment dragged down 3M's overall growth in 2022?", "Consumer, -0.9% organic", 24),
    ("domain/reasoning", "Is 3M a capital-intensive business based on FY2022 data?", "No (efficient capex/fixed assets)", 47),
    ("domain/reasoning", "What drove 3M's operating margin change in FY2022?", "Down 1.7%, mostly gross margin", 26),
]

def post(path, body):
    r = urllib.request.Request(CP + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())

def base(cid):
    return cid.split("/")[-1][:26]

for label, q, ans, pg in QUERIES:
    print("\n" + "=" * 100)
    print(f"[{label}]  {q}")
    print(f"   ground truth: {ans}  (evidence pg {pg})")
    d = post("/librechat/compare", {"request": q, "k": 3})
    for m in ["bm25", "vector", "hybrid", "community", "graph"]:
        hits = d["methods"].get(m, [])
        tops = " · ".join(f"{base(h['chunk_id'])}({h['score']:.2f})" for h in hits[:3])
        mark = " ◀ SERVED" if m == d["chosen"]["method"] else ""
        print(f"   {m:9} {tops}{mark}")
    print(f"   served: {d['chosen']['key']}  ({len(d['served']['citations'])} citations)")
