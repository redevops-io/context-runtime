"""The SimGraph-vs-HippoRAG graph-compare harness — wiring + recall computation (SimGraph path;
HippoRAG needs the heavy engine + a served LLM, exercised in the real run)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import graph_compare as gc  # noqa: E402


def test_simgraph_multihop_recall_on_demo():
    it = gc.DEMO[0]
    # both supporting passages recovered: one direct (hop-0), one bridge-only (multi-hop)
    assert gc.recall_at_k(gc.build_simgraph(it["texts"]), it, k=4) == 1.0


def test_recall_is_partial_when_bridge_missed_at_low_k():
    it = gc.DEMO[0]
    # at k=1 only the direct hop-0 passage is returned → half the supporting set
    assert gc.recall_at_k(gc.build_simgraph(it["texts"]), it, k=1) == 0.5


def test_musique_loader(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({
        "question": "who?",
        "paragraphs": [
            {"paragraph_text": "gold one", "is_supporting": True},
            {"paragraph_text": "distractor", "is_supporting": False},
            {"paragraph_text": "gold two", "is_supporting": True},
        ],
    }) + "\n" + json.dumps({"question": "", "paragraphs": []}) + "\n")  # malformed row is skipped
    items = gc.load_musique(str(p), limit=0)
    assert len(items) == 1
    assert items[0]["supporting"] == {0, 2}
    assert items[0]["texts"][0] == "gold one"
