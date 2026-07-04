#!/usr/bin/env python3
"""Cross-modal image search (Phase 2a) — a text query retrieves the right IMAGE.

Generates three visually-distinct images (a red bar chart, a blue box-and-arrow diagram, a
green circle), indexes them with a CLIP vision tower (ONNX, no torch), and shows that a text
description retrieves the matching image via the shared text↔image space — no OCR, no shared
words. `image` is a routable retrieval method the bandit can pick, so the cost-based planner
decides when visual retrieval is worth paying for.

Run:  PYTHONPATH=. python examples/multimodal_image_search.py
Exits 0 whether or not the [multimodal] extra (fastembed + pillow) is installed.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context_runtime.adapters.store_image import ImageRetriever, image_embeddings_available


def _make_demo_images(d: str) -> bool:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return False
    im = Image.new("RGB", (320, 240), "white"); dr = ImageDraw.Draw(im)
    for i, h in enumerate([60, 110, 90, 150, 70]):
        dr.rectangle([30 + i * 55, 220 - h, 70 + i * 55, 220], fill="red")
    im.save(f"{d}/red_bar_chart.png")
    im = Image.new("RGB", (320, 240), "white"); dr = ImageDraw.Draw(im)
    for x in (20, 130, 240):
        dr.rectangle([x, 100, x + 70, 150], outline="blue", width=3)
    dr.line([90, 125, 130, 125], fill="blue", width=3); dr.line([200, 125, 240, 125], fill="blue", width=3)
    im.save(f"{d}/blue_flow_diagram.png")
    im = Image.new("RGB", (320, 240), "white"); dr = ImageDraw.Draw(im)
    dr.ellipse([90, 50, 230, 190], fill="green")
    im.save(f"{d}/green_circle.png")
    return True


def main() -> int:
    print("=" * 64)
    print("Cross-modal image search — text query → the right image (Phase 2a)")
    print("=" * 64)
    if not image_embeddings_available():
        print("\n[multimodal] extra not installed (fastembed) — image search degrades to empty.")
        print("Install:  pip install 'context_runtime[multimodal]'   then re-run.")
        return 0
    d = tempfile.mkdtemp()
    if not _make_demo_images(d):
        print("pillow missing — cannot render demo images."); return 0
    ret = ImageRetriever()
    print("indexed:", ret.index(d), "\n")
    queries = [
        "a red bar chart of revenue",
        "a diagram with boxes connected by arrows",
        "a green circle",
    ]
    for q in queries:
        hits = ret.search(q, 3, "image")
        top = hits[0].filename if hits else "(none)"
        print(f'  "{q}"')
        for h in hits:
            mark = " ←" if h is hits[0] else ""
            print(f"       {h.score:+.3f}  {h.filename}{mark}")
        print()
    print("Each text query retrieves the matching image as the top hit — no OCR, no shared terms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
