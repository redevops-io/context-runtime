"""Render a Context Runtime EXPLAIN as EXPLAIN-ANALYZE-style text.

``LibreChatTenant.explain(request)`` returns the structured decision + retrieval trace; this turns
it into the developer-facing plaintext a database's ``EXPLAIN ANALYZE`` gives you — every candidate
arm with its learned value and quality/cost decomposition, the per-method retrieval trace with
calibrated P(relevant) (so a high-raw-score-but-irrelevant hit is obvious), what was served, the
abstention decision, and how the reward is computed. This is the observability layer that answers
"why did it retrieve *that*?" — the thing that turns a black-box argmax into an inspectable plan.

    from context_runtime.explain import render_explain
    print(render_explain(tenant.explain("what did revenue do after Q2")))

Pure formatting, no I/O, no deps — so it renders identically in a CLI, a test, or an endpoint.
"""
from __future__ import annotations

_BAR = "─" * 68


def _section(title: str) -> str:
    return f"─ {title} " + "─" * max(0, 66 - len(title))


def render_explain(exp: dict) -> str:
    out: list[str] = []
    qt = f"  query_type={exp['query_type']}" if exp.get("query_type") else ""
    out.append(f'EXPLAIN  "{exp["request"]}"')
    out.append(f"  intent={exp['intent_bucket']}{qt}  context={exp['context_key']}")
    out.append(_BAR)

    # ── decision ──
    out.append(_section("decision — candidate arms (learned)"))
    for c in exp["decision"]["candidates"]:
        mark = "►" if c["chosen"] else " "
        served = "  ✓ served" if c["chosen"] else ""
        b = c["bandit"]
        q = c.get("quality")
        qtxt = (f"   quality {q['quality']:.2f} / cost {q['cost']:.2f}" if q else
                f"   cost≈{c['cost_units']:.2f}")
        out.append(f"  {mark} {c['key']:<20} reward {b['value']:.3f} (n={b['n']}){qtxt}{served}")
        out.append(f"      └ {c['reason']}")

    # ── generation strategy (the answer plane) ──
    gen = exp.get("generation")
    if gen and gen.get("enabled"):
        out.append(_section(f"generation — strategy ladder for {gen['bucket']} (learned)"))
        for i, c in enumerate(gen["candidates"]):
            mark = "►" if c.get("entry_point") else " "
            b = c["bandit"]
            think = "think" if c["thinking"] else "no-think"
            entry = "  ← entry point" if c.get("entry_point") else ""
            out.append(f"  {mark} {c['strategy']:<12} {think} · {c['max_tokens']}t · cost≈{c['cost_units']:.1f}"
                       f"   reward {b['value']:.3f} (n={b['n']}){entry}")
    elif gen is not None:
        out.append(_section("generation"))
        out.append(f"  {gen.get('note', 'legacy single_shot')}")

    # ── retrieval trace ──
    out.append(_section("retrieval — every method, calibrated P(relevant)"))
    for method, rows in exp["retrieval"].items():
        if not rows:
            out.append(f"  {method:<10} (no hits)")
            continue
        for i, h in enumerate(rows):
            served = " ✓" if h.get("served") else "  "
            name = h.get("filename") or h.get("chunk_id") or "?"
            prel = h.get("p_rel")
            ptxt = f"  P(rel) {prel:.2f}" if prel is not None else ""
            flag = ""
            if prel is not None and float(h.get("score", 0)) > 1.5 and prel < 0.4:
                flag = "   ← high raw score, low calibrated relevance"
            head = f"{method}" if i == 0 else ""
            out.append(f"  {head:<10}{served} [{i+1}] {name:<28} score {float(h.get('score',0)):.2f}{ptxt}{flag}")

    # ── served + abstain ──
    out.append(_section("served"))
    s = exp["served"]
    mp = f" · max P(rel) {s['max_p_rel']:.2f}" if s.get("max_p_rel") is not None else ""
    ab = "ABSTAIN — " + s["abstain_reason"] if s.get("abstain") else "answered"
    out.append(f"  {s['n']} passage(s) via {s['method']}{mp} · {ab}")
    if s.get("citations"):
        out.append(f"  citations: {', '.join(s['citations'][:8])}")

    # ── reward provenance ──
    out.append(_section("reward"))
    r = exp["reward"]
    out.append(f"  {r['policy']}")
    bits = []
    if r.get("calibrated"):
        bits.append(f"calibrated relevance blended (β={r['reward_beta']})")
    if r.get("quality_routing"):
        bits.append("quality-routing ON (route by learned quality, not just cost)")
    if bits:
        out.append("  " + " · ".join(bits))
    out.append(f"  {r['note']}")
    return "\n".join(out)
