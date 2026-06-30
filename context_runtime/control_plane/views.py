"""HTML views for the control plane chrome: the per-module nav shell and the
overview / "how it works" page.

Kept out of control_plane.py so the big HTML strings don't bury the routes.
These reuse the same MD3 dark theme as the main dashboard (DASHBOARD_HTML in
control_plane.py); the shared design tokens live in ``_BASE_CSS`` below.

Why a shell: each module serves its OWN self-contained dashboard (inline CSS,
no shared chrome). Proxied raw, a module page is a dead end — no way back to the
control plane, no way to jump to a sibling module, no sense of where you are.
``module_shell`` wraps the proxied page (rendered in a same-origin iframe at
``/m/<name>/raw``) in a persistent top bar: back-to-OS, a breadcrumb, a module
switcher, a live health dot, and the source link.
"""
from __future__ import annotations

import html
import json

# Shared design tokens + primitives (a trimmed subset of the dashboard's CSS).
_BASE_CSS = """
  :root{
    --surface:#131316; --surface-container-low:#1b1b1f; --surface-container:#1f1f23;
    --surface-container-high:#2a2a2e; --surface-container-highest:#353539;
    --on-surface:#e4e2e6; --on-surface-variant:#c7c5ca; --on-surface-muted:#918f96;
    --outline-variant:#2f2f33;
    --primary:#4fd1c5; --on-primary:#00201c; --primary-container:#00504a; --on-primary-container:#a8f0e6;
    --secondary:#f5b544; --success:#5bd98a; --success-container:#0f3d22;
    --danger:#f2544f; --danger-container:#5c1512; --info:#5aa9f0; --info-container:#103a5c;
    --sp-2:8px;--sp-3:12px;--sp-4:16px;--sp-5:24px;--sp-6:32px;
    --radius-sm:8px;--radius-md:12px;--radius-lg:16px;--radius-pill:999px;
    --font-sans:"Roboto",system-ui,-apple-system,"Segoe UI",sans-serif;
    --font-mono:"Roboto Mono",ui-monospace,"SF Mono",monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface);color:var(--on-surface);font-family:var(--font-sans);line-height:1.45}
  a{color:var(--primary);text-decoration:none}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--on-surface-muted);flex:none;display:inline-block}
  .dot.up{background:var(--success);box-shadow:0 0 8px rgba(91,217,138,.7)}
  .dot.down{background:var(--danger)} .dot.na{background:var(--on-surface-muted)}
"""


def module_shell(*, name: str, group: str, repo: str, switcher: list[dict]) -> str:
    """Wrap a module's proxied dashboard in persistent nav chrome.

    ``switcher`` is the list of modules that have a live dashboard, each
    ``{"name","group"}`` — used to populate the jump-to-module dropdown.
    """
    # Build the <optgroup>-grouped switcher server-side so it works without JS.
    by_group: dict[str, list[str]] = {}
    for m in switcher:
        by_group.setdefault(m["group"], []).append(m["name"])
    opts = []
    for g, members in by_group.items():
        opts.append(f'<optgroup label="{html.escape(g)}">')
        for n in members:
            sel = " selected" if n == name else ""
            label = n.replace("agentic-", "").replace("-", " ").title()
            opts.append(f'<option value="{html.escape(n)}"{sel}>{html.escape(label)}</option>')
        opts.append("</optgroup>")
    switcher_html = "".join(opts)
    src_url = "https://github.com/" + html.escape(repo)
    crumb = html.escape(name.replace("agentic-", "").replace("-", " ").title())

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} · Agentic OS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&family=Roboto+Mono:wght@400;500&display=swap">
<style>{_BASE_CSS}
  body{{display:flex;flex-direction:column;height:100vh;overflow:hidden}}
  .navbar{{display:flex;align-items:center;gap:var(--sp-4);flex-wrap:wrap;
    padding:var(--sp-3) var(--sp-5);background:var(--surface-container-low);
    border-bottom:1px solid var(--outline-variant);flex:none}}
  .back{{display:inline-flex;align-items:center;gap:6px;font:600 15px/1 var(--font-sans);
    color:var(--on-primary);background:var(--primary);padding:11px 20px;border-radius:var(--radius-pill);box-shadow:var(--shadow-1)}}
  .back:hover{{background:var(--primary-container);color:var(--on-primary-container)}}
  .iconbtn{{background:var(--surface-container-high);color:var(--on-surface-variant);border:1px solid var(--outline-variant);
    border-radius:var(--radius-sm);padding:6px 10px;font:500 14px/1 var(--font-sans);cursor:pointer}}
  .iconbtn:hover{{color:var(--primary);border-color:var(--primary)}}
  .crumbs{{font:400 13px/18px var(--font-sans);color:var(--on-surface-muted)}}
  .crumbs a{{color:var(--on-surface-variant)}} .crumbs b{{color:var(--on-surface)}}
  .spacer{{flex:1}}
  .switch{{display:flex;align-items:center;gap:var(--sp-2);font:400 13px/1 var(--font-sans);color:var(--on-surface-muted)}}
  select{{background:var(--surface-container-high);color:var(--on-surface);border:1px solid var(--outline-variant);
    border-radius:var(--radius-sm);padding:7px 10px;font:500 13px/1 var(--font-sans)}}
  .health{{display:inline-flex;align-items:center;gap:6px;font:400 13px/1 var(--font-mono);color:var(--on-surface-muted)}}
  .src{{font:400 13px/1 var(--font-sans);color:var(--on-surface-muted)}} .src:hover{{color:var(--primary)}}
  .demo-banner{{display:flex;align-items:center;gap:var(--sp-3);flex:none;
    padding:7px var(--sp-5);background:var(--info-container);color:var(--info);
    font:400 13px/18px var(--font-sans);border-bottom:1px solid var(--outline-variant)}}
  .demo-banner b{{color:var(--on-surface)}}
  .demo-banner button{{margin-left:auto;background:none;border:1px solid currentColor;color:inherit;
    border-radius:var(--radius-pill);padding:2px 10px;font:500 12px/1 var(--font-sans);cursor:pointer}}
  .frame-wrap{{position:relative;flex:1;display:flex}}
  .loading{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;gap:var(--sp-3);
    background:var(--surface);color:var(--on-surface-muted);font:400 14px/1 var(--font-sans);z-index:1}}
  .spinner{{width:18px;height:18px;border:2px solid var(--outline-variant);border-top-color:var(--primary);
    border-radius:50%;animation:spin .8s linear infinite}}
  @keyframes spin{{to{{transform:rotate(360deg)}}}}
  iframe{{flex:1;width:100%;border:0;background:var(--surface)}}
</style>
</head>
<body>
  <nav class="navbar">
    <a class="back" href="/">&larr; Back to OS</a>
    <span class="crumbs"><a href="/">Agentic OS</a> / <a href="/overview">{html.escape(group)}</a> / <b>{crumb}</b></span>
    <span class="spacer"></span>
    <span class="health"><span class="dot na" id="hdot"></span><span id="hlbl">checking…</span></span>
    <label class="switch">jump to
      <select onchange="if(this.value)location.href='/m/'+this.value">{switcher_html}</select>
    </label>
    <button class="iconbtn" onclick="reframe()" title="Reload this dashboard">&#8635;</button>
    <a class="src" href="{src_url}" target="_blank" rel="noopener">source &#8599;</a>
  </nav>
  <div class="demo-banner" id="demoBanner">
    Demo data for <b>Summit Roofing Co.</b>, a fictional tenant — not a real customer's account.
    <button onclick="document.getElementById('demoBanner').remove()">dismiss</button>
  </div>
  <div class="frame-wrap">
    <div class="loading" id="loading"><span class="spinner"></span> Loading {html.escape(name)}…</div>
    <iframe src="/m/{html.escape(name)}/raw" title="{html.escape(name)} dashboard"
            onload="frameLoaded()"></iframe>
  </div>
<script>
const NAME = {json.dumps(name)};
function reframe(){{ var f = document.querySelector('iframe'); f.src = f.src; }}
function frameLoaded(){{
  var l = document.getElementById('loading'); if (l) l.style.display = 'none';
  // The module dashboard is proxied same-origin, so retarget its EXTERNAL links to
  // open in a new tab — otherwise clicking one navigates the iframe to a site that
  // refuses framing and the dashboard goes blank (no way back but a reload).
  try {{
    var d = document.querySelector('iframe').contentDocument;
    d.querySelectorAll('a[href]').forEach(function(a){{
      var h = a.getAttribute('href') || '';
      if (/^https?:\\/\\//i.test(h) && a.host !== location.host) {{ a.target = '_blank'; a.rel = 'noopener'; }}
    }});
  }} catch (e) {{}}
}}
async function health(){{
  try{{
    const r = await fetch('/api/fleet'); if(!r.ok) return;
    const m = (await r.json()).find(x => x.name === NAME); if(!m) return;
    const dot = document.getElementById('hdot'), lbl = document.getElementById('hlbl');
    dot.className = 'dot ' + (m.health === 'up' ? 'up' : m.health === 'down' ? 'down' : 'na');
    let t = m.health;
    if(m.core) t += ' · core: ' + m.core + (m.connected ? ' ✓' : ' ✕');
    lbl.textContent = t;
  }}catch(e){{}}
}}
health(); setInterval(health, 5000);
</script>
</body>
</html>"""


def overview_page(*, groups: dict, has_agent: set, module_meta: dict, workflows: list) -> str:
    """The /overview "how it works" page: kernel + module map + workflows.

    ``module_meta[name]`` = {"pain","tagline","core","repo","group"}; ``has_agent``
    is the set of module names with a live dashboard; ``workflows`` mirrors the
    dashboard's cross-module flows. Live health is fetched client-side.
    """
    # Kernel pieces — the "what it's comprised of" the user asked for.
    kernel = [
        ("Registry", "A simple config file (no coding needed) that lists every module, the agents it runs, and which actions must pause for your approval."),
        ("Fleet", "The coordinator: starts the modules, gives each its agents, runs them on a schedule, and drives workflows that span several modules."),
        ("Router", "Sends every task to the cheapest model that can do it well — a model on your own hardware first, a premium one only for the hard 5%."),
        ("Context", "The shared memory of your business — the profile, customers, and policies every agent works from."),
        ("Approvals & Audit", "The human-in-the-loop gate: anything that moves money, touches compliance, or changes infrastructure pauses here for your one-click sign-off, and every decision is logged."),
    ]
    kernel_html = "".join(
        f'<div class="kcard" id="{html.escape(t.split()[0].lower())}"><h3>{html.escape(t)}</h3><p>{html.escape(d)}</p></div>'
        for t, d in kernel
    )

    # One-line "what it does" gloss per open-source core.
    core_blurb = {
        "Lago": "subscriptions & billing", "ERPNext": "bookkeeping & accounting",
        "Chatwoot": "customer-support inbox", "Postiz": "social scheduling",
        "CrowdSec": "intrusion detection", "OpenSCAP": "security-compliance scans",
        "Metabase": "business analytics", "changedetection": "website change monitoring",
        "Umami": "web analytics",
    }

    # Module map, grouped, each with its real OSS core.
    sections = []
    for g, members in groups.items():
        cards = []
        for n in members:
            meta = module_meta.get(n, {})
            live = n in has_agent
            core = meta.get("core") or ""
            blurb = core_blurb.get(core)
            core_label = f"core: {core}" + (f" — {blurb}" if blurb else "")
            core_html = f'<span class="core">{html.escape(core_label)}</span>' if core else ""
            href = f"/m/{html.escape(n)}" if live else "https://github.com/" + html.escape(meta.get("repo", ""))
            tgt = "" if live else ' target="_blank" rel="noopener"'
            open_lbl = "Open dashboard &#8594;" if live else "source &#8599;"
            cards.append(
                f'<a class="omod" href="{href}"{tgt} data-name="{html.escape(n)}">'
                f'<span class="omod__top"><span class="dot na omod__dot"></span>'
                f'<span class="omod__name">{html.escape(n)}</span></span>'
                f'<span class="omod__pain">{html.escape(meta.get("pain",""))}</span>'
                f'{core_html}<span class="omod__open">{open_lbl}</span></a>'
            )
        sections.append(
            f'<section class="ogroup"><h2 class="ogroup__label">{html.escape(g)}</h2>'
            f'<div class="ogrid">{"".join(cards)}</div></section>'
        )
    modules_html = "".join(sections)

    workflows_html = "".join(
        f'<li><span class="wf-name">{html.escape(w["name"])}</span>'
        + (f'<span class="wf-desc">{html.escape(w["desc"])}</span>' if w.get("desc") else "")
        + f'<span class="wf-steps">{html.escape(" → ".join(w["steps"]))}</span></li>'
        for w in workflows
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How it works · Agentic OS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&family=Roboto+Mono:wght@400;500&display=swap">
<style>{_BASE_CSS}
  body{{padding:var(--sp-5)}}
  .shell{{max-width:1200px;margin-inline:auto;display:flex;flex-direction:column;gap:var(--sp-6)}}
  .navbar{{display:flex;align-items:center;gap:var(--sp-4);flex-wrap:wrap}}
  .back{{display:inline-flex;align-items:center;gap:6px;font:500 14px/1 var(--font-sans);
    color:var(--on-primary);background:var(--primary);padding:9px 14px;border-radius:var(--radius-pill)}}
  .back:hover{{background:var(--primary-container);color:var(--on-primary-container)}}
  h1{{margin:0;font:400 26px/32px var(--font-sans)}} h1 .accent{{color:var(--primary)}}
  .lede{{color:var(--on-surface-variant);font:400 15px/22px var(--font-sans);max-width:80ch;margin:0}}
  .flow{{display:flex;flex-wrap:wrap;align-items:center;gap:var(--sp-3);
    background:var(--surface-container-low);border:1px solid var(--outline-variant);
    border-radius:var(--radius-lg);padding:var(--sp-5)}}
  .flow .node{{font:500 13px/1 var(--font-sans);color:var(--on-surface);background:var(--surface-container-high);
    border:1px solid var(--outline-variant);padding:8px 12px;border-radius:var(--radius-pill)}}
  .flow .node.gate{{color:var(--secondary);border-color:var(--secondary)}}
  .flow .arr{{color:var(--on-surface-muted);font:400 16px/1 var(--font-mono)}}
  .sect-label{{font:500 12px/16px var(--font-sans);letter-spacing:.5px;text-transform:uppercase;
    color:var(--primary);display:flex;align-items:center;gap:var(--sp-3);margin:0}}
  .sect-label::after{{content:"";flex:1;height:1px;background:var(--outline-variant)}}
  .kgrid{{display:grid;gap:var(--sp-4);grid-template-columns:repeat(auto-fit,minmax(240px,1fr));margin-top:var(--sp-4)}}
  .kcard{{background:var(--surface-container);border:1px solid var(--outline-variant);border-radius:var(--radius-lg);padding:var(--sp-5)}}
  .kcard h3{{margin:0 0 var(--sp-2);font:500 16px/22px var(--font-sans);color:var(--primary)}}
  .kcard p{{margin:0;color:var(--on-surface-variant);font:400 14px/20px var(--font-sans)}}
  .ogroup{{display:flex;flex-direction:column;gap:var(--sp-4);margin-top:var(--sp-4)}}
  .ogroup__label{{font:500 12px/16px var(--font-sans);letter-spacing:.5px;text-transform:uppercase;color:var(--on-surface-muted);margin:0}}
  .ogrid{{display:grid;gap:var(--sp-3);grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}}
  .omod{{display:flex;flex-direction:column;gap:6px;background:var(--surface-container);
    border:1px solid var(--outline-variant);border-radius:var(--radius-md);padding:var(--sp-4);
    transition:border-color .15s,background .15s}}
  .omod:hover{{border-color:var(--primary);background:var(--surface-container-high)}}
  .omod__top{{display:flex;align-items:center;gap:var(--sp-2)}}
  .omod__name{{font:500 15px/20px var(--font-sans);color:var(--on-surface)}}
  .omod__pain{{font:400 13px/18px var(--font-sans);color:var(--on-surface-variant)}}
  .core{{align-self:flex-start;font:500 11px/16px var(--font-mono);color:var(--on-surface-muted)}}
  .omod__open{{align-self:flex-start;font:500 12px/16px var(--font-sans);color:var(--primary);margin-top:2px}}
  .wlist{{list-style:none;margin:var(--sp-4) 0 0;padding:0;display:flex;flex-direction:column;gap:var(--sp-3)}}
  .wlist li{{background:var(--surface-container);border:1px solid var(--outline-variant);border-radius:var(--radius-md);
    padding:var(--sp-4);display:flex;flex-direction:column;gap:4px}}
  .wf-name{{font:500 14px/20px var(--font-sans);color:var(--primary)}}
  .wf-desc{{font:400 13px/18px var(--font-sans);color:var(--on-surface-variant)}}
  .wf-steps{{font:400 12px/18px var(--font-mono);color:var(--on-surface-muted)}}
  .legend{{font:400 13px/18px var(--font-sans);color:var(--on-surface-muted);margin:0 0 var(--sp-2)}}
</style>
</head>
<body>
<div class="shell">
  <div class="navbar">
    <a class="back" href="/">&larr; Back to OS</a>
    <h1><span class="accent">How it works</span> — the Agentic Business OS</h1>
  </div>
  <p class="lede">One control plane runs your whole business as a fleet of <b>agents</b> — automated
  assistants that carry out tasks for you — on a server you own. Each module is built on a proven
  open-source tool and adds agents on top; the kernel coordinates them, sends every task to the
  cheapest capable AI model, and pauses anything risky for your one-click approval. Everything below
  is running live on demo data for a fictional tenant, <b>Summit Roofing Co.</b></p>

  <div class="flow">
    <span class="node">You</span><span class="arr">→</span>
    <span class="node">Control plane</span><span class="arr">→</span>
    <span class="node">Fleet</span><span class="arr">→</span>
    <span class="node">Router (cheapest model)</span><span class="arr">→</span>
    <span class="node">Module agent</span><span class="arr">→</span>
    <span class="node">OSS core</span>
    <span class="arr">·</span><span class="node gate">money / compliance / infra → approval</span>
  </div>

  <section>
    <h2 class="sect-label">The kernel — what runs underneath</h2>
    <div class="kgrid">{kernel_html}</div>
  </section>

  <section>
    <h2 class="sect-label">The modules — grouped by what they do</h2>
    <p class="legend">Each module's <b>core</b> is the open-source tool it is built on.</p>
    {modules_html}
  </section>

  <section>
    <h2 class="sect-label">Cross-module workflows — how they connect</h2>
    <ul class="wlist">{workflows_html}</ul>
  </section>
</div>
<script>
// Live health dots on the module map.
async function health(){{
  try{{
    const r = await fetch('/api/fleet'); if(!r.ok) return;
    const fleet = await r.json();
    document.querySelectorAll('.omod').forEach(a => {{
      const m = fleet.find(x => x.name === a.dataset.name); if(!m) return;
      const dot = a.querySelector('.omod__dot');
      dot.className = 'dot omod__dot ' + (m.health === 'up' ? 'up' : m.health === 'down' ? 'down' : 'na');
    }});
  }}catch(e){{}}
}}
health(); setInterval(health, 5000);
</script>
</body>
</html>"""
