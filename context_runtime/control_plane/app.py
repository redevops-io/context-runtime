"""FastAPI control plane — deploy, observe, and approve, from one place.

    uvicorn context_runtime.control_plane.app:app --port 8080
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import channels, views
from .context import Context
from .fleet import Fleet
from .registry import Module, Registry
from .router import Router

CONFIG_PATH = os.environ.get("CONTEXT_RUNTIME_CONFIG") or os.environ.get("AGENTIC_OS_CONFIG", "config.yaml")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Auth dependency for mutating endpoints.

    If ``AGENTIC_OS_API_KEY`` is set in the environment, every POST route requires
    a matching ``X-API-Key`` header (401 otherwise). If it is unset, auth is
    disabled — in that case the control plane MUST be bound to localhost only
    (do not expose it on 0.0.0.0 without setting AGENTIC_OS_API_KEY).
    """
    expected = os.environ.get("CONTEXT_RUNTIME_API_KEY") or os.environ.get("AGENTIC_OS_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "invalid or missing X-API-Key")


def _build() -> Fleet:
    registry = Registry.load()
    cfg = {}
    if Path(CONFIG_PATH).exists():
        cfg = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8")) or {}
    router = Router.from_config(cfg.get("router", {"tiers": []}))
    # Hermes 0.17 chat notifier — no-op until a Telegram/Slack token is configured.
    notifier = channels.Notifier()
    ctx = Context(os.environ.get("CONTEXT_RUNTIME_HOME") or os.environ.get("AGENTIC_OS_HOME", ".context-runtime"), notifier=notifier)
    return Fleet(registry, router, ctx)


app = FastAPI(title="Context Runtime control plane", version="0.1.0")

# CORS so the vibexgen web UI (a different origin) can call /vibex/* directly.
# Configurable via CR_CORS_ORIGINS (comma-separated); defaults to the vibexgen origins.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

_cors = os.environ.get(
    "CR_CORS_ORIGINS",
    "https://vibexgen.io,https://www.vibexgen.io,http://localhost:5173,http://localhost:8080",
).split(",")
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in _cors if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
)
fleet = _build()
fleet.up()  # (b) stand up every module as a Context Runtime tenant → deployed
# Hermes 0.17 inbound chatops gateway — daemon thread, started only if a channel
# is configured (closed by default; honors AGENTIC_OS_GATEWAY_ALLOW / _OPEN).
_gateway = channels.Gateway(fleet)


@app.on_event("startup")
def _start_gateway() -> None:
    _gateway.start()

# Serve the per-repo card images (deploy/assets/repos/<name>.png) at /assets/...
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "deploy" / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")

# Functional grouping of modules for the dashboard (order matters).
GROUPS: dict[str, list[str]] = {
    "Money": ["agentic-billing", "agentic-books"],
    "Customers": ["agentic-support", "social-autopilot"],
    "Security & Compliance": ["edge-sentinel", "agentic-compliance"],
    "Growth & Intelligence": ["control-tower", "market-radar", "growth-engine"],
    "Build & Platform": ["sidekick"],
}


def _group_for(name: str) -> str:
    for group, members in GROUPS.items():
        if name in members:
            return group
    return "Other"


# Cross-module workflows (mirrors WORKFLOWS in DASHBOARD_HTML) — shown on /overview.
WORKFLOWS: list[dict] = [
    {"name": "New customer onboarding",
     "desc": "Set up the subscription, send a welcome + onboarding, record the books entry, and file the consent record.",
     "steps": ["agentic-billing", "agentic-support", "agentic-books", "agentic-compliance"]},
    {"name": "Storm-damage lead → booked job",
     "desc": "Spot the demand spike, put spend where it converts, answer the lead, and bill the booked job.",
     "steps": ["market-radar", "growth-engine", "agentic-support", "agentic-billing"]},
    {"name": "Security incident",
     "desc": "Triage the threat and propose a block, then log the audit-ready compliance evidence.",
     "steps": ["edge-sentinel", "agentic-compliance"]},
]

# Static module → OSS core label (the live ✓/✕ still comes from each /health).
MODULE_CORES: dict[str, str] = {
    "agentic-billing": "Lago", "agentic-books": "ERPNext", "agentic-compliance": "OpenSCAP",
    "control-tower": "Metabase", "edge-sentinel": "CrowdSec", "market-radar": "changedetection",
    "growth-engine": "Umami", "social-autopilot": "Postiz", "agentic-support": "Chatwoot",
}


# --- real agent-service wiring ----------------------------------------------
# Map each catalog module (modules.yaml name) to the REAL agentic-module service
# running in the integrated compose. Service names are the agent dir names
# (billing, support, …) on internal ports 8201-8209; the control plane proxies
# /m/<name> here and health-checks /health here. Modules NOT in this map have no
# real agent yet (sidekick -> tool/source-only) and keep their existing behavior.
# agentic-books -> http://books:8209 wraps the real ERPNext core.
MODULE_SERVICES: dict[str, str] = {
    "agentic-billing": "http://billing:8201",
    "control-tower": "http://control-tower:8202",
    "edge-sentinel": "http://edge-sentinel:8203",
    "market-radar": "http://market-radar:8204",
    "growth-engine": "http://growth-engine:8205",
    "social-autopilot": "http://social-autopilot:8206",
    "agentic-support": "http://support:8207",
    "agentic-compliance": "http://compliance:8208",
    "agentic-books": "http://books:8209",
    "agentic-crm": "http://agentic-crm:8210",
    "lifecycle": "http://lifecycle:8211",
    "agentic-privacy": "http://agentic-privacy:8212",
    "growth-assistant": "http://growth-assistant:8213",
}

# Allow overriding the whole map (or single entries) via env, e.g.
#   MODULE_SERVICE_agentic_billing=http://billing:8201
for _name in list(MODULE_SERVICES):
    _override = os.environ.get("MODULE_SERVICE_" + _name.replace("-", "_"))
    if _override:
        MODULE_SERVICES[_name] = _override.rstrip("/")

# Short service-name aliases so /m/billing and /m/agentic-billing both resolve
# (the compose service / agent-dir name is the short one: billing, support, …).
SERVICE_ALIASES: dict[str, str] = {
    url.split("//", 1)[1].split(":", 1)[0]: name
    for name, url in MODULE_SERVICES.items()
}


def _resolve_module_name(name: str) -> str:
    """Map a short service alias (billing) to its catalog name (agentic-billing)."""
    return SERVICE_ALIASES.get(name, name)


def _has_real_agent(m: Module) -> bool:
    return m.name in MODULE_SERVICES


class ModuleList(BaseModel):
    name: str
    repo: str
    pain: str
    agents: list[str]


# --- live fleet aggregation --------------------------------------------------
async def _probe_health(client: httpx.AsyncClient, m: Module) -> dict:
    """Probe a module's REAL agent service /health. Never raises.

    Returns {"health","core","connected"}: ``health`` is the agent service reachability
    ("up"/"down"); ``core`` (e.g. "lago") and ``connected`` come straight from the agent's
    own /health JSON, so a card can show "core: Lago ✓". Modules without a real agent
    (agentic-books, sidekick) report health="n/a".
    """
    base = MODULE_SERVICES.get(m.name)
    if base is None:
        # No real agent yet: books -> coming soon / on EC2; sidekick -> tool.
        coming = "coming soon · on EC2" if m.deploy == "compose" else None
        return {"health": "n/a", "core": coming, "connected": None}
    try:
        resp = await client.get(f"{base}/health", timeout=2.5)
        up = 200 <= resp.status_code < 300
        body = {}
        try:
            body = resp.json()
        except Exception:
            body = {}
        return {
            "health": "up" if up else "down",
            "core": body.get("core"),
            "connected": body.get("connected"),
        }
    except Exception:
        return {"health": "down", "core": None, "connected": None}


@app.get("/api/fleet")
async def api_fleet() -> list[dict]:
    """Aggregate every module with a live health probe of its real agent service.

    Health checks run concurrently with a short timeout, so one slow or missing
    module can never block the response. ``core`` + ``connected`` are surfaced from
    each agent's /health so cards can show e.g. "core: Lago ✓".
    """
    mods = list(fleet.registry)
    async with httpx.AsyncClient() as client:
        probes = await asyncio.gather(*(_probe_health(client, m) for m in mods))
    return [
        {
            "name": m.name,
            "repo": m.repo,
            "pain": m.pain,
            "tagline": m.tagline,
            "agents": list(m.agents),
            "approval_required": list(m.approval_required),
            "deploy": m.deploy,
            "port": m.port,
            "health": p["health"],
            "core": p["core"],
            "connected": p["connected"],
            "has_agent": _has_real_agent(m),
            "group": _group_for(m.name),
        }
        for m, p in zip(mods, probes)
    ]


def _switcher_list() -> list[dict]:
    """Modules with a live dashboard, for the shell's jump-to-module dropdown."""
    return [
        {"name": m.name, "group": _group_for(m.name)}
        for m in fleet.registry
        if _has_real_agent(m)
    ]


def _module_meta() -> dict:
    """name -> {pain,tagline,core,repo,group} for the overview module map."""
    return {
        m.name: {
            "pain": m.pain, "tagline": m.tagline, "repo": m.repo,
            "core": MODULE_CORES.get(m.name), "group": _group_for(m.name),
        }
        for m in fleet.registry
    }


@app.get("/m/{name}/raw", response_class=HTMLResponse)
async def module_proxy_raw(name: str) -> HTMLResponse:
    """Reverse-proxy a module's REAL agent-service dashboard, same-origin on :8080.

    Proxies to the agent service's ``/`` (the live MD3 dashboard rendered from real
    OSS-core data). Agent pages are self-contained (inline CSS, no root-relative
    assets), so no URL rewriting is needed. This is the bare page; ``/m/<name>``
    wraps it in the nav shell (back / breadcrumb / switcher). Modules without a
    real agent (sidekick: deploy=tool) have no live dashboard.
    """
    name = _resolve_module_name(name)
    try:
        fleet.registry.get(name)
    except KeyError:
        raise HTTPException(404, f"no module {name}")
    base = MODULE_SERVICES.get(name)
    if base is None:
        raise HTTPException(404, f"module {name} has no live dashboard")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/", timeout=8.0)
    except Exception:
        raise HTTPException(502, f"module {name} is not reachable")
    return HTMLResponse(content=resp.text, status_code=resp.status_code)


@app.api_route("/m/{name}/api/{path:path}", methods=["GET", "POST"])
async def module_api(name: str, path: str, request: Request) -> Response:
    """Forward /m/<name>/api/<path> to the module's /api/<path> (GET + POST).

    ONLY /api/* is forwarded — a module's powerful /agent/run lives OUTSIDE /api/
    and therefore stays private (never reachable through demo.redevops.io). This is
    the public surface for chat + read-only data. The real client IP is passed
    through (CF-Connecting-IP / X-Forwarded-For) so a module can rate-limit.
    """
    name = _resolve_module_name(name)
    base = MODULE_SERVICES.get(name)
    if base is None:
        raise HTTPException(404, f"no module {name}")
    client_ip = (request.headers.get("cf-connecting-ip")
                 or (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
                 or (request.client.host if request.client else ""))
    body = await request.body()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.request(
                request.method, f"{base}/api/{path}",
                params=dict(request.query_params), content=body,
                headers={"content-type": request.headers.get("content-type", "application/json"),
                         "x-forwarded-for": client_ip},
                timeout=210.0)
    except Exception:
        raise HTTPException(502, f"module {name} api unreachable")
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"))


@app.get("/m/{name}", response_class=HTMLResponse)
async def module_shell(name: str) -> HTMLResponse:
    """A module dashboard wrapped in persistent nav chrome.

    Fixes the dead-end problem: the proxied page (now at ``/m/<name>/raw``) renders
    in a same-origin iframe inside a top bar carrying back-to-OS, a breadcrumb, a
    live health dot, a jump-to-module switcher, and the source link.
    """
    name = _resolve_module_name(name)
    try:
        m = fleet.registry.get(name)
    except KeyError:
        raise HTTPException(404, f"no module {name}")
    if name not in MODULE_SERVICES:
        raise HTTPException(404, f"module {name} has no live dashboard")
    return HTMLResponse(views.module_shell(
        name=name, group=_group_for(name), repo=m.repo, switcher=_switcher_list(),
    ))


@app.get("/overview", response_class=HTMLResponse)
async def overview() -> HTMLResponse:
    """The 'how it works' page: kernel + grouped module map + cross-module workflows."""
    return HTMLResponse(views.overview_page(
        groups=GROUPS,
        has_agent=set(MODULE_SERVICES),
        module_meta=_module_meta(),
        workflows=WORKFLOWS,
    ))


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modules": len(fleet.registry)}


@app.get("/modules", response_model=list[ModuleList])
def modules() -> list[ModuleList]:
    return [ModuleList(name=m.name, repo=m.repo, pain=m.pain, agents=list(m.agents))
            for m in fleet.registry]


@app.get("/status")
def status() -> list[dict]:
    return [asdict(s) for s in fleet.status()]


@app.post("/up", dependencies=[Depends(require_api_key)])
def up(names: list[str] | None = None) -> list[dict]:
    return [asdict(s) for s in fleet.up(*(names or []))]


@app.post("/down", dependencies=[Depends(require_api_key)])
def down(names: list[str] | None = None) -> list[dict]:
    return [asdict(s) for s in fleet.down(*(names or []))]


@app.get("/approvals")
def approvals() -> list[dict]:
    return [asdict(a) for a in fleet.context.pending()]


@app.post("/approvals/{approval_id}/{decision}", dependencies=[Depends(require_api_key)])
def resolve(approval_id: str, decision: str) -> dict:
    if decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be approve|reject")
    ap = fleet.context.resolve(approval_id, approved=decision == "approve")
    if ap is None:
        raise HTTPException(404, f"no pending approval {approval_id}")
    return asdict(ap)


class DispatchRequest(BaseModel):
    module: str
    agent: str
    action: str
    prompt: str = ""
    capability: str = "reason"
    background: bool = False


@app.post("/dispatch", dependencies=[Depends(require_api_key)])
def dispatch(req: DispatchRequest) -> dict:
    """Run one agent action through the fleet (router-picked model).

    Approval-gated actions return a pending Approval (and ping chat if a notifier
    is configured); ``background=true`` returns a job handle immediately (Hermes
    0.17 background subagents) — poll ``/jobs/<id>``."""
    try:
        out = fleet.dispatch(req.module, req.agent, req.action, req.prompt,
                             capability=req.capability, background=req.background)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))
    if hasattr(out, "id"):          # an Approval
        return {"kind": "approval", **asdict(out)}
    if isinstance(out, dict) and "job_id" in out:   # a background job handle
        return {"kind": "job", **out}
    if isinstance(out, dict):                        # a Context Runtime plan result
        return out
    return {"kind": "result", "text": out}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = fleet.job(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return {"job_id": job_id, **job}


@app.get("/notify/status")
def notify_status() -> dict:
    """What chat channels are wired + whether the inbound gateway is running."""
    n = fleet.context.notifier
    chans = [c.name for c in n.channels] if n is not None else []
    return {"channels": chans, "notifier_enabled": bool(chans),
            "gateway_running": bool(_gateway.channels)}


# --- single-page dashboard ---------------------------------------------------
DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>redevops.io — Agentic Business OS · Summit Roofing Co.</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500&family=Roboto+Mono:wght@400;500&display=swap">
<style>
  :root{
    --surface-dim:#0e0e11; --surface:#131316; --surface-bright:#393a3d;
    --surface-container-lowest:#0d0e10; --surface-container-low:#1b1b1f;
    --surface-container:#1f1f23; --surface-container-high:#2a2a2e; --surface-container-highest:#353539;
    --on-surface:#e4e2e6; --on-surface-variant:#c7c5ca; --on-surface-muted:#918f96;
    --outline:#938f99; --outline-variant:#2f2f33;
    --primary:#4fd1c5; --on-primary:#00201c; --primary-container:#00504a; --on-primary-container:#a8f0e6;
    --secondary:#f5b544; --on-secondary:#3d2e00; --secondary-container:#5c4500;
    --success:#5bd98a; --success-container:#0f3d22; --warning:#f5b544; --warning-container:#4a3500;
    --danger:#f2544f; --danger-container:#5c1512; --info:#5aa9f0; --info-container:#103a5c;
    --sp-1:4px;--sp-2:8px;--sp-3:12px;--sp-4:16px;--sp-5:24px;--sp-6:32px;--sp-7:40px;--sp-8:48px;
    --radius-sm:8px;--radius-md:12px;--radius-lg:16px;--radius-xl:28px;--radius-pill:999px;
    --shadow-1:0 1px 2px rgba(0,0,0,.45);--shadow-2:0 2px 6px rgba(0,0,0,.5);
    --font-sans:"Roboto",system-ui,-apple-system,"Segoe UI",sans-serif;
    --font-mono:"Roboto Mono",ui-monospace,"SF Mono",monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface);color:var(--on-surface);font-family:var(--font-sans);line-height:1.45;padding:var(--sp-5)}
  a{color:var(--primary);text-decoration:none}
  .shell{max-width:1440px;margin-inline:auto;display:flex;flex-direction:column;gap:var(--sp-5)}
  .grid{display:grid;gap:var(--sp-4)}
  .pill{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 10px;border-radius:var(--radius-pill);font:500 12px/1 var(--font-sans)}
  .pill--success{background:var(--success-container);color:var(--success)}
  .pill--warn{background:var(--warning-container);color:var(--warning)}
  .pill--danger{background:var(--danger-container);color:var(--danger)}
  .pill--neutral{background:var(--surface-container-highest);color:var(--on-surface-variant)}
  .pill__dot{width:6px;height:6px;border-radius:50%;background:currentColor}

  /* compact app bar */
  .appbar{background:var(--surface-container-low);border:1px solid var(--outline-variant);border-radius:var(--radius-lg);padding:var(--sp-5)}
  .appbar__row{display:flex;align-items:center;justify-content:space-between;gap:var(--sp-4);flex-wrap:wrap}
  .appbar h1{margin:0;font:400 24px/32px var(--font-sans);color:var(--on-surface)}
  .appbar h1 .accent{color:var(--primary)}
  .appbar__sub{margin-top:var(--sp-1);color:var(--on-surface-muted);font:400 14px/20px var(--font-sans)}
  .appbar__tenant{margin-top:var(--sp-2);color:var(--on-surface-variant);font:400 13px/18px var(--font-sans)}
  .appbar__tenant b{color:var(--on-surface)}
  .fleet-pill{font:500 13px/1 var(--font-mono);font-feature-settings:"tnum"}

  /* slim two-card band */
  .band{display:grid;gap:var(--sp-4);grid-template-columns:1fr 1fr}
  @media(max-width:839px){.band{grid-template-columns:1fr}}
  .card{background:var(--surface-container);border:1px solid var(--outline-variant);border-radius:var(--radius-lg);padding:var(--sp-5);display:flex;flex-direction:column;gap:var(--sp-3)}
  .card__title{margin:0;font:500 12px/16px var(--font-sans);letter-spacing:.5px;text-transform:uppercase;color:var(--on-surface-muted)}
  .card ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:var(--sp-3)}
  .card li{font:400 14px/20px var(--font-sans);color:var(--on-surface)}
  .meta{color:var(--on-surface-muted);font:400 12px/16px var(--font-mono)}
  .wf-name{color:var(--primary);font:500 14px/20px var(--font-sans)}
  .wf-steps{color:var(--on-surface-muted);font:400 12px/16px var(--font-sans)}
  .empty{color:var(--on-surface-muted);font:400 14px/20px var(--font-sans)}

  /* functional groups */
  .group{display:flex;flex-direction:column;gap:var(--sp-4)}
  .group-label{font:500 12px/16px var(--font-sans);letter-spacing:.5px;text-transform:uppercase;color:var(--primary);display:flex;align-items:center;gap:var(--sp-3);margin:0}
  .group-label::after{content:"";flex:1;height:1px;background:var(--outline-variant)}
  .module-grid{display:grid;gap:var(--sp-4);grid-template-columns:repeat(auto-fit,minmax(320px,1fr));align-items:stretch}

  /* equal-height module cards */
  .mcard{background:var(--surface-container);border:1px solid var(--outline-variant);border-radius:var(--radius-lg);
    display:flex;flex-direction:column;overflow:hidden;transition:border-color .15s,background .15s}
  .mcard:hover{border-color:var(--primary);background:var(--surface-container-high)}
  .mcard .thumb{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:var(--surface-container-high);border-bottom:1px solid var(--outline-variant)}
  .mcard .body{padding:var(--sp-4) var(--sp-5);display:flex;flex-direction:column;gap:var(--sp-3);flex:1}
  .mcard .top{display:flex;align-items:center;justify-content:space-between;gap:var(--sp-3)}
  .mcard .name{font:500 16px/24px var(--font-sans);letter-spacing:.15px;color:var(--on-surface)}
  .mcard .pain{align-self:flex-start;font:500 12px/16px var(--font-sans);letter-spacing:.5px;text-transform:uppercase;
    color:var(--primary);background:var(--primary-container);color:var(--on-primary-container);padding:2px 10px;border-radius:var(--radius-pill)}
  .mcard .tagline{color:var(--on-surface-variant);font:400 14px/20px var(--font-sans)}
  .chips{display:flex;flex-wrap:wrap;gap:var(--sp-2)}
  .chip{font:400 12px/16px var(--font-sans);color:var(--on-surface-variant);background:var(--surface-container-high);
    border:1px solid var(--outline-variant);padding:2px 8px;border-radius:var(--radius-sm)}
  .approval-note{align-self:flex-start;font:400 12px/16px var(--font-sans);color:var(--warning);
    background:var(--warning-container);border-radius:var(--radius-sm);padding:4px 10px}
  .status{display:flex;align-items:center;gap:var(--sp-2);font:400 13px/18px var(--font-sans);color:var(--on-surface-muted)}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--on-surface-muted);flex:none}
  .dot.up{background:var(--success);box-shadow:0 0 8px rgba(91,217,138,.7)}
  .dot.down{background:var(--danger)}
  .dot.na{background:var(--on-surface-muted)}
  .mcard .foot{margin-top:auto;padding:var(--sp-4) var(--sp-5);border-top:1px solid var(--outline-variant);
    display:flex;align-items:center;gap:var(--sp-4)}
  .btn-open{font:500 14px/1 var(--font-sans);color:var(--on-primary);background:var(--primary);
    padding:9px 16px;border-radius:var(--radius-pill)}
  .btn-open:hover{background:var(--primary-container);color:var(--on-primary-container)}
  .src{font:400 13px/18px var(--font-sans);color:var(--on-surface-muted)}
  .src:hover{color:var(--primary)}
  .tool-note{font:400 13px/18px var(--font-sans);color:var(--secondary)}
  .demo-note{display:flex;align-items:center;gap:var(--sp-3);background:var(--info-container);
    color:var(--info);border:1px solid var(--outline-variant);border-radius:var(--radius-lg);
    padding:10px var(--sp-5);font:400 13px/18px var(--font-sans)}
  .demo-note b{color:var(--on-surface)} .demo-note a{color:inherit;text-decoration:underline}
  .demo-note button{margin-left:auto;background:none;border:1px solid currentColor;color:inherit;
    border-radius:var(--radius-pill);padding:2px 10px;font:500 12px/1 var(--font-sans);cursor:pointer}
  .card__link{margin-top:auto;font:500 13px/18px var(--font-sans);color:var(--primary)}
  footer{color:var(--on-surface-muted);font:400 12px/16px var(--font-sans);text-align:center;padding-top:var(--sp-2)}
  footer code{font-family:var(--font-mono)}
</style>
</head>
<body>
<div class="shell">
  <header class="appbar">
    <div class="appbar__row">
      <div>
        <h1><span class="accent">redevops.io</span> — Agentic Business OS</h1>
        <div class="appbar__sub">AI agents that run your billing, support, security, and growth — on proven open-source tools, on hardware you own.</div>
        <div class="appbar__tenant">Demo tenant: <b>Summit Roofing Co.</b> — a fictional roofing contractor running entirely on agents. Everything below is live on demo data.</div>
      </div>
      <div style="display:flex;align-items:center;gap:var(--sp-3);flex-wrap:wrap">
        <a href="https://github.com/redevops-io/agentic-os" target="_blank" rel="noopener" style="font:500 14px/1 var(--font-sans);color:var(--on-primary);background:var(--primary);padding:9px 16px;border-radius:var(--radius-pill)">Get started &#8599;</a>
        <a href="/overview" style="font:500 14px/1 var(--font-sans);color:var(--primary);background:var(--surface-container-high);border:1px solid var(--outline-variant);padding:9px 16px;border-radius:var(--radius-pill)">How it works</a>
        <span class="pill pill--success fleet-pill" id="summary"><span class="pill__dot"></span>loading…</span>
      </div>
    </div>
  </header>

  <div class="demo-note" id="demoNote">
    <span>Live demo on fictional <b>Summit Roofing Co.</b> data — real open-source cores, simulated business. Actions that move money or change infrastructure are gated and safe to explore.</span>
    <button onclick="document.getElementById('demoNote').remove()">dismiss</button>
  </div>

  <section class="band">
    <div class="card">
      <h2 class="card__title">Approvals — your one-click sign-offs</h2>
      <ul id="approvals"><li class="empty">checking…</li></ul>
      <a class="card__link" href="/overview#approvals">How approvals work &#8594;</a>
    </div>
    <div class="card">
      <h2 class="card__title">Cross-module workflows</h2>
      <ul id="workflows"></ul>
    </div>
  </section>

  <div id="groups" class="shell" style="gap:var(--sp-6)"></div>

  <footer>redevops.io — Agentic Business OS · self-hosted &amp; open-core · <a href="https://github.com/redevops-io/agentic-os" target="_blank" rel="noopener">source &#8599;</a></footer>
</div>

<script>
const WORKFLOWS = [
  { name: "New customer onboarding",
    steps: ["agentic-billing", "agentic-support", "agentic-books", "agentic-compliance"] },
  { name: "Storm-damage lead → booked job",
    steps: ["market-radar", "growth-engine", "agentic-support", "agentic-billing"] },
  { name: "Security incident",
    steps: ["edge-sentinel", "agentic-compliance"] },
];

// Functional groups, in display order (mirrors GROUPS in control_plane.py).
const GROUP_ORDER = [
  "Money",
  "Customers",
  "Security & Compliance",
  "Growth & Intelligence",
  "Build & Platform",
];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function renderWorkflows() {
  const ul = document.getElementById("workflows");
  ul.innerHTML = "";
  WORKFLOWS.forEach(w => {
    const li = el("li");
    li.appendChild(el("span", "wf-name", w.name));
    li.appendChild(document.createElement("br"));
    li.appendChild(el("span", "wf-steps", w.steps.join(" → ")));
    ul.appendChild(li);
  });
}

function dotClass(health) {
  if (health === "up") return "dot up";
  if (health === "down") return "dot down";
  return "dot na";
}

function coreLabel(core) {
  // Pretty-print the OSS core name surfaced by each agent's /health.
  const map = {
    lago: "Lago", metabase: "Metabase", crowdsec: "CrowdSec",
    changedetection: "changedetection", umami: "Umami", postiz: "Postiz",
    chatwoot: "Chatwoot", openscap: "OpenSCAP", oscap: "OpenSCAP",
  };
  if (!core) return null;
  return map[String(core).toLowerCase()] || core;
}

function makeCard(m) {
  // Three kinds: real agent (has_agent), coming-soon (deploy=compose, no agent),
  // and tool (deploy=tool, e.g. sidekick).
  const hasAgent = !!m.has_agent;
  const isTool = m.deploy !== "compose";
  const card = el("div", "mcard");

  // Thumbnail (16:9 cover; links to live dashboard, or GitHub when there's none).
  const img = el("img", "thumb");
  img.src = "/assets/repos/" + m.name + ".png";
  img.alt = m.name;
  img.loading = "lazy";
  const imgLink = el("a");
  imgLink.href = hasAgent ? ("/m/" + m.name) : ("https://github.com/" + m.repo);
  if (!hasAgent) { imgLink.target = "_blank"; imgLink.rel = "noopener"; }
  imgLink.appendChild(img);
  card.appendChild(imgLink);

  const bodyEl = el("div", "body");
  const top = el("div", "top");
  top.appendChild(el("span", "name", m.name));
  const status = el("span", "status");
  status.appendChild(el("span", dotClass(hasAgent ? m.health : "na")));
  status.appendChild(el("span", null, isTool ? "tool" : (hasAgent ? m.health : "soon")));
  top.appendChild(status);
  bodyEl.appendChild(top);

  bodyEl.appendChild(el("span", "pain", m.pain));
  if (m.tagline) bodyEl.appendChild(el("div", "tagline", m.tagline));

  // Real OSS core badge: "core: Lago ✓" (connected) / "✕" (unreachable).
  if (hasAgent) {
    const core = coreLabel(m.core);
    if (core) {
      const cls = m.connected ? "pill pill--success" : "pill pill--danger";
      const mark = m.connected ? " ✓" : " ✕";
      const cp = el("span", cls);
      cp.appendChild(el("span", "pill__dot"));
      cp.appendChild(el("span", null, "core: " + core + mark));
      bodyEl.appendChild(cp);
    }
  } else if (m.deploy === "compose" && m.core) {
    const cp = el("span", "pill pill--neutral");
    cp.appendChild(el("span", "pill__dot"));
    cp.appendChild(el("span", null, m.core));
    bodyEl.appendChild(cp);
  }

  if (m.agents && m.agents.length) {
    const chips = el("div", "chips");
    m.agents.forEach(a => chips.appendChild(el("span", "chip", a)));
    bodyEl.appendChild(chips);
  }
  if (m.approval_required && m.approval_required.length) {
    bodyEl.appendChild(el("div", "approval-note", "approval-gated: " + m.approval_required.join(", ")));
  }
  card.appendChild(bodyEl);

  // Footer: primary filled-teal "Open dashboard" button; secondary "source" link.
  const foot = el("div", "foot");
  if (hasAgent) {
    const open = el("a", "btn-open", "Open dashboard");
    open.href = "/m/" + m.name;
    foot.appendChild(open);
  } else {
    foot.appendChild(el("span", "tool-note", isTool ? "CLI tool — no dashboard by design; see source" : "coming soon — on EC2"));
  }
  const src = el("a", "src", "source ↗");
  src.href = "https://github.com/" + m.repo;
  src.target = "_blank"; src.rel = "noopener";
  foot.appendChild(src);
  card.appendChild(foot);
  return card;
}

function renderFleet(mods) {
  const root = document.getElementById("groups");
  root.innerHTML = "";
  let up = 0, total = 0;

  const byGroup = {};
  mods.forEach(m => {
    if (m.has_agent) {
      total++;
      if (m.health === "up") up++;
    }
    const g = m.group || "Other";
    (byGroup[g] = byGroup[g] || []).push(m);
  });

  const order = GROUP_ORDER.slice();
  Object.keys(byGroup).forEach(g => { if (!order.includes(g)) order.push(g); });

  order.forEach(g => {
    const members = byGroup[g];
    if (!members || !members.length) return;
    const section = el("section", "group");
    section.appendChild(el("div", "group-label", g));
    const grid = el("div", "module-grid");
    members.forEach(m => grid.appendChild(makeCard(m)));
    section.appendChild(grid);
    root.appendChild(section);
  });

  const summary = document.getElementById("summary");
  summary.innerHTML = "";
  summary.appendChild(el("span", "pill__dot"));
  summary.appendChild(el("span", null, up + "/" + total + " modules up"));
}

async function pollFleet() {
  try {
    const r = await fetch("/api/fleet");
    if (r.ok) renderFleet(await r.json());
  } catch (e) { /* keep last render */ }
}

async function pollApprovals() {
  const ul = document.getElementById("approvals");
  try {
    const r = await fetch("/approvals");
    if (!r.ok) throw new Error("bad status");
    const list = await r.json();
    ul.innerHTML = "";
    if (!list.length) {
      ul.appendChild(el("li", "empty", "Nothing needs your sign-off right now — money, compliance, and infra actions pause here."));
      return;
    }
    list.forEach(a => {
      const li = el("li");
      li.appendChild(el("span", null, (a.module || "?") + ": " + (a.summary || a.action || "pending")));
      if (a.id) {
        li.appendChild(document.createElement("br"));
        li.appendChild(el("span", "meta", "id " + a.id));
      }
      // Hermes 0.17 safety scan: warn the approver before they click approve.
      if (a.findings && a.findings.length) {
        li.appendChild(document.createElement("br"));
        li.appendChild(el("span", "approval-note", "⚠ safety: " + a.findings.join("; ")));
      }
      ul.appendChild(li);
    });
  } catch (e) {
    ul.innerHTML = "";
    ul.appendChild(el("li", "empty", "nothing waiting on you"));
  }
}

renderWorkflows();
pollFleet();
pollApprovals();
setInterval(pollFleet, 5000);
setInterval(pollApprovals, 5000);
</script>
</body>
</html>
"""


class RunRequest(BaseModel):
    module: str
    question: str


@app.post("/agent/run", dependencies=[Depends(require_api_key)])
def agent_run(req: RunRequest) -> dict:
    """Plan a module's answer through its Context Runtime tenant (no side effects)."""
    try:
        if req.module not in fleet.tenants:
            fleet.up(req.module)
        return fleet._plan(req.module, "", "answer", req.question)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


class OutcomeRequest(BaseModel):
    module: str
    question: str
    success: bool


@app.post("/agent/outcome", dependencies=[Depends(require_api_key)])
def agent_outcome(req: OutcomeRequest) -> dict:
    """Close the learning loop: report whether a planned answer succeeded so the
    module's tenant updates its policy. (This is how the fleet self-improves.)"""
    if req.module not in fleet.tenants:
        raise HTTPException(404, f"module {req.module} not deployed")
    reward = fleet.record_outcome(req.module, req.question, req.success)
    return {"module": req.module, "reward": reward, "policy": fleet.tenants[req.module].policy()}


@app.get("/agent/policy/{module}")
def agent_policy(module: str) -> dict:
    """The module tenant's current learned policy (best source bundle per intent)."""
    if module not in fleet.tenants:
        raise HTTPException(404, f"module {module} not deployed")
    return {"module": module, "policy": fleet.tenants[module].policy()}


# ──────────────────────────── vibexgen — video generation planning ────────────────────────────
from ..integrations.vibexgen import CRITERIA_WEIGHTS, SceneSpec, VibexgenPlanner

vibex = VibexgenPlanner(runtime=fleet.runtime)   # shares the fleet's cost model


class ScenePayload(BaseModel):
    characters: list[str] = []
    lighting: str = ""
    scenery: str = ""
    motion: str = "static"
    style: str = "realistic"
    has_speech: bool = False

    def to_spec(self) -> SceneSpec:
        return SceneSpec(tuple(self.characters), self.lighting, self.scenery,
                         self.motion, self.style, self.has_speech)


class VibexScenarioReq(BaseModel):
    request: str
    candidates: list[str]


class VibexPlanReq(BaseModel):
    template: str
    scene: ScenePayload


class VibexScoreReq(BaseModel):
    template: str
    scene: ScenePayload
    scores: dict[str, float]
    gen_cost_usd: float = 0.0
    gen_latency_s: float = 0.0


@app.get("/vibex/criteria")
def vibex_criteria() -> dict:
    """The scoring criteria + weights — the UI renders one grading slider per criterion."""
    return {"criteria": CRITERIA_WEIGHTS}


@app.post("/vibex/scenario")
def vibex_scenario(req: VibexScenarioReq) -> dict:
    """Stage 1: pick the best of 2-3 candidate scenario texts BEFORE generating."""
    c = vibex.select_scenario(req.request, req.candidates)
    return {"index": c.index, "scenario": c.scenario, "predicted": c.predicted}


@app.post("/vibex/plan")
def vibex_plan(req: VibexPlanReq) -> dict:
    """Stage 2: the suggested generation chain for this template + scene (shown pre-generation)."""
    scene = req.scene.to_spec()
    chain = vibex.plan_chain(req.template, scene)
    return {"chain": chain.key, "engine": chain.engine, "mode": chain.mode, "model": chain.model,
            "steps": chain.steps, "resolution": chain.resolution,
            "est_cost_units": round(chain.cost_units(), 3),
            "suggestion": vibex.suggest(req.template, scene)}


@app.post("/vibex/score", dependencies=[Depends(require_api_key)])
def vibex_score(req: VibexScoreReq) -> dict:
    """Stage 3: record the user's multi-criteria scores → the policy learns."""
    scene = req.scene.to_spec()
    reward = vibex.record_scores(req.template, scene, req.scores, req.gen_cost_usd, req.gen_latency_s)
    return {"reward": reward, "suggestion": vibex.suggest(req.template, scene)}


@app.get("/vibex/scoreboard")
def vibex_scoreboard() -> dict:
    """Leaderboard of generation chains by learned score — for the UI scoreboard."""
    return {"leaderboard": vibex.leaderboard(), "by_context": vibex.scoreboard()}
