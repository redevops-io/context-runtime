"""The geo demo app served under a configurable URL prefix (PUBLIC_BASE_PATH), with no proxy-side rewrite.

The app is driven through its ASGI interface in-process (no socket, no extra client library). With the prefix empty it
serves at the root as before; with `/geo` (and a two-segment prefix) every route exists only under the prefix.
`check_contract` is the one statement of that behaviour; the mutation tests run it against copies of app.py with one
prefix removed and require it to fail each time.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from unittest import mock
from urllib.parse import urljoin, urlsplit

import pytest

pytest.importorskip("fastapi")           # control-plane extra; skip if not installed

REPO = Path(__file__).resolve().parents[1]
APP_PY = REPO / "deploy" / "geo-demo" / "app.py"
INDEX_HTML = REPO / "deploy" / "geo-demo" / "static" / "index.html"
INTERNAL_HOST = "geo-internal.invalid:8098"        # what a request's Host header could carry behind a proxy
PREFIXES = ["", "/geo", "/shared/geo"]


# --- helpers -------------------------------------------------------------------------------------------------------

def load_module(source: str | None = None, env: dict | None = None, name: str = "geo_demo_app") -> types.ModuleType:
    """Import app.py (or a modified copy of its source) fresh, with PUBLIC_BASE_PATH set as given (unset by default)."""
    clean = {k: v for k, v in os.environ.items() if k != "PUBLIC_BASE_PATH"}
    clean.update(env or {})
    with mock.patch.dict(os.environ, clean, clear=True):
        spec = importlib.util.spec_from_file_location(name, APP_PY)
        mod = importlib.util.module_from_spec(spec)
        if source is None:
            spec.loader.exec_module(mod)
        else:
            mod.__file__ = str(APP_PY)
            exec(compile(source, str(APP_PY), "exec"), mod.__dict__)
        return mod


class Resp:
    def __init__(self, status: int, headers: list[tuple[str, str]], body: bytes):
        self.status, self.raw_headers, self.body = status, headers, body
        self.headers = {k.lower(): v for k, v in headers}

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self):
        return json.loads(self.body)

    def everything(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.raw_headers) + "\n" + self.text


def get(app, target: str, method: str = "GET") -> Resp:
    path, _, query = target.partition("?")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
             "headers": [(b"host", INTERNAL_HOST.encode())], "server": ("127.0.0.1", 8098), "client": ("127.0.0.1", 50000)}
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return Resp(start["status"], [(k.decode("latin-1"), v.decode("latin-1")) for k, v in start["headers"]], body)


def scan_root_relative(page: str, prefix: str) -> list[str]:
    """Every root-relative URL-looking string in the page source that is not under `prefix` (and every use of an API
    that builds URLs the scan cannot follow)."""
    found = []
    for pat in (r"""(?:src|href|action)\s*=\s*["'](/[^"']*)""",          # attributes
                r"""url\(\s*["']?(/[^)"']*)""",                           # css url(...)
                r"""fetch\(\s*["'`](/[^"'`]*)""",                         # fetch with a literal
                r"""["'`](/[A-Za-z0-9_.~-][^"'`\s]*)["'`]"""):             # any quoted root-relative literal
        for m in re.finditer(pat, page):
            u = m.group(1)
            if prefix and (u == prefix or u.startswith(prefix + "/")):
                continue
            found.append(u)
    for word in ("EventSource", "XMLHttpRequest", "window.open", "location", "sendBeacon", "WebSocket", "import("):
        if word in page:
            found.append(f"<uses {word}>")
    return found


def page_urls(page: str, base: str) -> list[str]:
    """The URLs the page uses, as the browser would request them (path form), following the page's own URL builder."""
    assert re.search(r"const BASE = " + re.escape(json.dumps(base)) + ";", page), "page BASE differs from the prefix"
    assert "const api = name => `${BASE}/api/${name}`;" in page
    urls = [f"{base}/api/{n}" for n in re.findall(r"fetch\(api\('(\w+)'\)\)", page)]
    assert page.count("fetch(") == len(urls), "a fetch() the test cannot follow"
    page_url = f"http://{INTERNAL_HOST}{base}/"
    for m in re.finditer(r"""(?:src|href|action)\s*=\s*["']([^"']*)""", page):
        urls.append(urljoin(page_url, m.group(1)) if not m.group(1).startswith(("http:", "https:")) else m.group(1))
    return urls


def route_paths(routes, prefix: str = ""):
    """Every registered path of the routes that expose one, descending into mounts. Newer FastAPI keeps an included
    router as an object without a path or sub-routes of its own; its paths are checked through app.openapi() in the
    caller and through the HTTP-level tests."""
    for r in routes:
        path = getattr(r, "path", None)
        if path is not None:
            yield prefix + path
            continue
        sub = getattr(r, "routes", None)
        if sub is None:
            sub = getattr(getattr(r, "router", None), "routes", None)
        if sub is None:
            continue        # an object FastAPI keeps for an included router: its paths are checked through the OpenAPI schema
        yield from route_paths(sub, prefix + (getattr(r, "prefix", "") or ""))


def check_contract(mod: types.ModuleType, base: str) -> None:
    """The prefix behaviour, stated once. Raises AssertionError on the first violation."""
    app = mod.create_app(base)
    responses: list[Resp] = []
    # no registered route (page, API, docs, health, redirect, static mount) lies outside the prefix
    paths = list(route_paths(app.routes))
    assert paths and all(isinstance(p, str) and (p == base or p.startswith(base + "/")) for p in paths), paths
    schema_paths = list(app.openapi().get("paths", {}))          # includes the routes of included routers
    assert schema_paths and all(p == base or p.startswith(base + "/") for p in schema_paths), schema_paths

    def fetch(target: str, **kw) -> Resp:
        r = get(app, target, **kw)
        responses.append(r)
        return r

    # the page, at <prefix>/ and <prefix>/index.html, and every URL it uses
    page = fetch(f"{base}/")
    assert page.status == 200 and page.headers["content-type"].startswith("text/html"), page.status
    assert fetch(f"{base}/index.html").text == page.text
    if base:
        assert not scan_root_relative(page.text, base), scan_root_relative(page.text, base)
    api_seen = 0
    for u in page_urls(page.text, base):
        if u.startswith(("http:", "https:")):
            host = urlsplit(u).netloc
            assert host in ("redevops.io", INTERNAL_HOST), u         # only the site itself or the app's own origin
            if host == "redevops.io":
                continue
            u = urlsplit(u).path
        if u == "/enterprise-plugins":                              # the one link to the host's root, not to this app
            continue
        assert u == base or u.startswith(base + "/"), f"outside the prefix: {u}"
        r = fetch(u)
        assert r.status == 200, (u, r.status)
        api_seen += u.startswith(f"{base}/api/")
    assert api_seen == 3
    # the other endpoints, health included
    for path in ("/healthz", "/api/parcels", "/openapi.json", "/docs", "/redoc"):
        assert fetch(f"{base}{path}").status == 200, path
    h = fetch(f"{base}/healthz").json()
    assert h["ok"] is True and h["parcels"] == len(mod._PARCELS)
    # nothing outside the prefix (an empty prefix has no outside)
    if base:
        for path in ("/", "/index.html", "/healthz", "/api/summary", "/api/evaluate", "/api/spatial", "/api/parcels",
                     "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect", "/static/index.html", "/anything",
                     base + "x", base + "x/api/summary", base[: base.rindex("/")] + "/zz" if base.count("/") > 1 else "/ge"):
            r = fetch(path)
            assert r.status == 404, (path, r.status)
        # <prefix> -> <prefix>/, path only, query kept; a stray trailing slash elsewhere does not redirect to the origin
        r = fetch(base)
        assert r.status in (301, 302, 307, 308) and r.headers["location"] == base + "/", (r.status, r.headers)
        r = fetch(base + "?a=1&b=two")
        assert r.headers["location"] == base + "/?a=1&b=two"
        assert fetch(base + "/api/summary/").status == 404
        assert fetch(base + "/healthz/").status == 404
    # no response names the internal origin, sets a cookie, or points a Location outside the prefix
    for r in responses:
        assert INTERNAL_HOST not in r.everything() and "127.0.0.1" not in r.everything()
        assert "set-cookie" not in r.headers
        loc = r.headers.get("location")
        assert loc is None or (loc.startswith("/") and not loc.startswith("//")
                               and (loc == base or loc.startswith(base + "/")))


# --- the base path value -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "/geo", "/a/b", "/geo-1_x.y~z", "/Geo/V2"])
def test_valid_values_are_returned_unchanged(value):
    assert load_module().validate_base_path(value) == value


@pytest.mark.parametrize("value", [
    "geo", "geo/", "/", "/geo/", "//geo", "/geo//x", "/..", "/a/../b", "/a..b", "/.", "/./a", "/geo bar", "/geo?x=1",
    "/geo#x", "/g%65o", "/géo", " /geo", "/geo ", "/geo\n", "/geo\\x", "/geo;x", "/geo:80", "http://h/geo", "/" + "a" * 200,
])
def test_invalid_values_are_refused(value):
    mod = load_module()
    with pytest.raises(mod.InvalidBasePath):
        mod.validate_base_path(value)
    with pytest.raises(mod.InvalidBasePath):
        mod.create_app(value)


@pytest.mark.parametrize("value", ["geo", "/geo/", "/a/../b", "/geo bar"])
def test_the_app_refuses_to_start_on_an_invalid_environment_value(value):
    with pytest.raises(ValueError) as e:
        load_module(env={"PUBLIC_BASE_PATH": value})                # what uvicorn does when it imports app:app
    assert "PUBLIC_BASE_PATH" in str(e.value)


def test_the_environment_variable_sets_the_prefix_and_unset_means_root():
    assert get(load_module().app, "/healthz").status == 200
    assert get(load_module().app, "/geo/healthz").status == 404
    prefixed = load_module(env={"PUBLIC_BASE_PATH": "/geo"})
    assert get(prefixed.app, "/geo/healthz").status == 200 and get(prefixed.app, "/healthz").status == 404
    assert get(load_module(env={"PUBLIC_BASE_PATH": ""}).app, "/healthz").status == 200


# --- behaviour -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("base", PREFIXES)
def test_contract_holds_for_each_prefix(base):
    check_contract(load_module(), base)


def test_empty_prefix_serves_at_the_root_exactly_as_before():
    app = load_module().app
    page = get(app, "/")
    assert page.status == 200 and "const BASE = \"\";" in page.text and "fetch(api('summary'))" in page.text
    assert get(app, "/index.html").text == page.text
    assert get(app, "/healthz").json()["ok"] is True
    for p in ("/api/summary", "/api/parcels", "/api/evaluate", "/api/spatial", "/docs", "/openapi.json"):
        assert get(app, p).status == 200, p
    assert get(app, "/geo/api/summary").status == 404
    assert get(app, "/missing").status == 404
    # the one link that leaves the app is the same root path it always was
    assert 'href="/enterprise-plugins"' in page.text


def test_results_are_identical_under_every_prefix_and_deterministic():
    mod = load_module()
    root = {p: get(mod.create_app(""), p).body for p in ("/api/summary", "/api/parcels", "/api/evaluate", "/api/spatial")}
    for base in PREFIXES[1:]:
        app = mod.create_app(base)
        for p, body in root.items():
            assert get(app, base + p).body == body, (base, p)
    again = {p: get(mod.create_app(""), p).body for p in root}
    assert again == root
    ev = json.loads(root["/api/evaluate"])
    assert ev["false_permits"] == 0 and ev["parcels"] == len(mod._PARCELS)
    assert sum(ev["tally"].values()) == ev["parcels"] * ev["uses"]


def test_page_has_no_remaining_root_relative_url_when_a_prefix_is_set():
    mod = load_module()
    for base in PREFIXES[1:]:
        page = get(mod.create_app(base), base + "/").text
        assert scan_root_relative(page, base) == []
        assert "__BASE_PATH_JSON__" not in page and "__SITE_ROOT__" not in page
        for word in ("/api/summary", "/api/evaluate", "/api/spatial", "/static"):
            assert f"'{word}" not in page and f'"{word}' not in page and f"`{word}" not in page
        # the site-root link resolves to the host root whatever the prefix depth
        href = re.search(r'href="([^"]*enterprise-plugins)"', page).group(1)
        assert urlsplit(urljoin(f"http://h{base}/", href)).path == "/enterprise-plugins"
    # the template itself holds no root-relative literal at all: everything is built from the injected base
    assert scan_root_relative(INDEX_HTML.read_text(encoding="utf-8"), "") == []


def test_the_scan_detects_what_it_is_meant_to_detect():
    for bad in ('<script>fetch("/api/x")</script>', '<img src="/static/a.png">', '<a href="/healthz">', "x{background:url(/a.png)}",
                "<script>fetch('/geo2/api/x')</script>", "<script>new EventSource(u)</script>", "<script>location.href=u</script>",
                "<script>const u = '/api/summary'</script>"):
        assert scan_root_relative(bad, "/geo"), bad
    assert scan_root_relative('<script>fetch("/geo/api/x")</script><a href="/geo/">', "/geo") == []


def test_no_cookie_is_set_anywhere():
    mod = load_module()
    for base in PREFIXES:
        app = mod.create_app(base)
        for p in ("/", "/healthz", "/api/summary", "/docs"):
            assert "set-cookie" not in get(app, base + p).headers
    assert "set_cookie" not in APP_PY.read_text().lower()


# --- mutation checks: the contract must notice a prefix removed from any one place ---------------------------------

def _mutate(old: str, new: str) -> types.ModuleType:
    src = APP_PY.read_text(encoding="utf-8")
    assert src.count(old) == 1, old
    return load_module(src.replace(old, new), name="geo_demo_mutant")


MUTATIONS = {
    "router without prefix": ("app.include_router(router, prefix=base)", "app.include_router(router)"),
    "page without prefix": ('app.add_api_route(f"{base}/", index', 'app.add_api_route("/", index'),
    "index.html without prefix": ('f"{base}/index.html"', '"/index.html"'),
    "no redirect from the bare prefix": ("app.add_api_route(base, to_prefix_root", 'app.add_api_route(base + "/gone", to_prefix_root'),
    "redirect drops the prefix": ('RedirectResponse(f"{base}/"', 'RedirectResponse("/"'),
    "redirect carries an origin": ('RedirectResponse(f"{base}/"', 'RedirectResponse(str(request.url.replace(path=base + "/"))'),
    "static mount at the root": ('app.mount(base or "/"', 'app.mount("/"'),
    "docs at the root": ('docs_url=f"{base}/docs"', 'docs_url="/docs"'),
    "openapi at the root": ('openapi_url=f"{base}/openapi.json"', 'openapi_url="/openapi.json"'),
    "oauth redirect at the root": ('swagger_ui_oauth2_redirect_url=f"{base}/docs/oauth2-redirect"',
                                   'swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect"'),
    "page built for the root": ("json.dumps(base_path)", 'json.dumps("")'),
    "site-root link not adjusted": ('"../" * len(base_path.split("/")[1:]) if base_path else "/"', '"/"'),
}


def test_the_contract_passes_on_the_real_module():
    check_contract(load_module(), "/geo")


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_removing_the_prefix_from_one_place_fails_the_contract(name):
    old, new = MUTATIONS[name]
    mutant = _mutate(old, new)
    with pytest.raises(AssertionError):
        check_contract(mutant, "/geo")


def test_a_stray_unprefixed_route_is_caught():
    mod = load_module()
    app = mod.create_app("/geo")
    app.add_api_route("/api/extra", lambda: {"x": 1}, methods=["GET"])
    assert get(app, "/api/extra").status == 200                       # the stray route exists, outside the prefix ...
    mod.create_app = lambda base, _orig=mod.create_app: _with_stray(_orig(base))
    with pytest.raises(AssertionError):                               # ... which the contract forbids
        check_contract(mod, "/geo")


def _with_stray(app):
    app.add_api_route("/api/extra", lambda: {"x": 1}, methods=["GET"])
    return app
