# Geospatial / zoning demo

A small FastAPI application (`app.py`) that serves a page and a JSON API over a harvested sample of zoning parcels. The
dispositions are computed by the deterministic resolver in `app.py`; spatial relations come from the runtime's geometry
engine (`context_runtime.geospatial`). Nothing is stored and nothing calls out.

```bash
PYTHONPATH=../.. uvicorn app:app --host 127.0.0.1 --port 8098
```

## Serving under a URL prefix

By default the app serves at the root of its origin. Set `PUBLIC_BASE_PATH` to serve it under a path prefix instead, for
example when one host routes several applications by path and does not rewrite paths on the way in:

```bash
PUBLIC_BASE_PATH=/geo PYTHONPATH=../.. uvicorn app:app --host 127.0.0.1 --port 8098
```

| | `PUBLIC_BASE_PATH` empty (default) | `PUBLIC_BASE_PATH=/geo` |
|---|---|---|
| page | `/` | `/geo/` |
| API | `/api/summary`, `/api/parcels`, `/api/evaluate`, `/api/spatial` | `/geo/api/...` |
| health | `/healthz` | `/geo/healthz` |
| interactive docs | `/docs`, `/redoc`, `/openapi.json` | `/geo/docs`, `/geo/redoc`, `/geo/openapi.json` |
| anything else | 404 | 404 (including the root and `/healthz`) |

How it behaves:

- No proxy rewrite is needed or assumed: every route is registered under the prefix, so the proxy forwards the request
  path unchanged. A request outside the prefix is a 404 (the root path belongs to whatever else is on the host).
- `/geo` (no trailing slash) redirects to `/geo/` with status 307. The `Location` is a path only (`/geo/`, query string
  kept), never an absolute URL, so it cannot carry the internal host name.
- The page builds every request URL from the prefix that the server injects into it (`const BASE = "/geo"`), so it
  works at any depth. Its single link to the host's own root (`Enterprise plugins`) is written relative to the prefix
  depth (`../enterprise-plugins` under `/geo`). With no prefix the page requests the same root paths as before.
- The app sets no cookies. If one is ever added it must use `Path=<prefix>`.
- The data and the zoning results are the same under every prefix; only the paths change.
- Health: the app answers health only at `<prefix>/healthz` when a prefix is set, and only at `/healthz` when it is not.
  A Kubernetes probe (or any other check) must therefore use `<prefix>/healthz`. The root `/healthz` is deliberately
  not served under a prefix, so that it does not collide with another application on the same host.

The value must be empty or a path that starts with `/`, has no trailing slash, and consists of segments made only of
letters, digits and `. _ ~ -`; empty segments, `.` and `..` are rejected. An invalid value stops the application at start
(`uvicorn` exits with an `InvalidBasePath` error naming `PUBLIC_BASE_PATH`) instead of serving a wrong path.

## Container image

`Dockerfile.aks` reads `PUBLIC_BASE_PATH` at run time (set it in the pod or container environment; it is not baked into
the image). Its `HEALTHCHECK` runs `healthcheck.py`, which requests `<PUBLIC_BASE_PATH>/healthz` on `127.0.0.1:$PORT`, so
the container health follows the same prefix. A Kubernetes `httpGet` probe takes the path explicitly, for example
`/geo/healthz`.

The image copies `app.py` from the pinned source release named in `.github/workflows/build-geo-image.yml`; the prefix
support is in the image only once that pin names a release that contains it.

## Tests

```bash
python -m pytest tests/test_geo_demo_base_path.py tests/test_geo_demo_healthcheck.py -q
```

`test_geo_demo_base_path.py` drives the app in-process (ASGI) with an empty prefix, `/geo` and `/shared/geo`: every URL
found in the served page resolves under the prefix and answers, the API results are identical, paths outside the prefix
are 404, the redirect is path-only, invalid values are refused, and a scan of the page source finds no root-relative URL.
It also runs the same checks against copies of `app.py` with the prefix removed from one place at a time, and requires
each to fail. It needs the `control-plane` extra (FastAPI) and is skipped without it.
