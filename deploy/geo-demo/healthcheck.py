"""Container health probe (HEALTHCHECK CMD). Stdlib only; exits 0 when <PUBLIC_BASE_PATH>/healthz answers 200.

PUBLIC_BASE_PATH is the same variable the app reads (empty = served at the root). A value the app would refuse
(it does not start) fails the probe as well, rather than probing a different path.
"""
import os
import re
import sys
import urllib.request

base = os.environ.get("PUBLIC_BASE_PATH", "")
# Same rule as app.validate_base_path: empty, or /seg[/seg...] of unreserved characters, no trailing slash, no dot segment.
if base and not (len(base) <= 128 and re.fullmatch(r"(/[A-Za-z0-9._~-]+)+", base) and ".." not in base
                 and "." not in base.split("/")[1:]):
    sys.exit(1)
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('PORT', '8098')}{base}/healthz", timeout=4) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
