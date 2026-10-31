"""Container health probe (HEALTHCHECK CMD). Stdlib only; exits 0 when /healthz answers 200."""
import os
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('PORT', '8098')}/healthz", timeout=4) as resp:
        sys.exit(0 if resp.status == 200 else 1)
except Exception:
    sys.exit(1)
