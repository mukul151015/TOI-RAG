#!/usr/bin/env python3
"""
Auth helper for promptfoo eval.

Logs in to the TOI RAG API and prints the toi_rag_session cookie value to stdout.
Used by run_eval.sh to inject the cookie into promptfoo HTTP provider requests.

Usage:
    python eval/scripts/get_session_cookie.py

Environment variables:
    TOI_BASE_URL        API base URL (default: http://localhost:8000)
    TOI_EVAL_EMAIL      Login email  (default: eval@toi-rag.test)
    TOI_EVAL_PASSWORD   Login password
"""
import os
import sys

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

BASE_URL = os.getenv("TOI_BASE_URL", "http://localhost:8000")
EMAIL    = os.getenv("TOI_EVAL_EMAIL", "eval@toi-rag.test")
PASSWORD = os.getenv("TOI_EVAL_PASSWORD", "")

if not PASSWORD:
    print("ERROR: TOI_EVAL_PASSWORD env var is required", file=sys.stderr)
    sys.exit(1)

try:
    resp = httpx.post(
        f"{BASE_URL}/auth/login",
        json={"email": EMAIL, "password": PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
except httpx.HTTPStatusError as e:
    print(f"ERROR: login failed ({e.response.status_code}): {e.response.text}", file=sys.stderr)
    sys.exit(1)
except httpx.RequestError as e:
    print(f"ERROR: could not connect to {BASE_URL}: {e}", file=sys.stderr)
    sys.exit(1)

# httpx stores cookies automatically; extract the token value
token = resp.cookies.get("toi_rag_session")

if not token:
    # Fallback: parse raw Set-Cookie header
    for part in resp.headers.get("set-cookie", "").split(";"):
        part = part.strip()
        if part.startswith("toi_rag_session="):
            token = part.split("=", 1)[1]
            break

if not token:
    print("ERROR: could not extract toi_rag_session cookie from response", file=sys.stderr)
    sys.exit(1)

print(token)
