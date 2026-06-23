"""
Gateway probe — verifies which request shape the Amali AI gateway accepts.

Run from project root:
    python scripts/probe_gateway.py

Cross-reference with the Swagger docs:
    https://ai-api.amalitech.org/swagger-ui/index.html
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from app.core.config import settings

BASE   = settings.AMALI_GATEWAY_URL.rstrip("/")
KEY    = settings.AMALI_API_KEY
PROMPT = [{"role": "user", "content": "Say hi in one word."}]


def probe(label: str, url: str, payload: dict, headers: dict = None) -> None:
    print(f"\n{'─'*65}")
    print(f"  {label}")
    print(f"  URL: POST {url}")
    print(f"{'─'*65}")
    print("  Body:", json.dumps(payload))
    if headers:
        safe_h = {k: ("Bearer ***" if k == "Authorization" else v) for k, v in headers.items()}
        print("  Headers:", safe_h)
    try:
        r = httpx.post(url, json=payload, headers=headers or {}, timeout=15)
        print(f"\n  → {r.status_code}  {r.text[:400]}")
    except Exception as e:
        print(f"\n  → ERROR: {e}")


endpoint = f"{BASE}/api/v2/public/"

# 1 — correct shape: model + messages + stream, Bearer auth
probe("Correct shape + Bearer auth (expected to work)", endpoint, {
    "model": "gpt-3.5-turbo",
    "messages": PROMPT,
    "stream": False,
}, headers={"Authorization": f"Bearer {KEY}"})

# 2 — same body, no auth header (public endpoint test)
probe("Correct shape — no auth header", endpoint, {
    "model": "gpt-3.5-turbo",
    "messages": PROMPT,
    "stream": False,
})

# 3 — anthropic model + Bearer auth
probe("Anthropic model + Bearer auth", endpoint, {
    "model": "claude-3-5-sonnet-20241022",
    "messages": PROMPT,
    "stream": False,
}, headers={"Authorization": f"Bearer {KEY}"})

# 4 — X-API-Key header instead of Authorization Bearer
probe("X-API-Key header", endpoint, {
    "model": "gpt-3.5-turbo",
    "messages": PROMPT,
    "stream": False,
}, headers={"X-API-Key": KEY})

# 5 — stream: true (some gateways only support streaming)
probe("stream: true + Bearer auth", endpoint, {
    "model": "gpt-3.5-turbo",
    "messages": PROMPT,
    "stream": True,
}, headers={"Authorization": f"Bearer {KEY}"})

print(f"\n\n{'═'*65}")
print("  Share the output with your instructor if all show 401/500.")
print("  The key in AMALI_API_KEY may need to be refreshed from the portal.")
print(f"{'═'*65}\n")
