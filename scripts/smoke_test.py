"""
Integration smoke test — verifies the full provider stack against the real gateway.

Run from project root:
    python scripts/smoke_test.py                    (Linux / WSL)
    .venv\Scripts\python scripts\smoke_test.py      (Windows)

Makes real API calls. To test provider fallback: set your primary key to a
garbage value in .env, restart, and confirm the secondary provider answers.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

from app.providers.resilient_service import build_service
from app.infrastructure.usage_tracker import usage_tracker
from app.infrastructure.rate_limiter import RateLimiter, RateLimitExceeded


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


# ── Part 1: basic generate ────────────────────────────────────────────────────

section("Test 1: basic generate")
service = build_service()
response = service.generate("Say hello in one sentence.")
print(f"Response: {response}")
assert isinstance(response, str) and len(response) > 0

section("Test 2: with system prompt")
response = service.generate(
    prompt="What can you help me with?",
    system="You are a library assistant. Keep your answer to one sentence.",
)
print(f"Response: {response}")
assert isinstance(response, str) and len(response) > 0


# ── Part 2 — Infrastructure tests ────────────────────────────────────────────

# Test 3: Cache — second identical call must not add a new usage record
section("Test 3: Cache — second call served from cache")
records_before = usage_tracker.total_requests_today()
r1 = service.generate("List three classic novels in one sentence.")
records_after_first = usage_tracker.total_requests_today()
assert records_after_first == records_before + 1, "First call should add a record"

r2 = service.generate("List three classic novels in one sentence.")  # identical prompt
records_after_second = usage_tracker.total_requests_today()
assert records_after_second == records_after_first, (
    f"Second call should be a cache hit (no new record). "
    f"Records went from {records_after_first} to {records_after_second}."
)
assert r1 == r2, "Cached response must match the original"
print(f"  First call  → API hit, records: {records_after_first}")
print(f"  Second call → cache hit, records: {records_after_second} ✓")


# Test 4: Rate limiter — fires after capacity is exhausted
# We test the RateLimiter directly to avoid 20+ real API calls.
section("Test 4: Rate limiter — fires after N requests")
test_limiter = RateLimiter(requests_per_minute=5)

fired_at = None
for i in range(20):
    try:
        test_limiter.acquire()
    except RateLimitExceeded:
        fired_at = i
        break

assert fired_at == 5, f"Expected RateLimitExceeded at call 6 (index 5), got {fired_at}"
print(f"  RateLimitExceeded raised at call {fired_at + 1} (capacity=5) ✓")


# Test 5: Cost tracking — non-zero after real calls
section("Test 5: Usage tracker — cost is non-zero after real calls")
cost = usage_tracker.total_cost_today()
requests = usage_tracker.total_requests_today()
assert cost > 0, f"Expected non-zero cost, got {cost}"
assert requests > 0
print(f"  Requests today : {requests}")
print(f"  Cost today     : ${cost:.6f} ✓")


print("\n✓ All integration tests passed.")
