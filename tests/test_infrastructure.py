import time
import pytest
from unittest.mock import patch

from app.infrastructure.cache import Cache
from app.infrastructure.rate_limiter import RateLimiter, RateLimitExceeded
from app.infrastructure.usage_tracker import UsageTracker


# ── Cache ─────────────────────────────────────────────────────────────────────

class TestCache:
    def setup_method(self):
        self.cache = Cache.__new__(Cache)
        self.cache._redis_client = None
        self.cache._memory_store = {}

    def test_miss_returns_none(self):
        assert self.cache.get("nonexistent") is None

    def test_hit_returns_stored_value(self):
        self.cache.set("key", "hello")
        assert self.cache.get("key") == "hello"

    def test_stores_any_json_serialisable_value(self):
        self.cache.set("k", {"answer": 42, "items": [1, 2, 3]})
        assert self.cache.get("k") == {"answer": 42, "items": [1, 2, 3]}

    def test_overwrite_replaces_value(self):
        self.cache.set("k", "first")
        self.cache.set("k", "second")
        assert self.cache.get("k") == "second"

    def test_expired_entry_returns_none(self):
        with patch("app.infrastructure.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            self.cache.set("k", "value", ttl_seconds=60)

            mock_time.time.return_value = 1061.0  # 1 second past TTL
            assert self.cache.get("k") is None

    def test_entry_within_ttl_is_returned(self):
        with patch("app.infrastructure.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            self.cache.set("k", "value", ttl_seconds=60)

            mock_time.time.return_value = 1059.0  # still within TTL
            assert self.cache.get("k") == "value"

    def test_expired_entry_is_deleted_from_store(self):
        with patch("app.infrastructure.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            self.cache.set("k", "value", ttl_seconds=10)
            mock_time.time.return_value = 1020.0
            self.cache.get("k")
        assert "k" not in self.cache._memory_store

    def test_make_key_is_deterministic(self):
        assert Cache.make_key("prompt", "system") == Cache.make_key("prompt", "system")

    def test_make_key_different_inputs_produce_different_keys(self):
        assert Cache.make_key("a", "b") != Cache.make_key("b", "a")

    def test_make_key_returns_hex_string(self):
        key = Cache.make_key("hello")
        assert len(key) == 64
        int(key, 16)  # raises if not valid hex


# ── RateLimiter ───────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_acquire_succeeds_within_capacity(self):
        limiter = RateLimiter(5)
        for _ in range(5):
            limiter.acquire()  # should not raise

    def test_acquire_raises_when_tokens_exhausted(self):
        limiter = RateLimiter(3)
        for _ in range(3):
            limiter.acquire()
        with pytest.raises(RateLimitExceeded):
            limiter.acquire()

    def test_tokens_refill_over_time(self):
        limiter = RateLimiter(60)  # 1 token per second refill rate

        # Drain all tokens
        for _ in range(60):
            limiter.acquire()
        with pytest.raises(RateLimitExceeded):
            limiter.acquire()

        # Advance time by 1 second — should refill 1 token
        with patch("app.infrastructure.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = limiter.last_refill + 1.0
            limiter.acquire()  # should succeed after 1s of refill

    def test_rate_limit_exceeded_message(self):
        limiter = RateLimiter(1)
        limiter.acquire()
        with pytest.raises(RateLimitExceeded, match="Rate limit exceeded"):
            limiter.acquire()

    def test_capacity_is_respected_as_ceiling(self):
        """Tokens should never exceed capacity even after a long idle period."""
        limiter = RateLimiter(10)
        with patch("app.infrastructure.rate_limiter.time") as mock_time:
            mock_time.monotonic.return_value = limiter.last_refill + 9999.0
            limiter._refill()
        assert limiter.tokens <= limiter.capacity


# ── UsageTracker ──────────────────────────────────────────────────────────────

class TestUsageTracker:
    def setup_method(self):
        self.tracker = UsageTracker()

    def test_record_returns_usage_record(self):
        record = self.tracker.record("openai", "gpt-3.5-turbo", "hello", "world")
        assert record.provider == "openai"
        assert record.model == "gpt-3.5-turbo"

    def test_record_counts_nonzero_tokens(self):
        record = self.tracker.record("openai", "gpt-3.5-turbo", "hello world", "response text")
        assert record.prompt_tokens > 0
        assert record.completion_tokens > 0

    def test_record_calculates_nonzero_cost(self):
        record = self.tracker.record("openai", "gpt-3.5-turbo", "hello", "world")
        assert record.cost_usd > 0

    def test_total_requests_today_increments(self):
        assert self.tracker.total_requests_today() == 0
        self.tracker.record("openai", "gpt-3.5-turbo", "a", "b")
        assert self.tracker.total_requests_today() == 1
        self.tracker.record("openai", "gpt-3.5-turbo", "c", "d")
        assert self.tracker.total_requests_today() == 2

    def test_total_cost_today_accumulates(self):
        self.tracker.record("openai", "gpt-3.5-turbo", "hello", "world")
        first = self.tracker.total_cost_today()
        self.tracker.record("openai", "gpt-3.5-turbo", "hello again", "another response")
        assert self.tracker.total_cost_today() > first

    def test_uses_default_pricing_for_unknown_model(self):
        record = self.tracker.record("custom", "unknown-model", "hello", "world")
        assert record.cost_usd > 0

    def test_records_from_yesterday_excluded_from_today(self):
        from datetime import datetime, timezone, timedelta

        self.tracker.record("openai", "gpt-3.5-turbo", "hello", "world")
        assert self.tracker.total_requests_today() == 1

        # Backdate the only record to yesterday
        yesterday_ts = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
        self.tracker._records[0] = self.tracker._records[0].__class__(
            **{**self.tracker._records[0].__dict__, "timestamp": yesterday_ts}
        )
        assert self.tracker.total_requests_today() == 0
        assert self.tracker.total_cost_today() == 0.0
