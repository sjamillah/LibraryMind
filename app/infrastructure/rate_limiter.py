import threading
import time

from app.core.config import settings


class RateLimitExceeded(Exception):
    pass


class RateLimiter:
    """Token bucket rate limiter — thread-safe."""

    def __init__(self, requests_per_minute: int):
        self.capacity = requests_per_minute
        self.tokens = float(requests_per_minute)
        self.refill_rate = requests_per_minute / 60.0  # tokens added per second
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self) -> None:
        with self._lock:
            self._refill()
            if self.tokens < 1:
                raise RateLimitExceeded("Rate limit exceeded, try again shortly")
            self.tokens -= 1


rate_limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)
