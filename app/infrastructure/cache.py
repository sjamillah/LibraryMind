import json
import hashlib
import logging
import time
from typing import Any, Optional

try:
    import redis
except ImportError:
    redis = None

from app.core.config import settings

logger = logging.getLogger(__name__)


class Cache:
    def __init__(self):
        self._redis_client = None
        self._memory_store: dict[str, tuple[Any, float]] = {}

        if settings.REDIS_URL and redis is not None:
            try:
                client = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
                client.ping()
                self._redis_client = client
            except Exception as e:
                logger.warning("[cache] Redis unavailable, falling back to in-memory: %s", e)
                self._redis_client = None

    @staticmethod
    def make_key(*parts: str) -> str:
        """SHA-256 hash of the joined parts — safe to use as a Redis or dict key."""
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if self._redis_client is not None:
            try:
                raw = self._redis_client.get(key)
                return json.loads(raw) if raw is not None else None
            except Exception as e:
                logger.warning("[cache] Redis get failed, treating as miss: %s", e)
                return None

        entry = self._memory_store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry < time.time():
            del self._memory_store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        if self._redis_client is not None:
            try:
                self._redis_client.set(key, json.dumps(value), ex=ttl_seconds)
                return
            except Exception as e:
                logger.warning("[cache] Redis set failed, falling back to in-memory: %s", e)
        self._memory_store[key] = (value, time.time() + ttl_seconds)


cache = Cache()
