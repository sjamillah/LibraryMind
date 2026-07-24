import logging
import threading

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.exceptions import EmbeddingModelError
from app.infrastructure.cache import cache

logger = logging.getLogger(__name__)
_EMBED_CACHE_TTL = 86_400  # 24 h


class EmbeddingService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        """Load the model on first use rather than at import time.

        Loading eagerly in __init__ meant a failure here (no network on first
        run to download weights, a bad EMBEDDING_MODEL_NAME, disk issues) took
        down the whole app at startup, including /classify and /summarise,
        which never touch embeddings at all. Deferring the load means those
        endpoints keep working regardless, and only the embedding-dependent
        ones (search, ask, chat) fail, and only when actually used.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:  # re-check: another thread may have loaded it first
                    logger.info("[embedding] loading %s", settings.EMBEDDING_MODEL_NAME)
                    try:
                        self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                    except Exception as exc:
                        raise EmbeddingModelError(
                            f"Could not load embedding model '{settings.EMBEDDING_MODEL_NAME}': {exc}. "
                            "Check your network connection (the model downloads on first use) "
                            "and the EMBEDDING_MODEL_NAME setting."
                        ) from exc
                    logger.info("[embedding] model ready")
        return self._model

    def embed(self, text: str) -> list[float]:
        key = cache.make_key("embed", text)
        cached = cache.get(key)
        if cached is not None:
            return cached
        vector: list[float] = self._get_model().encode(text).tolist()
        cache.set(key, vector, ttl_seconds=_EMBED_CACHE_TTL)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        keys = [cache.make_key("embed", t) for t in texts]
        results: list[list[float] | None] = [cache.get(k) for k in keys]

        misses = [(i, texts[i]) for i, r in enumerate(results) if r is None]
        if misses:  # encode all misses in one call — way faster than one by one
            indices, raw_texts = zip(*misses)
            logger.debug("[embedding] encoding %d misses in one batch", len(raw_texts))
            vectors = self._get_model().encode(list(raw_texts)).tolist()
            for idx, vector in zip(indices, vectors):
                cache.set(keys[idx], vector, ttl_seconds=_EMBED_CACHE_TTL)
                results[idx] = vector

        return results  # type: ignore[return-value]


embedding_service = EmbeddingService()
