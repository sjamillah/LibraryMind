import logging

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.infrastructure.cache import cache

logger = logging.getLogger(__name__)
_EMBED_CACHE_TTL = 86_400  # 24 h


class EmbeddingService:
    def __init__(self) -> None:
        logger.info("[embedding] loading %s", settings.EMBEDDING_MODEL_NAME)
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("[embedding] model ready")

    def embed(self, text: str) -> list[float]:
        key = cache.make_key("embed", text)
        cached = cache.get(key)
        if cached is not None:
            return cached
        vector: list[float] = self._model.encode(text).tolist()
        cache.set(key, vector, ttl_seconds=_EMBED_CACHE_TTL)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        keys = [cache.make_key("embed", t) for t in texts]
        results: list[list[float] | None] = [cache.get(k) for k in keys]

        misses = [(i, texts[i]) for i, r in enumerate(results) if r is None]
        if misses:  # encode all misses in one call — way faster than one by one
            indices, raw_texts = zip(*misses)
            logger.debug("[embedding] encoding %d misses in one batch", len(raw_texts))
            vectors = self._model.encode(list(raw_texts)).tolist()
            for idx, vector in zip(indices, vectors):
                cache.set(keys[idx], vector, ttl_seconds=_EMBED_CACHE_TTL)
                results[idx] = vector

        return results  # type: ignore[return-value]


embedding_service = EmbeddingService()
