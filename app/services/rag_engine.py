from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core.config import settings
from app.infrastructure.cache import cache
from app.infrastructure.rate_limiter import rate_limiter
from app.infrastructure.vector_store import vector_store
from app.providers.resilient_service import ResilientAIService, ai_service as _default_ai_service
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

_TOP_K = 5
_DISTANCE_THRESHOLD: float = settings.RAG_RELEVANCE_THRESHOLD

# Phrases that indicate the AI found nothing useful in the provided context.
# When matched, sources are cleared — returning sources alongside a refusal
# implies relevance the AI itself just denied.
_REFUSAL_PHRASES = (
    "couldn't find",
    "could not find",
    "no books",
    "no titles",
    "no results",
    "not in the catalogue",
    "not in our catalogue",
    "not available in",
    "there are no",
    "don't have any",
    "do not have any",
    "nothing relevant",
)

# Without the explicit "don't use training data" rule, the model fills gaps
# with books it knows from training — which aren't in our catalogue.
_SYSTEM_PROMPT = """\
You are LibraryMind, a library assistant helping patrons discover books.

Your answers must be grounded entirely in the catalogue entries provided in the context.

Rules you must follow without exception:
- Cite every book you mention by its exact title and author as listed in the catalogue.
- Do not reference, suggest, or invent any book that does not appear in the context.
- Do not draw on knowledge from your training data to supplement or replace the catalogue.
- If the catalogue does not contain enough information to answer the question fully, \
say so clearly and honestly — it is better to acknowledge limited knowledge than to \
fill gaps with fabrications.\
"""

_NO_RESULTS_MESSAGE = (
    "I couldn't find any books in our catalogue that are relevant to your question. "
    "Try asking about a specific genre, theme, author, or mood."
)


@dataclass
class Source:
    title: str
    author: str
    relevance_score: float


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source] = field(default_factory=list)
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [
                {
                    "title": s.title,
                    "author": s.author,
                    "relevance_score": s.relevance_score,
                }
                for s in self.sources
            ],
            "cached": self.cached,
        }


class RAGEngine:
    def __init__(self, ai_service: ResilientAIService | None = None) -> None:
        self._service = ai_service or _default_ai_service

    def query(self, question: str) -> RAGResponse:
        """Embed, search, filter, generate — returns answer + sources + cache flag."""
        cache_key = cache.make_key("rag", question)

        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("[rag] cache hit")
            return RAGResponse(
                answer=cached["answer"],
                sources=[Source(**s) for s in cached["sources"]],
                cached=True,
            )

        rate_limiter.acquire()

        question_vector = embedding_service.embed(question)
        results = vector_store.search(question_vector, top_k=_TOP_K)
        relevant = [r for r in results if r["distance"] <= _DISTANCE_THRESHOLD]

        if not relevant:
            logger.info("[rag] no results passed the relevance threshold")
            return RAGResponse(answer=_NO_RESULTS_MESSAGE, sources=[], cached=False)

        context = _build_context(relevant)
        prompt = f"Patron question: {question}\n\nCatalogue context:\n{context}"

        answer = self._service.generate(prompt=prompt, system=_SYSTEM_PROMPT)

        if _is_refusal(answer) and not _mentions_any_title(answer, relevant):
            logger.info("[rag] AI declined — clearing sources")
            cache.set(cache_key, {"answer": answer, "sources": []})
            return RAGResponse(answer=answer, sources=[], cached=False)

        sources = [
            Source(
                title=r["metadata"]["title"],
                author=r["metadata"]["author"],
                relevance_score=round(1.0 - r["distance"], 4),  # flip distance to score
            )
            for r in relevant
        ]

        cache.set(cache_key, {
            "answer": answer,
            "sources": [
                {
                    "title": s.title,
                    "author": s.author,
                    "relevance_score": s.relevance_score,
                }
                for s in sources
            ],
        })

        logger.info("[rag] answered using %d sources", len(sources))
        return RAGResponse(answer=answer, sources=sources, cached=False)


def _is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def _mentions_any_title(answer: str, relevant: list[dict]) -> bool:
    """True if the answer actually names one of the retrieved books.

    A refusal phrase like "there are no" can appear in an otherwise good
    answer ("there are no OTHER books on this narrower theme") after the
    model has already cited real titles. Checking for an actual citation
    before clearing sources avoids treating that as a full refusal.
    """
    lower = answer.lower()
    return any(r["metadata"]["title"].lower() in lower for r in relevant)


def _build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        block = (
            f"{i}. \"{m['title']}\" by {m['author']} "
            f"({m['year']}, {m['genre']})\n"
            f"   {m.get('description', '')}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


rag_engine = RAGEngine()
