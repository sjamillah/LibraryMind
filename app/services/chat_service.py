from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from app.core.config import settings
from app.infrastructure.vector_store import vector_store
from app.providers.resilient_service import ResilientAIService, ai_service as _default_ai_service
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

_HISTORY_WINDOW = 10
_TOP_K = 5
_DISTANCE_THRESHOLD: float = settings.RAG_RELEVANCE_THRESHOLD

_SYSTEM_PROMPT = """\
You are LibraryMind, a friendly and knowledgeable library assistant.
You help patrons discover books through natural conversation.
You remember everything discussed earlier in this conversation — use that context
to answer follow-up questions accurately.
Ground all book recommendations in the catalogue context provided.
Do not invent or reference books that are not in the context.\
"""

Message = dict  # {"role": "user" | "assistant", "content": str}


class ConversationStore:
    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = {}

    def get(self, conversation_id: str) -> list[Message]:
        return self._store.get(conversation_id, [])

    def append(self, conversation_id: str, message: Message) -> None:
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store[conversation_id].append(message)


@dataclass
class ChatSource:
    title: str
    author: str
    relevance_score: float


@dataclass
class ChatResponse:
    conversation_id: str
    reply: str
    sources: list[ChatSource] = field(default_factory=list)


# Module-level singleton — shared across all requests
conversation_store = ConversationStore()


class ChatService:
    def __init__(
        self,
        ai_service: ResilientAIService | None = None,
        store: ConversationStore | None = None,
    ) -> None:
        self._service = ai_service or _default_ai_service
        self._store = store if store is not None else conversation_store

    def chat(self, message: str, conversation_id: str | None = None) -> ChatResponse:
        """Process one turn of conversation. Generates a new ID when none is provided."""
        cid = conversation_id if conversation_id is not None else str(uuid.uuid4())
        history = self._store.get(cid)

        # Call vector_store directly — bypassing rag_engine.query() which caches by
        # question string and would serve one user's conversation context to another.
        question_vector = embedding_service.embed(message)
        results = vector_store.search(question_vector, top_k=_TOP_K)
        relevant = [r for r in results if r["distance"] <= _DISTANCE_THRESHOLD]

        prompt = _build_prompt(history, relevant, message)

        reply = self._service.generate(prompt=prompt, system=_SYSTEM_PROMPT)

        self._store.append(cid, {"role": "user", "content": message})
        self._store.append(cid, {"role": "assistant", "content": reply})

        sources = [
            ChatSource(
                title=r["metadata"]["title"],
                author=r["metadata"]["author"],
                relevance_score=round(1.0 - r["distance"], 4),
            )
            for r in relevant
        ]

        logger.info(
            "[chat] cid=%s turn=%d sources=%d",
            cid,
            len(history) // 2 + 1,
            len(sources),
        )
        return ChatResponse(conversation_id=cid, reply=reply, sources=sources)


def _build_prompt(
    history: list[Message],
    relevant: list[dict],
    message: str,
) -> str:
    parts: list[str] = []

    for msg in history[-_HISTORY_WINDOW:]:
        label = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{label}: {msg['content']}")

    if relevant:
        context_block = _build_context(relevant)
        parts.append(f"\nCatalogue context:\n{context_block}")

    parts.append(f"User: {message}")
    return "\n".join(parts)


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


chat_service = ChatService()
