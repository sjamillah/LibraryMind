from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.exceptions import EmbeddingModelError
from app.providers.resilient_service import RateLimitExceeded, AllProvidersFailedError
from app.services.chat_service import chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["What science fiction books do you have?"],
    )
    conversation_id: str | None = Field(
        None,
        description="Omit to start a new conversation; include to continue an existing one.",
    )


class ChatSourceSchema(BaseModel):
    title: str
    author: str
    relevance_score: float = Field(description="Semantic similarity (0–1, higher is more relevant)")


class ChatResponseSchema(BaseModel):
    conversation_id: str = Field(description="Use this ID in follow-up messages")
    reply: str
    sources: list[ChatSourceSchema]


@router.post(
    "/",
    response_model=ChatResponseSchema,
    summary="Send a message to the library chatbot",
    response_description="AI reply with the conversation ID to continue the session",
)
def chat(body: ChatRequest) -> ChatResponseSchema:
    """
    Send a message and receive a reply grounded in the book catalogue.

    Omit `conversation_id` to start a new session — the server generates and
    returns one. Include it on follow-up messages to continue the conversation.
    The assistant remembers the last 10 exchanges and uses them to answer
    follow-up questions correctly.
    """
    try:
        result = chat_service.chat(
            message=body.message,
            conversation_id=body.conversation_id,
        )
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before trying again.",
        )
    except (AllProvidersFailedError, EmbeddingModelError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponseSchema(
        conversation_id=result.conversation_id,
        reply=result.reply,
        sources=[
            ChatSourceSchema(
                title=s.title,
                author=s.author,
                relevance_score=s.relevance_score,
            )
            for s in result.sources
        ],
    )
