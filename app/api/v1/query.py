from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.providers.resilient_service import RateLimitExceeded, AllProvidersFailedError
from app.services.rag_engine import rag_engine

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What science fiction books do you have about desert planets?"],
    )


class SourceSchema(BaseModel):
    title: str
    author: str
    relevance_score: float = Field(
        description="Semantic similarity to the question (0–1, higher is more relevant)"
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceSchema]
    cached: bool = Field(description="True if this response was served from cache")


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Ask the library assistant a question",
    response_description="AI-generated answer grounded in the book catalogue",
)
def ask(body: QueryRequest) -> QueryResponse:
    """
    Submit a natural-language question and receive an answer grounded in the
    library catalogue.

    - The engine embeds the question, searches ChromaDB for semantically
      similar books, and sends only the relevant context to the AI.
    - If no books are relevant, a polite refusal is returned — the AI is
      never asked to fabricate.
    - Repeated identical questions are served from cache instantly.
    """
    try:
        result = rag_engine.query(body.question)
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before trying again.",
        )
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceSchema(
                title=s.title,
                author=s.author,
                relevance_score=s.relevance_score,
            )
            for s in result.sources
        ],
        cached=result.cached,
    )
