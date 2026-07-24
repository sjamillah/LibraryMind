from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.exceptions import EmbeddingModelError
from app.providers.resilient_service import RateLimitExceeded, AllProvidersFailedError
from app.infrastructure.vector_store import vector_store
from app.services.embedding_service import embedding_service
from app.services.rag_engine import rag_engine

router = APIRouter(prefix="/search", tags=["Search"])


# ── POST /search/books ────────────────────────────────────────────────────────

class SearchBooksRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["space exploration adventure"],
    )
    limit: int = Field(5, ge=1, le=20, description="Maximum number of results to return")


class BookSearchResult(BaseModel):
    id: str
    title: str
    author: str
    year: int
    genre: str
    description: str
    relevance_score: float = Field(description="Semantic similarity to the query (0–1)")


@router.post(
    "/books",
    response_model=list[BookSearchResult],
    summary="Semantic search over the book catalogue",
    response_description="Books ranked by semantic similarity to the query",
)
def search_books(body: SearchBooksRequest) -> list[BookSearchResult]:
    """
    Search the catalogue by meaning, not keywords.

    Embeds the query and retrieves the most semantically similar books from
    ChromaDB. Returns all results regardless of relevance threshold — use
    the relevance_score to filter client-side if needed.
    """
    try:
        query_vector = embedding_service.embed(body.query)
        results = vector_store.search(query_vector, top_k=body.limit)
    except EmbeddingModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return [
        BookSearchResult(
            id=r["id"],
            title=r["metadata"]["title"],
            author=r["metadata"]["author"],
            year=int(r["metadata"]["year"]),
            genre=r["metadata"]["genre"],
            description=r["metadata"].get("description", ""),
            relevance_score=round(1.0 - r["distance"], 4),
        )
        for r in results
    ]


# ── POST /search/ask ──────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["What science fiction books do you have about desert planets?"],
    )


class SourceSchema(BaseModel):
    title: str
    author: str
    relevance_score: float = Field(description="Semantic similarity (0–1)")


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceSchema]
    cached: bool = Field(description="True if this response was served from cache")


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="RAG-powered question answering over the catalogue",
    response_description="AI-generated answer grounded in the book catalogue",
)
def ask(body: AskRequest) -> AskResponse:
    """
    Submit a natural-language question and receive an answer grounded in the
    library catalogue. Identical to `/query/` but at the canonical spec path.
    """
    try:
        result = rag_engine.query(body.question)
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before trying again.",
        )
    except (AllProvidersFailedError, EmbeddingModelError) as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceSchema(title=s.title, author=s.author, relevance_score=s.relevance_score)
            for s in result.sources
        ],
        cached=result.cached,
    )
