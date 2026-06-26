from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.infrastructure.rate_limiter import RateLimitExceeded
from app.services.summarisation_service import summarisation_service

router = APIRouter(prefix="/summarise", tags=["Summarise"])


class SummariseRequest(BaseModel):
    reviews: list[str] = Field(
        ...,
        min_length=1,
        examples=[["Loved the world-building.", "Dense but rewarding read.", "A timeless classic."]],
    )


class SummariseResponse(BaseModel):
    overall_sentiment: str = Field(description="positive, mixed, or negative")
    average_rating: float = Field(description="Estimated average rating from 1.0 to 5.0")
    key_themes: list[str] = Field(description="2–4 recurring themes across all reviews")
    praise: list[str] = Field(description="Common points of praise mentioned across reviews")
    criticism: list[str] = Field(description="Common points of criticism (empty if none)")
    recommendation: str = Field(description="One-sentence recommendation for patrons")


@router.post(
    "/reviews",
    response_model=SummariseResponse,
    summary="Summarise a collection of book reviews",
    response_description="Holistic AI-generated summary of all reviews",
)
def summarise(body: SummariseRequest) -> SummariseResponse:
    """
    Submit a list of patron reviews and receive a single holistic summary.

    The AI treats all reviews as a whole — it is explicitly instructed not to
    summarise each individually. Returns sentiment, rating estimate, themes,
    praise, criticism, and a one-sentence recommendation.
    """
    try:
        result = summarisation_service.summarise(body.reviews)
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before trying again.",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SummariseResponse(
        overall_sentiment=result.overall_sentiment,
        average_rating=result.average_rating,
        key_themes=result.key_themes,
        praise=result.praise,
        criticism=result.criticism,
        recommendation=result.recommendation,
    )
