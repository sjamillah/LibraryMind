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
    summary: str = Field(description="Holistic paragraph summarising the overall patron experience")
    overall_sentiment: str = Field(description="positive, mixed, or negative")
    key_themes: list[str] = Field(description="2–4 recurring themes across all reviews")
    recommended: bool = Field(description="True if the overall reception is positive")


@router.post(
    "/",
    response_model=SummariseResponse,
    summary="Summarise a collection of book reviews",
    response_description="Holistic AI-generated summary of all reviews",
)
def summarise(body: SummariseRequest) -> SummariseResponse:
    """
    Submit a list of patron reviews and receive a single holistic summary.

    The AI is explicitly instructed to treat all reviews as a whole — not summarise
    each individually. Markdown code fences are stripped before parsing.
    """
    try:
        result = summarisation_service.summarise(body.reviews)
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before trying again.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SummariseResponse(
        summary=result.summary,
        overall_sentiment=result.overall_sentiment,
        key_themes=result.key_themes,
        recommended=result.recommended,
    )
