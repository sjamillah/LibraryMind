from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.infrastructure.rate_limiter import RateLimitExceeded
from app.services.classification_service import classification_service

router = APIRouter(prefix="/classify", tags=["Classify"])


class ClassifyRequest(BaseModel):
    ticket: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        examples=["A patron is requesting a renewal for 'Dune' but the system shows it is unavailable."],
    )


class ClassifyResponse(BaseModel):
    category: str = Field(description="Type of support ticket")
    priority: str = Field(description="Urgency level: low, medium, or high")
    sentiment: str = Field(description="Patron sentiment: positive, neutral, or negative")
    requires_human: bool = Field(description="Whether a human librarian should handle this")


@router.post(
    "/",
    response_model=ClassifyResponse,
    summary="Classify a library support ticket",
    response_description="Structured classification of the ticket",
)
def classify(body: ClassifyRequest) -> ClassifyResponse:
    """
    Submit a raw support ticket and receive a structured classification.

    The AI returns exact JSON keys with constrained values — no freeform text.
    Markdown code fences are stripped before parsing. Malformed AI output raises a 500.
    """
    try:
        result = classification_service.classify(body.ticket)
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before trying again.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ClassifyResponse(
        category=result.category,
        priority=result.priority,
        sentiment=result.sentiment,
        requires_human=result.requires_human,
    )
