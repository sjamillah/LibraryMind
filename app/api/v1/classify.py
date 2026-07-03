from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.providers.resilient_service import RateLimitExceeded, AllProvidersFailedError
from app.services.classification_service import classification_service

router = APIRouter(prefix="/classify", tags=["Classify"])


class ClassifyRequest(BaseModel):
    ticket: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        examples=["My library card isn't working at the self-checkout and I'm very frustrated."],
    )


class ClassifyResponse(BaseModel):
    category: str = Field(description="account | borrowing | technical | complaint | suggestion | general")
    priority: str = Field(description="low | medium | high | urgent")
    sentiment: str = Field(description="positive | neutral | negative")
    suggested_department: str = Field(description="Department best suited to handle this ticket")
    summary: str = Field(description="One-sentence description of what the patron needs")


@router.post(
    "/ticket",
    response_model=ClassifyResponse,
    summary="Classify a library support ticket",
    response_description="Structured classification of the ticket",
)
def classify(body: ClassifyRequest) -> ClassifyResponse:
    """
    Submit a raw support ticket and receive a structured classification.

    Returns category, priority, sentiment, suggested routing department,
    and a one-sentence summary. The AI uses low temperature for consistent,
    deterministic outputs. Markdown code fences are stripped before parsing.
    """
    try:
        result = classification_service.classify(body.ticket)
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail="Too many requests — please wait a moment before trying again.",
        )
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ClassifyResponse(
        category=result.category,
        priority=result.priority,
        sentiment=result.sentiment,
        suggested_department=result.suggested_department,
        summary=result.summary,
    )
