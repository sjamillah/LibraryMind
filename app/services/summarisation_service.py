from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.core.exceptions import InvalidAIResponseError
from app.providers.resilient_service import ResilientAIService, ai_service as _default_ai_service

logger = logging.getLogger(__name__)

_TEMPERATURE = 0.2

_SYSTEM_PROMPT = """\
You are a book review analyst for a library system.

You will receive a collection of patron reviews for a single book.
Treat all reviews holistically — produce one unified analysis, not a per-review summary.

Return ONLY a JSON object with these exact keys:
- "overall_sentiment": one of "positive", "mixed", "negative"
- "average_rating": estimated average rating from 1.0 to 5.0 based on the tone of the reviews
- "key_themes": a list of 2–4 recurring themes that appear across the reviews
- "praise": a list of 1–3 common points of praise mentioned across the reviews
- "criticism": a list of 1–3 common points of criticism (empty list if none)
- "recommendation": a single sentence recommending whether patrons should read this book

Do not wrap the JSON in code fences or add any explanation.\
"""


@dataclass
class SummarisationResult:
    overall_sentiment: str
    average_rating: float
    key_themes: list[str] = field(default_factory=list)
    praise: list[str] = field(default_factory=list)
    criticism: list[str] = field(default_factory=list)
    recommendation: str = ""


class SummarisationService:
    def __init__(self, ai_service: ResilientAIService | None = None) -> None:
        self._service = ai_service or _default_ai_service

    def summarise(self, reviews: list[str]) -> SummarisationResult:
        """Summarise a list of patron reviews into a single holistic analysis."""
        if not reviews:
            raise ValueError("At least one review is required.")

        reviews_block = "\n\n".join(
            f"Review {i}:\n{r}" for i, r in enumerate(reviews, 1)
        )
        raw = self._service.generate(
            prompt=f"Reviews:\n{reviews_block}",
            system=_SYSTEM_PROMPT,
            temperature=_TEMPERATURE,
        )
        logger.info("[summarise] raw response length=%d", len(raw))
        data = _parse_json(raw)
        _require_keys(data, ("overall_sentiment", "average_rating", "recommendation"), raw)
        try:
            average_rating = float(data["average_rating"])
        except (TypeError, ValueError) as exc:
            raise InvalidAIResponseError(
                f"Summarisation response had a non-numeric average_rating: {data['average_rating']!r}.\n"
                f"Raw response:\n{raw}"
            ) from exc
        return SummarisationResult(
            overall_sentiment=data["overall_sentiment"],
            average_rating=average_rating,
            key_themes=list(data.get("key_themes", [])),
            praise=list(data.get("praise", [])),
            criticism=list(data.get("criticism", [])),
            recommendation=data["recommendation"],
        )


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def _parse_json(raw: str) -> dict:
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise InvalidAIResponseError(
            f"Summarisation response was not valid JSON.\n"
            f"Parse error: {exc}\n"
            f"Raw response:\n{raw}"
        ) from exc


def _require_keys(data: dict, keys: tuple[str, ...], raw: str) -> None:
    """Valid JSON can still be missing a field the schema requires — a plain
    dict[key] would raise an unhandled KeyError that leaks a raw Python
    exception message straight into the API response instead of a clear one."""
    missing = [k for k in keys if k not in data]
    if missing:
        raise InvalidAIResponseError(
            f"Summarisation response was missing required field(s): {', '.join(missing)}.\n"
            f"Raw response:\n{raw}"
        )


summarisation_service = SummarisationService()
