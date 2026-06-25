from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.providers.resilient_service import ResilientAIService, build_service

logger = logging.getLogger(__name__)

_TEMPERATURE = 0.2

_SYSTEM_PROMPT = """\
You are a book review analyst for a library system.

You will receive a collection of patron reviews for a single book.
Treat all reviews holistically — produce one unified analysis, not a per-review summary.

Return ONLY a JSON object with these exact keys:
- "summary": a single paragraph capturing the overall patron experience
- "overall_sentiment": one of "positive", "mixed", "negative"
- "key_themes": a list of 2–4 recurring themes that appear across the reviews
- "recommended": true if the overall reception is positive, false otherwise

Do not wrap the JSON in code fences or add any explanation.\
"""


@dataclass
class SummarisationResult:
    summary: str
    overall_sentiment: str
    key_themes: list[str] = field(default_factory=list)
    recommended: bool = False


class SummarisationService:
    def __init__(self, ai_service: ResilientAIService | None = None) -> None:
        self._service = ai_service or build_service()

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
        return SummarisationResult(
            summary=data["summary"],
            overall_sentiment=data["overall_sentiment"],
            key_themes=list(data.get("key_themes", [])),
            recommended=bool(data["recommended"]),
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
        raise ValueError(
            f"Summarisation response was not valid JSON.\n"
            f"Parse error: {exc}\n"
            f"Raw response:\n{raw}"
        ) from exc


summarisation_service = SummarisationService()
