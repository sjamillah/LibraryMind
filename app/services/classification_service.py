from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.providers.resilient_service import ResilientAIService, build_service

logger = logging.getLogger(__name__)

_TEMPERATURE = 0.1

_SYSTEM_PROMPT = """\
You are a library support ticket classifier.

Classify the ticket into structured JSON with these exact keys and valid values:
- "category": one of "book_request", "renewal", "complaint", "inquiry", "damage_report"
- "priority": one of "low", "medium", "high"
- "sentiment": one of "positive", "neutral", "negative"
- "requires_human": true or false

Return ONLY the JSON object. Do not wrap it in code fences or add any explanation.\
"""


@dataclass
class ClassificationResult:
    category: str
    priority: str
    sentiment: str
    requires_human: bool


class ClassificationService:
    def __init__(self, ai_service: ResilientAIService | None = None) -> None:
        self._service = ai_service or build_service()

    def classify(self, ticket: str) -> ClassificationResult:
        """Classify a raw support ticket string into a structured result."""
        raw = self._service.generate(
            prompt=f"Ticket:\n{ticket}",
            system=_SYSTEM_PROMPT,
            temperature=_TEMPERATURE,
        )
        logger.info("[classify] raw response length=%d", len(raw))
        data = _parse_json(raw)
        return ClassificationResult(
            category=data["category"],
            priority=data["priority"],
            sentiment=data["sentiment"],
            requires_human=bool(data["requires_human"]),
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
            f"Classification response was not valid JSON.\n"
            f"Parse error: {exc}\n"
            f"Raw response:\n{raw}"
        ) from exc


classification_service = ClassificationService()
