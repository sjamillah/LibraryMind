import pytest
from unittest.mock import MagicMock

from app.services.summarisation_service import (
    SummarisationService,
    SummarisationResult,
    _strip_fences,
)

_VALID_JSON = (
    '{"summary": "Patrons loved this book.", '
    '"overall_sentiment": "positive", '
    '"key_themes": ["world-building", "depth"], '
    '"recommended": true}'
)


def _make_service(response: str) -> SummarisationService:
    mock_ai = MagicMock()
    mock_ai.generate.return_value = response
    return SummarisationService(ai_service=mock_ai)


# ── happy path ────────────────────────────────────────────────────────────────

class TestSummarise:
    def test_parses_clean_json(self):
        result = _make_service(_VALID_JSON).summarise(["Great!", "Loved it."])
        assert result.summary == "Patrons loved this book."
        assert result.overall_sentiment == "positive"
        assert result.key_themes == ["world-building", "depth"]
        assert result.recommended is True

    def test_strips_json_code_fence(self):
        result = _make_service(f"```json\n{_VALID_JSON}\n```").summarise(["review"])
        assert result.summary == "Patrons loved this book."

    def test_strips_plain_code_fence(self):
        result = _make_service(f"```\n{_VALID_JSON}\n```").summarise(["review"])
        assert result.summary == "Patrons loved this book."

    def test_recommended_coerced_to_bool(self):
        raw = (
            '{"summary": "Mixed.", "overall_sentiment": "mixed", '
            '"key_themes": ["pacing"], "recommended": false}'
        )
        result = _make_service(raw).summarise(["Mediocre."])
        assert result.recommended is False

    def test_all_reviews_included_in_prompt(self):
        mock_ai = MagicMock()
        mock_ai.generate.return_value = _VALID_JSON
        svc = SummarisationService(ai_service=mock_ai)
        reviews = ["First review.", "Second review.", "Third review."]
        svc.summarise(reviews)
        prompt_arg = mock_ai.generate.call_args[1]["prompt"]
        for r in reviews:
            assert r in prompt_arg


# ── temperature ───────────────────────────────────────────────────────────────

class TestTemperature:
    def test_low_temperature_used(self):
        mock_ai = MagicMock()
        mock_ai.generate.return_value = _VALID_JSON
        SummarisationService(ai_service=mock_ai).summarise(["review"])
        _, kwargs = mock_ai.generate.call_args
        assert kwargs["temperature"] == 0.2


# ── validation ────────────────────────────────────────────────────────────────

class TestValidation:
    def test_raises_on_empty_reviews(self):
        with pytest.raises(ValueError, match="At least one review"):
            _make_service(_VALID_JSON).summarise([])


# ── error handling ────────────────────────────────────────────────────────────

class TestErrors:
    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _make_service("not json").summarise(["review"])

    def test_error_includes_raw_response(self):
        raw = "definitely not json {{{"
        with pytest.raises(ValueError, match=raw):
            _make_service(raw).summarise(["review"])


# ── fence stripping ───────────────────────────────────────────────────────────

class TestStripFences:
    def test_strips_json_fence(self):
        assert _strip_fences("```json\n{}\n```") == "{}"

    def test_strips_plain_fence(self):
        assert _strip_fences("```\n{}\n```") == "{}"

    def test_leaves_plain_json_untouched(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'
