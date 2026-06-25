import pytest
from unittest.mock import MagicMock

from app.services.classification_service import (
    ClassificationService,
    ClassificationResult,
    _strip_fences,
)

_VALID_JSON = (
    '{"category": "renewal", "priority": "low", '
    '"sentiment": "neutral", "requires_human": false}'
)


def _make_service(response: str) -> ClassificationService:
    mock_ai = MagicMock()
    mock_ai.generate.return_value = response
    return ClassificationService(ai_service=mock_ai)


# ── happy path ────────────────────────────────────────────────────────────────

class TestClassify:
    def test_parses_clean_json(self):
        result = _make_service(_VALID_JSON).classify("Please renew my book.")
        assert result.category == "renewal"
        assert result.priority == "low"
        assert result.sentiment == "neutral"
        assert result.requires_human is False

    def test_strips_json_code_fence(self):
        result = _make_service(f"```json\n{_VALID_JSON}\n```").classify("ticket")
        assert result.category == "renewal"

    def test_strips_plain_code_fence(self):
        result = _make_service(f"```\n{_VALID_JSON}\n```").classify("ticket")
        assert result.category == "renewal"

    def test_requires_human_coerced_to_bool(self):
        raw = '{"category": "complaint", "priority": "high", "sentiment": "negative", "requires_human": true}'
        result = _make_service(raw).classify("I am unhappy.")
        assert result.requires_human is True


# ── temperature ───────────────────────────────────────────────────────────────

class TestTemperature:
    def test_low_temperature_used(self):
        mock_ai = MagicMock()
        mock_ai.generate.return_value = _VALID_JSON
        ClassificationService(ai_service=mock_ai).classify("ticket")
        _, kwargs = mock_ai.generate.call_args
        assert kwargs["temperature"] == 0.1


# ── error handling ────────────────────────────────────────────────────────────

class TestErrors:
    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            _make_service("not json at all").classify("some ticket")

    def test_error_includes_raw_response(self):
        raw = "definitely not json {{{"
        with pytest.raises(ValueError, match=raw):
            _make_service(raw).classify("some ticket")


# ── fence stripping ───────────────────────────────────────────────────────────

class TestStripFences:
    def test_strips_json_fence(self):
        assert _strip_fences("```json\n{}\n```") == "{}"

    def test_strips_plain_fence(self):
        assert _strip_fences("```\n{}\n```") == "{}"

    def test_leaves_plain_json_untouched(self):
        assert _strip_fences('{"a": 1}') == '{"a": 1}'

    def test_strips_surrounding_whitespace(self):
        assert _strip_fences("  {}  ") == "{}"
