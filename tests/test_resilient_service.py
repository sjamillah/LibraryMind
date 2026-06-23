import pytest
from unittest.mock import patch, MagicMock

from app.providers.base import AIProvider
from app.providers.resilient_service import ResilientAIService, build_service
from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicAIProvider
from app.infrastructure.rate_limiter import RateLimitExceeded


# ── test double ───────────────────────────────────────────────────────────────

class StubProvider(AIProvider):
    """Deterministic provider stand-in — no network, no config."""

    def __init__(self, name: str, model: str, response: str = "stub", raises: Exception = None):
        self._name = name
        self._model = model
        self._response = response
        self._raises = raises
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt, system="", temperature=0.7, max_tokens=1000) -> str:
        self.call_count += 1
        if self._raises:
            raise self._raises
        return self._response


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def primary():
    return StubProvider("openai", "gpt-3.5-turbo", response="openai says hello")

@pytest.fixture
def fallback():
    return StubProvider("anthropic", "claude-3", response="anthropic says hello")

@pytest.fixture
def patched_infra():
    """Patch all three singletons so every test is fully isolated."""
    with patch("app.providers.resilient_service.cache") as mock_cache, \
         patch("app.providers.resilient_service.rate_limiter") as mock_rl, \
         patch("app.providers.resilient_service.usage_tracker") as mock_ut:
        mock_cache.get.return_value = None  # default: cache miss
        yield mock_cache, mock_rl, mock_ut


def _patch_infra(cache_return=None):
    """Helper for inline patches — always patches all three."""
    cache_patch = patch("app.providers.resilient_service.cache")
    rl_patch = patch("app.providers.resilient_service.rate_limiter")
    ut_patch = patch("app.providers.resilient_service.usage_tracker")
    return cache_patch, rl_patch, ut_patch


# ── provider selection ────────────────────────────────────────────────────────

class TestProviderSelection:
    def test_uses_primary_provider_when_it_succeeds(self, primary, fallback, patched_infra):
        service = ResilientAIService([primary, fallback])
        result = service.generate("hello")
        assert result == "openai says hello"
        assert primary.call_count == 1
        assert fallback.call_count == 0

    def test_falls_back_to_secondary_when_primary_fails(self, fallback, patched_infra):
        failing = StubProvider("openai", "gpt", raises=RuntimeError("API error"))
        service = ResilientAIService([failing, fallback])
        result = service.generate("hello")
        assert result == "anthropic says hello"
        assert failing.call_count == 1
        assert fallback.call_count == 1

    def test_raises_when_all_providers_fail(self, patched_infra):
        p1 = StubProvider("openai", "gpt", raises=RuntimeError("error A"))
        p2 = StubProvider("anthropic", "claude", raises=RuntimeError("error B"))
        service = ResilientAIService([p1, p2])
        with pytest.raises(RuntimeError, match="All providers failed"):
            service.generate("hello")

    def test_error_message_includes_every_failure(self, patched_infra):
        p1 = StubProvider("openai", "gpt", raises=RuntimeError("timeout"))
        p2 = StubProvider("anthropic", "claude", raises=RuntimeError("auth error"))
        service = ResilientAIService([p1, p2])
        with pytest.raises(RuntimeError) as exc_info:
            service.generate("hello")
        msg = str(exc_info.value)
        assert "timeout" in msg
        assert "auth error" in msg


# ── rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_rate_limit_exceeded_bubbles_up(self, primary, fallback):
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter") as mock_rl, \
             patch("app.providers.resilient_service.usage_tracker"):
            mock_cache.get.return_value = None
            mock_rl.acquire.side_effect = RateLimitExceeded("too many requests")
            service = ResilientAIService([primary, fallback])
            with pytest.raises(RateLimitExceeded):
                service.generate("hello")
        assert primary.call_count == 0

    def test_rate_limiter_not_called_on_cache_hit(self, primary):
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter") as mock_rl, \
             patch("app.providers.resilient_service.usage_tracker"):
            mock_cache.get.return_value = "cached response"
            service = ResilientAIService([primary])
            service.generate("hello")
        mock_rl.acquire.assert_not_called()

    def test_rate_limiter_called_on_cache_miss(self, primary):
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter") as mock_rl, \
             patch("app.providers.resilient_service.usage_tracker"):
            mock_cache.get.return_value = None
            service = ResilientAIService([primary])
            service.generate("hello")
        mock_rl.acquire.assert_called_once()


# ── cache behaviour ───────────────────────────────────────────────────────────

class TestCacheBehaviour:
    def test_cache_hit_returns_cached_value_without_calling_provider(self, primary):
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter"), \
             patch("app.providers.resilient_service.usage_tracker"):
            mock_cache.get.return_value = "cached answer"
            service = ResilientAIService([primary])
            result = service.generate("hello")
        assert result == "cached answer"
        assert primary.call_count == 0

    def test_cache_miss_stores_response(self, primary):
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter"), \
             patch("app.providers.resilient_service.usage_tracker"):
            mock_cache.get.return_value = None
            service = ResilientAIService([primary])
            service.generate("hello")
        mock_cache.set.assert_called_once()
        stored_value = mock_cache.set.call_args.args[1]
        assert stored_value == "openai says hello"

    def test_cache_hit_skips_usage_tracking(self, primary):
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter"), \
             patch("app.providers.resilient_service.usage_tracker") as mock_ut:
            mock_cache.get.return_value = "cached"
            service = ResilientAIService([primary])
            service.generate("hello")
        mock_ut.record.assert_not_called()

    def test_failed_call_is_not_cached(self, patched_infra):
        mock_cache, _, _ = patched_infra
        p = StubProvider("openai", "gpt", raises=RuntimeError("fail"))
        service = ResilientAIService([p])
        with pytest.raises(RuntimeError):
            service.generate("hello")
        mock_cache.set.assert_not_called()


# ── usage tracking ────────────────────────────────────────────────────────────

class TestUsageTracking:
    def test_records_usage_after_successful_call(self, primary, fallback):
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter"), \
             patch("app.providers.resilient_service.usage_tracker") as mock_ut:
            mock_cache.get.return_value = None
            service = ResilientAIService([primary, fallback])
            service.generate("hello", system="Be concise.")
        mock_ut.record.assert_called_once_with(
            provider="openai",
            model="gpt-3.5-turbo",
            prompt_text="Be concise.\nhello",
            completion_text="openai says hello",
        )

    def test_records_usage_for_fallback_provider(self, fallback):
        failing = StubProvider("openai", "gpt", raises=RuntimeError("fail"))
        with patch("app.providers.resilient_service.cache") as mock_cache, \
             patch("app.providers.resilient_service.rate_limiter"), \
             patch("app.providers.resilient_service.usage_tracker") as mock_ut:
            mock_cache.get.return_value = None
            service = ResilientAIService([failing, fallback])
            service.generate("hello")
        mock_ut.record.assert_called_once_with(
            provider="anthropic",
            model="claude-3",
            prompt_text="hello",
            completion_text="anthropic says hello",
        )

    def test_does_not_record_usage_when_all_providers_fail(self, patched_infra):
        _, _, mock_ut = patched_infra
        p = StubProvider("openai", "gpt", raises=RuntimeError("fail"))
        service = ResilientAIService([p])
        with pytest.raises(RuntimeError):
            service.generate("hello")
        mock_ut.record.assert_not_called()


# ── build_service factory ─────────────────────────────────────────────────────

class TestBuildService:
    def test_returns_resilient_service_with_two_providers(self):
        service = build_service()
        assert isinstance(service, ResilientAIService)
        assert len(service.providers) == 2

    def test_anthropic_first_when_primary_provider_is_anthropic(self):
        import app.core.config as cfg
        original = cfg.settings.PRIMARY_PROVIDER
        cfg.settings.PRIMARY_PROVIDER = "anthropic"
        try:
            service = build_service()
            assert isinstance(service.providers[0], AnthropicAIProvider)
            assert isinstance(service.providers[1], OpenAIProvider)
        finally:
            cfg.settings.PRIMARY_PROVIDER = original

    def test_openai_first_when_primary_provider_is_openai(self):
        import app.core.config as cfg
        original = cfg.settings.PRIMARY_PROVIDER
        cfg.settings.PRIMARY_PROVIDER = "openai"
        try:
            service = build_service()
            assert isinstance(service.providers[0], OpenAIProvider)
            assert isinstance(service.providers[1], AnthropicAIProvider)
        finally:
            cfg.settings.PRIMARY_PROVIDER = original
