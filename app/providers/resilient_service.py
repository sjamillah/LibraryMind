from app.providers.base import AIProvider
from app.core.exceptions import AllProvidersFailedError
from app.infrastructure.rate_limiter import rate_limiter, RateLimitExceeded
from app.infrastructure.usage_tracker import usage_tracker
from app.infrastructure.cache import cache


class ResilientAIService:
    def __init__(self, providers: list[AIProvider]):
        self.providers = providers

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Try providers in order, fall back on failure. Returns the first successful response."""
        # Cache hits bypass rate limiting — no provider is called, no cost incurred.
        # temperature/max_tokens are part of the key so two calls with identical
        # prompt+system but different generation settings don't collide.
        cache_key = cache.make_key(prompt, system, temperature, max_tokens)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        rate_limiter.acquire()  # RateLimitExceeded bubbles up to the caller

        errors: list[str] = []
        for provider in self.providers:
            try:
                response = provider.generate(prompt, system, temperature, max_tokens)
                usage_tracker.record(
                    provider=provider.provider_name,
                    model=provider.model_name,
                    prompt_text=f"{system}\n{prompt}" if system else prompt,
                    completion_text=response,
                )
                cache.set(cache_key, response)
                return response
            except Exception as e:
                errors.append(f"{type(provider).__name__}: {e}")

        raise AllProvidersFailedError(f"All providers failed: {errors}")


def build_service() -> ResilientAIService:
    """Factory that orders providers based on PRIMARY_PROVIDER config."""
    from app.providers.openai_provider import OpenAIProvider
    from app.providers.anthropic_provider import AnthropicAIProvider
    from app.core.config import settings

    openai = OpenAIProvider()
    anthropic = AnthropicAIProvider()

    if settings.PRIMARY_PROVIDER.lower() == "openai":
        return ResilientAIService([openai, anthropic])
    return ResilientAIService([anthropic, openai])


# Single shared instance — every service gets the same fallback policy and
# provider order. Built once here (composition root) instead of each service
# calling build_service() independently, matching how cache/rate_limiter/
# usage_tracker are already shared singletons.
ai_service = build_service()
