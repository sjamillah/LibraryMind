from app.providers.base import AIProvider
from app.providers.gateway_client import call_gateway
from app.core.config import settings


class OpenAIProvider(AIProvider):
    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return settings.OPENAI_API_MODEL

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return call_gateway("openai", settings.OPENAI_API_MODEL, settings.AMALI_API_KEY, messages, settings.AMALI_GATEWAY_URL)
