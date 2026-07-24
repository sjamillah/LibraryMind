import argparse
import logging
import sys
import time
from pathlib import Path

import httpx


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.core.config import settings
from app.core.exceptions import ConfigurationError, GatewayRequestError

logger = logging.getLogger(__name__)


def _normalize_base_url(base_url: str) -> str:
    """Normalise to a stable form: trimmed, always ends with '/' for consistent routing."""
    return base_url.strip().rstrip("/") + "/"


def _default_model(provider: str) -> str:
    if provider.lower() == "anthropic":
        return settings.ANTHROPIC_API_MODEL
    return settings.OPENAI_API_MODEL


def call_gateway(
    provider: str,
    model: str,
    api_key: str,
    messages: list,
    base_url: str,
    max_retries: int = 3,
) -> str:
    last_error = None
    url = _normalize_base_url(base_url)

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
                headers={
                    "Provider": provider,
                    "X-Api-Key": api_key,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            logger.debug("[gateway:%s] raw response: %s", provider, data)
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            last_error = e
            logger.warning(
                "[gateway:%s] attempt %d/%d HTTP %s — body: %s",
                provider, attempt + 1, max_retries, e.response.status_code, e.response.text,
            )
            if e.response.status_code < 500:
                break  # 4xx errors are client mistakes; retrying won't help
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except httpx.TimeoutException as e:
            last_error = e
            logger.warning(
                "[gateway:%s] attempt %d/%d timed out, retrying in %ds",
                provider, attempt + 1, max_retries, 2 ** attempt,
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise GatewayRequestError(f"Gateway call failed after {max_retries} attempts: {last_error}")


def _build_messages(prompt: str, system: str = "") -> list:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def main() -> int:
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="Smoke-test the LibraryMind gateway client")
    parser.add_argument("--provider", default=settings.PRIMARY_PROVIDER or "openai")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url", default=settings.AMALI_GATEWAY_URL)
    parser.add_argument("--api-key", default=settings.AMALI_API_KEY)
    parser.add_argument("--system", default="")
    parser.add_argument("--prompt", default="Reply with a short hello message.")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    if not args.base_url:
        raise ConfigurationError("AMALI_GATEWAY_URL is missing from .env")
    if not args.api_key:
        raise ConfigurationError("AMALI_API_KEY is missing from .env")

    model = args.model or _default_model(args.provider)
    response_text = call_gateway(
        provider=args.provider,
        model=model,
        api_key=args.api_key,
        messages=_build_messages(args.prompt, args.system),
        base_url=args.base_url,
        max_retries=args.max_retries,
    )
    print(response_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
