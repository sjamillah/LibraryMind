import pytest
import httpx
from unittest.mock import MagicMock, patch

from app.providers.gateway_client import call_gateway, _normalize_base_url


# ── helpers ──────────────────────────────────────────────────────────────────

def _ok_response(content: str = "hello") -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = {"choices": [{"message": {"content": content}}]}
    mock.raise_for_status.return_value = None
    return mock


# ── URL normalisation ─────────────────────────────────────────────────────────

class TestNormalizeBaseUrl:
    def test_preserves_path(self):
        assert _normalize_base_url("https://api.example.com/v2") == "https://api.example.com/v2/"

    def test_keeps_host_only_url_unchanged(self):
        assert _normalize_base_url("https://api.example.com") == "https://api.example.com/"

    def test_normalises_to_trailing_slash(self):
        assert _normalize_base_url("https://api.example.com/") == "https://api.example.com/"


# ── happy path ────────────────────────────────────────────────────────────────

class TestCallGatewaySuccess:
    def test_returns_content_from_response(self):
        with patch("app.providers.gateway_client.httpx.post", return_value=_ok_response("hi there")):
            result = call_gateway("openai", "gpt-3.5-turbo", "key", [], "https://api.example.com")
        assert result == "hi there"

    def test_sends_model_and_messages_in_body(self):
        messages = [{"role": "user", "content": "hello"}]
        with patch("app.providers.gateway_client.httpx.post", return_value=_ok_response()) as mock_post:
            call_gateway("anthropic", "claude-3", "my-key", messages, "https://api.example.com")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "claude-3"
        assert payload["messages"] == messages
        assert "provider" not in payload
        assert "api-key" not in payload

    def test_sends_provider_and_api_key_headers(self):
        with patch("app.providers.gateway_client.httpx.post", return_value=_ok_response()) as mock_post:
            call_gateway("openai", "gpt-3.5-turbo", "my-secret-key", [], "https://api.example.com")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Provider"] == "openai"
        assert headers["X-Api-Key"] == "my-secret-key"

    def test_gateway_url_always_has_trailing_slash(self):
        with patch("app.providers.gateway_client.httpx.post", return_value=_ok_response()) as mock_post:
            call_gateway("openai", "gpt", "key", [], "https://api.example.com/v2/public")
        url_used = mock_post.call_args.args[0]
        assert url_used == "https://api.example.com/v2/public/"


# ── retry behaviour ───────────────────────────────────────────────────────────

class TestCallGatewayRetries:
    def test_retries_on_timeout_and_succeeds(self):
        with patch("app.providers.gateway_client.httpx.post") as mock_post, \
             patch("app.providers.gateway_client.time.sleep"):
            mock_post.side_effect = [httpx.TimeoutException("timeout"), _ok_response("recovered")]
            result = call_gateway("openai", "gpt", "key", [], "https://api.example.com")
        assert result == "recovered"
        assert mock_post.call_count == 2

    def test_retries_on_5xx_and_succeeds(self):
        server_err = httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock())
        server_err.response.status_code = 503
        with patch("app.providers.gateway_client.httpx.post") as mock_post, \
             patch("app.providers.gateway_client.time.sleep"):
            mock_post.side_effect = [server_err, _ok_response("ok")]
            result = call_gateway("openai", "gpt", "key", [], "https://api.example.com")
        assert result == "ok"

    def test_does_not_retry_on_4xx(self):
        """401/403/404 won't resolve on retry — fail fast."""
        auth_err = httpx.HTTPStatusError("401", request=MagicMock(), response=MagicMock())
        auth_err.response.status_code = 401
        auth_err.response.text = "Unauthorized"
        with patch("app.providers.gateway_client.httpx.post") as mock_post, \
             patch("app.providers.gateway_client.time.sleep") as mock_sleep:
            mock_post.side_effect = auth_err
            with pytest.raises(RuntimeError):
                call_gateway("openai", "gpt", "key", [], "https://api.example.com", max_retries=3)
        assert mock_post.call_count == 1   # gave up immediately
        mock_sleep.assert_not_called()     # no wasted backoff time

    def test_raises_runtime_error_after_all_retries_exhausted(self):
        with patch("app.providers.gateway_client.httpx.post") as mock_post, \
             patch("app.providers.gateway_client.time.sleep"):
            mock_post.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(RuntimeError, match="Gateway call failed"):
                call_gateway("openai", "gpt", "key", [], "https://api.example.com", max_retries=3)
        assert mock_post.call_count == 3

    def test_does_not_sleep_after_last_attempt(self):
        """Sleep is wasted time after the final failure — it should not happen."""
        with patch("app.providers.gateway_client.httpx.post") as mock_post, \
             patch("app.providers.gateway_client.time.sleep") as mock_sleep:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(RuntimeError):
                call_gateway("openai", "gpt", "key", [], "https://api.example.com", max_retries=3)
        assert mock_sleep.call_count == 2  # slept after attempt 1 and 2, never after attempt 3

    def test_exponential_backoff_delays(self):
        """First retry waits 1s, second waits 2s."""
        with patch("app.providers.gateway_client.httpx.post") as mock_post, \
             patch("app.providers.gateway_client.time.sleep") as mock_sleep:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            with pytest.raises(RuntimeError):
                call_gateway("openai", "gpt", "key", [], "https://api.example.com", max_retries=3)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2]
