from unittest.mock import patch

from app.providers.openai_provider import OpenAIProvider
from app.providers.anthropic_provider import AnthropicAIProvider


# ── OpenAI provider ───────────────────────────────────────────────────────────

class TestOpenAIProvider:
    def test_provider_name_is_openai(self):
        assert OpenAIProvider().provider_name == "openai"

    def test_calls_gateway_with_openai_as_provider(self):
        with patch("app.providers.openai_provider.call_gateway", return_value="ok") as mock_gw:
            OpenAIProvider().generate("hello")
        assert mock_gw.call_args.args[0] == "openai"

    def test_user_message_is_last_in_list(self):
        with patch("app.providers.openai_provider.call_gateway", return_value="ok") as mock_gw:
            OpenAIProvider().generate("what is a library?")
        messages = mock_gw.call_args.args[3]
        assert messages[-1] == {"role": "user", "content": "what is a library?"}

    def test_system_message_is_prepended(self):
        with patch("app.providers.openai_provider.call_gateway", return_value="ok") as mock_gw:
            OpenAIProvider().generate("hello", system="You are a librarian.")
        messages = mock_gw.call_args.args[3]
        assert messages[0] == {"role": "system", "content": "You are a librarian."}
        assert messages[1] == {"role": "user", "content": "hello"}

    def test_no_system_message_when_empty_string(self):
        with patch("app.providers.openai_provider.call_gateway", return_value="ok") as mock_gw:
            OpenAIProvider().generate("hello", system="")
        messages = mock_gw.call_args.args[3]
        assert all(m["role"] != "system" for m in messages)
        assert len(messages) == 1


# ── Anthropic provider ────────────────────────────────────────────────────────

class TestAnthropicProvider:
    def test_provider_name_is_anthropic(self):
        assert AnthropicAIProvider().provider_name == "anthropic"

    def test_calls_gateway_with_anthropic_as_provider(self):
        with patch("app.providers.anthropic_provider.call_gateway", return_value="ok") as mock_gw:
            AnthropicAIProvider().generate("hello")
        assert mock_gw.call_args.args[0] == "anthropic"

    def test_user_message_is_last_in_list(self):
        with patch("app.providers.anthropic_provider.call_gateway", return_value="ok") as mock_gw:
            AnthropicAIProvider().generate("recommend a book")
        messages = mock_gw.call_args.args[3]
        assert messages[-1] == {"role": "user", "content": "recommend a book"}

    def test_system_message_is_prepended(self):
        with patch("app.providers.anthropic_provider.call_gateway", return_value="ok") as mock_gw:
            AnthropicAIProvider().generate("hi", system="Be brief.")
        messages = mock_gw.call_args.args[3]
        assert messages[0] == {"role": "system", "content": "Be brief."}

    def test_no_system_message_when_empty_string(self):
        with patch("app.providers.anthropic_provider.call_gateway", return_value="ok") as mock_gw:
            AnthropicAIProvider().generate("hi", system="")
        messages = mock_gw.call_args.args[3]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
