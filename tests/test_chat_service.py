import pytest
from unittest.mock import MagicMock, patch

from app.services.chat_service import (
    ChatService,
    ConversationStore,
    _build_prompt,
    _HISTORY_WINDOW,
    _DISTANCE_THRESHOLD,
)


def _make_result(title: str, author: str, distance: float) -> dict:
    return {
        "id": f"book-{title[:4].lower()}",
        "distance": distance,
        "metadata": {
            "title": title,
            "author": author,
            "year": 2000,
            "genre": "science fiction",
            "description": f"A book about {title.lower()}.",
        },
    }


@pytest.fixture
def service():
    with patch("app.services.chat_service.embedding_service") as mock_emb, \
         patch("app.services.chat_service.vector_store") as mock_vs:

        mock_emb.embed.return_value = [0.1] * 384
        mock_vs.search.return_value = []

        mock_ai = MagicMock()
        store = ConversationStore()
        svc = ChatService(ai_service=mock_ai, store=store)
        svc._mock_ai = mock_ai
        svc._mock_emb = mock_emb
        svc._mock_vs = mock_vs
        yield svc


# ── conversation ID ───────────────────────────────────────────────────────────

class TestConversationId:
    def test_generates_id_when_none_given(self, service):
        service._mock_ai.generate.return_value = "reply"
        result = service.chat("hello")
        assert result.conversation_id
        assert len(result.conversation_id) == 36  # UUID4 format

    def test_preserves_provided_id(self, service):
        service._mock_ai.generate.return_value = "reply"
        result = service.chat("hello", conversation_id="my-custom-id")
        assert result.conversation_id == "my-custom-id"

    def test_two_new_conversations_get_different_ids(self, service):
        service._mock_ai.generate.return_value = "reply"
        r1 = service.chat("hello")
        r2 = service.chat("hello")
        assert r1.conversation_id != r2.conversation_id


# ── history storage ───────────────────────────────────────────────────────────

class TestHistoryStorage:
    def test_both_turns_stored_after_one_exchange(self, service):
        service._mock_ai.generate.return_value = "The AI reply."
        result = service.chat("User message")
        history = service._store.get(result.conversation_id)
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "User message"}
        assert history[1] == {"role": "assistant", "content": "The AI reply."}

    def test_history_grows_with_each_turn(self, service):
        service._mock_ai.generate.return_value = "reply"
        r = service.chat("first")
        cid = r.conversation_id
        service.chat("second", conversation_id=cid)
        service.chat("third", conversation_id=cid)
        assert len(service._store.get(cid)) == 6  # 3 turns × 2 messages each


# ── two-turn memory ───────────────────────────────────────────────────────────

class TestTwoTurnMemory:
    def test_second_prompt_contains_first_turn(self, service):
        service._mock_ai.generate.side_effect = [
            "I recommend Dune by Frank Herbert.",
            "Dune is set on the desert planet Arrakis.",
        ]
        r1 = service.chat("What books do you have about desert planets?")
        cid = r1.conversation_id

        service.chat("Tell me more about that.", conversation_id=cid)

        second_call_prompt = service._mock_ai.generate.call_args_list[1][1]["prompt"]
        assert "I recommend Dune by Frank Herbert." in second_call_prompt

    def test_first_prompt_has_no_history(self, service):
        service._mock_ai.generate.return_value = "reply"
        service.chat("First question ever.")
        first_call_prompt = service._mock_ai.generate.call_args_list[0][1]["prompt"]
        assert "User: First question ever." in first_call_prompt
        # No prior Assistant: lines
        assert "Assistant:" not in first_call_prompt


# ── isolation ─────────────────────────────────────────────────────────────────

class TestIsolation:
    def test_two_conversations_have_separate_histories(self, service):
        service._mock_ai.generate.return_value = "reply"
        r1 = service.chat("Message in conversation one")
        r2 = service.chat("Message in conversation two")

        assert r1.conversation_id != r2.conversation_id
        h1 = service._store.get(r1.conversation_id)
        h2 = service._store.get(r2.conversation_id)
        assert h1[0]["content"] == "Message in conversation one"
        assert h2[0]["content"] == "Message in conversation two"
        assert len(h1) == 2
        assert len(h2) == 2

    def test_continuing_one_conversation_does_not_affect_another(self, service):
        service._mock_ai.generate.return_value = "reply"
        r1 = service.chat("Conversation one, turn one")
        r2 = service.chat("Conversation two, turn one")

        service.chat("Conversation one, turn two", conversation_id=r1.conversation_id)

        assert len(service._store.get(r1.conversation_id)) == 4
        assert len(service._store.get(r2.conversation_id)) == 2


# ── history window truncation ─────────────────────────────────────────────────

class TestHistoryWindow:
    def test_only_last_n_messages_sent_to_ai(self, service):
        service._mock_ai.generate.return_value = "reply"

        # Fill history beyond the window
        cid = None
        for i in range(_HISTORY_WINDOW + 2):
            r = service.chat(f"message {i}", conversation_id=cid)
            cid = r.conversation_id

        # The last generate() call should not contain the very first message
        last_prompt = service._mock_ai.generate.call_args_list[-1][1]["prompt"]
        assert "message 0" not in last_prompt

    def test_full_history_still_stored(self, service):
        service._mock_ai.generate.return_value = "reply"

        cid = None
        total_turns = _HISTORY_WINDOW + 3
        for i in range(total_turns):
            r = service.chat(f"message {i}", conversation_id=cid)
            cid = r.conversation_id

        assert len(service._store.get(cid)) == total_turns * 2


# ── sources ───────────────────────────────────────────────────────────────────

class TestSources:
    def test_sources_returned_for_relevant_results(self, service):
        service._mock_vs.search.return_value = [
            _make_result("Dune", "Frank Herbert", 0.2),
        ]
        service._mock_ai.generate.return_value = "reply"
        result = service.chat("desert planets")
        assert len(result.sources) == 1
        assert result.sources[0].title == "Dune"
        assert result.sources[0].relevance_score == round(1.0 - 0.2, 4)

    def test_results_above_threshold_excluded(self, service):
        service._mock_vs.search.return_value = [
            _make_result("Relevant", "Author A", 0.3),
            _make_result("Irrelevant", "Author B", _DISTANCE_THRESHOLD + 0.01),
        ]
        service._mock_ai.generate.return_value = "reply"
        result = service.chat("question")
        assert len(result.sources) == 1
        assert result.sources[0].title == "Relevant"

    def test_no_sources_when_no_relevant_results(self, service):
        service._mock_vs.search.return_value = []
        service._mock_ai.generate.return_value = "reply"
        result = service.chat("obscure question")
        assert result.sources == []


# ── prompt builder ────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_empty_history_and_no_matches_still_ends_with_user_message(self):
        prompt = _build_prompt([], [], "Hello")
        assert prompt.endswith("User: Hello")

    def test_no_relevant_results_warns_against_inventing_titles(self):
        prompt = _build_prompt([], [], "Tell me about some obscure author")
        assert "Do not introduce any other book titles from memory" in prompt

    def test_history_appears_before_user_message(self):
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]
        prompt = _build_prompt(history, [], "Second question")
        lines = prompt.split("\n")
        assert lines[0] == "User: First question"
        assert lines[1] == "Assistant: First answer"
        assert lines[-1] == "User: Second question"

    def test_context_appears_between_history_and_message(self):
        result = _make_result("Dune", "Frank Herbert", 0.2)
        prompt = _build_prompt([], [result], "desert planets")
        assert "Catalogue context:" in prompt
        assert prompt.index("Catalogue context:") < prompt.index("User: desert planets")
