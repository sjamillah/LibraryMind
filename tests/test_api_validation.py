"""API-layer input validation tests.

These test the boundary of every request model directly through the HTTP
layer with TestClient, not the service logic underneath (that's covered in
tests/test_*_service.py). Pydantic validation runs before the route handler
does, so invalid-input cases never touch a service at all - only the
valid-input boundary cases need the underlying service mocked out, so no
real AI call or network request ever happens here.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── /search/books ────────────────────────────────────────────────────────────

class TestSearchBooksValidation:
    def test_query_at_min_length_is_accepted(self):
        with patch("app.api.v1.search.embedding_service") as mock_emb, \
             patch("app.api.v1.search.vector_store") as mock_vs:
            mock_emb.embed.return_value = [0.1] * 384
            mock_vs.search.return_value = []
            r = client.post("/api/v1/search/books", json={"query": "abc"})
        assert r.status_code == 200

    def test_query_below_min_length_rejected(self):
        r = client.post("/api/v1/search/books", json={"query": "ab"})
        assert r.status_code == 422

    def test_query_at_max_length_is_accepted(self):
        with patch("app.api.v1.search.embedding_service") as mock_emb, \
             patch("app.api.v1.search.vector_store") as mock_vs:
            mock_emb.embed.return_value = [0.1] * 384
            mock_vs.search.return_value = []
            r = client.post("/api/v1/search/books", json={"query": "a" * 500})
        assert r.status_code == 200

    def test_query_above_max_length_rejected(self):
        r = client.post("/api/v1/search/books", json={"query": "a" * 501})
        assert r.status_code == 422

    def test_missing_query_rejected(self):
        r = client.post("/api/v1/search/books", json={})
        assert r.status_code == 422

    @pytest.mark.parametrize("limit", [1, 20])
    def test_limit_at_boundaries_accepted(self, limit):
        with patch("app.api.v1.search.embedding_service") as mock_emb, \
             patch("app.api.v1.search.vector_store") as mock_vs:
            mock_emb.embed.return_value = [0.1] * 384
            mock_vs.search.return_value = []
            r = client.post("/api/v1/search/books", json={"query": "space travel", "limit": limit})
        assert r.status_code == 200

    @pytest.mark.parametrize("limit", [0, 21, -1])
    def test_limit_outside_boundaries_rejected(self, limit):
        r = client.post("/api/v1/search/books", json={"query": "space travel", "limit": limit})
        assert r.status_code == 422

    def test_default_limit_used_when_omitted(self):
        with patch("app.api.v1.search.embedding_service") as mock_emb, \
             patch("app.api.v1.search.vector_store") as mock_vs:
            mock_emb.embed.return_value = [0.1] * 384
            mock_vs.search.return_value = []
            client.post("/api/v1/search/books", json={"query": "space travel"})
        mock_vs.search.assert_called_once_with([0.1] * 384, top_k=5)


# ── /search/ask and /query/ (identical validation shape) ─────────────────────

@pytest.mark.parametrize("path,module", [
    ("/api/v1/search/ask", "app.api.v1.search"),
    ("/api/v1/query/", "app.api.v1.query"),
])
class TestQuestionValidation:
    def test_question_at_min_length_accepted(self, path, module):
        fake = MagicMock(answer="hi", sources=[], cached=False)
        with patch(f"{module}.rag_engine") as mock_engine:
            mock_engine.query.return_value = fake
            r = client.post(path, json={"question": "abc"})
        assert r.status_code == 200

    def test_question_below_min_length_rejected(self, path, module):
        r = client.post(path, json={"question": "ab"})
        assert r.status_code == 422

    def test_question_at_max_length_accepted(self, path, module):
        fake = MagicMock(answer="hi", sources=[], cached=False)
        with patch(f"{module}.rag_engine") as mock_engine:
            mock_engine.query.return_value = fake
            r = client.post(path, json={"question": "a" * 500})
        assert r.status_code == 200

    def test_question_above_max_length_rejected(self, path, module):
        r = client.post(path, json={"question": "a" * 501})
        assert r.status_code == 422

    def test_missing_question_rejected(self, path, module):
        r = client.post(path, json={})
        assert r.status_code == 422

    def test_empty_string_question_rejected(self, path, module):
        r = client.post(path, json={"question": ""})
        assert r.status_code == 422

    def test_wrong_type_rejected(self, path, module):
        r = client.post(path, json={"question": 12345})
        assert r.status_code == 422


# ── /chat/ ─────────────────────────────────────────────────────────────────

class TestChatValidation:
    def _fake_response(self):
        return MagicMock(conversation_id="00000000-0000-0000-0000-000000000000", reply="hi", sources=[])

    def test_message_at_min_length_accepted(self):
        with patch("app.api.v1.chat.chat_service") as mock_svc:
            mock_svc.chat.return_value = self._fake_response()
            r = client.post("/api/v1/chat/", json={"message": "a"})
        assert r.status_code == 200

    def test_empty_message_rejected(self):
        r = client.post("/api/v1/chat/", json={"message": ""})
        assert r.status_code == 422

    def test_message_at_max_length_accepted(self):
        with patch("app.api.v1.chat.chat_service") as mock_svc:
            mock_svc.chat.return_value = self._fake_response()
            r = client.post("/api/v1/chat/", json={"message": "a" * 1000})
        assert r.status_code == 200

    def test_message_above_max_length_rejected(self):
        r = client.post("/api/v1/chat/", json={"message": "a" * 1001})
        assert r.status_code == 422

    def test_missing_message_rejected(self):
        r = client.post("/api/v1/chat/", json={})
        assert r.status_code == 422

    def test_conversation_id_optional(self):
        with patch("app.api.v1.chat.chat_service") as mock_svc:
            mock_svc.chat.return_value = self._fake_response()
            r = client.post("/api/v1/chat/", json={"message": "hello"})
        assert r.status_code == 200
        mock_svc.chat.assert_called_once_with(message="hello", conversation_id=None)

    def test_conversation_id_passed_through_when_given(self):
        with patch("app.api.v1.chat.chat_service") as mock_svc:
            mock_svc.chat.return_value = self._fake_response()
            r = client.post("/api/v1/chat/", json={"message": "hello", "conversation_id": "abc-123"})
        assert r.status_code == 200
        mock_svc.chat.assert_called_once_with(message="hello", conversation_id="abc-123")


# ── /classify/ticket ──────────────────────────────────────────────────────────

class TestClassifyValidation:
    def _fake_result(self):
        return MagicMock(
            category="general", priority="low", sentiment="neutral",
            suggested_department="General Enquiries", summary="test",
        )

    def test_ticket_at_min_length_accepted(self):
        with patch("app.api.v1.classify.classification_service") as mock_svc:
            mock_svc.classify.return_value = self._fake_result()
            r = client.post("/api/v1/classify/ticket", json={"ticket": "a" * 10})
        assert r.status_code == 200

    def test_ticket_below_min_length_rejected(self):
        r = client.post("/api/v1/classify/ticket", json={"ticket": "a" * 9})
        assert r.status_code == 422

    def test_ticket_at_max_length_accepted(self):
        with patch("app.api.v1.classify.classification_service") as mock_svc:
            mock_svc.classify.return_value = self._fake_result()
            r = client.post("/api/v1/classify/ticket", json={"ticket": "a" * 2000})
        assert r.status_code == 200

    def test_ticket_above_max_length_rejected(self):
        r = client.post("/api/v1/classify/ticket", json={"ticket": "a" * 2001})
        assert r.status_code == 422

    def test_missing_ticket_rejected(self):
        r = client.post("/api/v1/classify/ticket", json={})
        assert r.status_code == 422


# ── /summarise/reviews ─────────────────────────────────────────────────────────

class TestSummariseValidation:
    def _fake_result(self):
        return MagicMock(
            overall_sentiment="positive", average_rating=4.0, key_themes=[],
            praise=[], criticism=[], recommendation="test",
        )

    def test_single_review_accepted(self):
        with patch("app.api.v1.summarise.summarisation_service") as mock_svc:
            mock_svc.summarise.return_value = self._fake_result()
            r = client.post("/api/v1/summarise/reviews", json={"reviews": ["Loved it."]})
        assert r.status_code == 200

    def test_fifty_reviews_accepted(self):
        with patch("app.api.v1.summarise.summarisation_service") as mock_svc:
            mock_svc.summarise.return_value = self._fake_result()
            r = client.post("/api/v1/summarise/reviews", json={"reviews": ["Great book."] * 50})
        assert r.status_code == 200

    def test_fifty_one_reviews_rejected(self):
        r = client.post("/api/v1/summarise/reviews", json={"reviews": ["Great book."] * 51})
        assert r.status_code == 422

    def test_empty_review_list_rejected(self):
        r = client.post("/api/v1/summarise/reviews", json={"reviews": []})
        assert r.status_code == 422

    def test_missing_reviews_field_rejected(self):
        r = client.post("/api/v1/summarise/reviews", json={})
        assert r.status_code == 422

    def test_reviews_wrong_type_rejected(self):
        r = client.post("/api/v1/summarise/reviews", json={"reviews": "not a list"})
        assert r.status_code == 422


# ── /books/ ────────────────────────────────────────────────────────────────

class TestBooksValidation:
    def test_list_all_books_returns_full_catalogue(self):
        r = client.get("/api/v1/books/")
        assert r.status_code == 200
        assert len(r.json()) == 24

    def test_get_existing_book_by_id(self):
        r = client.get("/api/v1/books/book-001")
        assert r.status_code == 200
        assert r.json()["id"] == "book-001"

    def test_get_nonexistent_book_returns_404(self):
        r = client.get("/api/v1/books/does-not-exist")
        assert r.status_code == 404

    def test_get_book_with_special_characters_does_not_crash(self):
        r = client.get("/api/v1/books/../../etc/passwd")
        assert r.status_code in (404, 200)

    def test_get_book_with_unicode_id_does_not_crash(self):
        r = client.get("/api/v1/books/📚")
        assert r.status_code == 404


# ── health ────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_expected_shape(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "daily_cost_usd" in data
        assert "total_requests_today" in data
