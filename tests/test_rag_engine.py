import pytest
from unittest.mock import MagicMock, patch

from app.services.rag_engine import RAGEngine, RAGResponse, Source, _DISTANCE_THRESHOLD, _TOP_K, _is_refusal


# ── fixtures ──────────────────────────────────────────────────────────────────

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
def engine():
    with patch("app.services.rag_engine.cache") as mock_cache, \
         patch("app.services.rag_engine.rate_limiter") as mock_rl, \
         patch("app.services.rag_engine.vector_store") as mock_vs, \
         patch("app.services.rag_engine.embedding_service") as mock_emb:

        mock_cache.get.return_value = None
        mock_emb.embed.return_value = [0.1] * 384

        eng = RAGEngine(ai_service=MagicMock())
        eng._mock_cache = mock_cache
        eng._mock_rl = mock_rl
        eng._mock_vs = mock_vs
        eng._mock_emb = mock_emb
        yield eng


# ── cache behaviour ───────────────────────────────────────────────────────────

class TestCache:
    def test_cache_hit_returns_immediately(self, engine):
        engine._mock_cache.get.return_value = {
            "answer": "cached answer",
            "sources": [{"title": "Dune", "author": "Frank Herbert", "relevance_score": 0.9}],
        }
        response = engine.query("any question")
        assert response.cached is True
        assert response.answer == "cached answer"
        engine._mock_rl.acquire.assert_not_called()
        engine._mock_vs.search.assert_not_called()

    def test_cache_hit_restores_sources(self, engine):
        engine._mock_cache.get.return_value = {
            "answer": "answer",
            "sources": [{"title": "Dune", "author": "Frank Herbert", "relevance_score": 0.92}],
        }
        response = engine.query("any question")
        assert len(response.sources) == 1
        assert response.sources[0].title == "Dune"
        assert response.sources[0].relevance_score == 0.92

    def test_cache_miss_stores_result(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Frank Herbert", 0.2)]
        engine._service.generate.return_value = "great book"
        engine.query("desert planets")
        engine._mock_cache.set.assert_called_once()
        stored = engine._mock_cache.set.call_args[0][1]
        assert stored["answer"] == "great book"
        assert stored["sources"][0]["title"] == "Dune"


# ── relevance filtering ───────────────────────────────────────────────────────

class TestRelevanceFilter:
    def test_returns_refusal_when_no_results_pass_threshold(self, engine):
        engine._mock_vs.search.return_value = [
            _make_result("Unrelated", "Author", _DISTANCE_THRESHOLD + 0.01),
        ]
        response = engine.query("meaning of life")
        assert response.sources == []
        assert "couldn't find" in response.answer.lower()
        engine._service.generate.assert_not_called()

    def test_results_at_threshold_boundary_are_kept(self, engine):
        engine._mock_vs.search.return_value = [
            _make_result("Dune", "Frank Herbert", _DISTANCE_THRESHOLD),
        ]
        engine._service.generate.return_value = "answer"
        response = engine.query("desert")
        assert len(response.sources) == 1

    def test_results_above_threshold_are_excluded(self, engine):
        engine._mock_vs.search.return_value = [
            _make_result("Relevant", "Author A", 0.3),
            _make_result("Irrelevant", "Author B", _DISTANCE_THRESHOLD + 0.1),
        ]
        engine._service.generate.return_value = "answer"
        response = engine.query("question")
        assert len(response.sources) == 1
        assert response.sources[0].title == "Relevant"


# ── response structure ────────────────────────────────────────────────────────

class TestResponseStructure:
    def test_relevance_score_is_one_minus_distance(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.3)]
        engine._service.generate.return_value = "answer"
        response = engine.query("space")
        assert response.sources[0].relevance_score == round(1.0 - 0.3, 4)

    def test_cached_false_on_fresh_response(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        engine._service.generate.return_value = "answer"
        response = engine.query("space")
        assert response.cached is False

    def test_to_dict_serialises_correctly(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        engine._service.generate.return_value = "answer"
        d = engine.query("space").to_dict()
        assert "answer" in d
        assert "sources" in d
        assert "cached" in d
        assert d["sources"][0]["relevance_score"] == round(1.0 - 0.2, 4)


# ── pipeline ordering ─────────────────────────────────────────────────────────

class TestPipeline:
    def test_rate_limiter_called_on_cache_miss(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        engine._service.generate.return_value = "answer"
        engine.query("question")
        engine._mock_rl.acquire.assert_called_once()

    def test_rate_limit_exceeded_bubbles_up(self, engine):
        from app.infrastructure.rate_limiter import RateLimitExceeded
        engine._mock_rl.acquire.side_effect = RateLimitExceeded("limit hit")
        with pytest.raises(RateLimitExceeded):
            engine.query("question")

    def test_embedding_used_for_vector_search(self, engine):
        engine._mock_emb.embed.return_value = [0.5] * 384
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        engine._service.generate.return_value = "answer"
        engine.query("space travel")
        engine._mock_emb.embed.assert_called_once_with("space travel")
        engine._mock_vs.search.assert_called_once_with([0.5] * 384, top_k=_TOP_K)

    def test_no_ai_call_when_no_relevant_results(self, engine):
        engine._mock_vs.search.return_value = [
            _make_result("X", "Y", _DISTANCE_THRESHOLD + 0.1),
        ]
        engine.query("meaning of life")
        engine._service.generate.assert_not_called()
        engine._mock_cache.set.assert_not_called()


# ── refusal detection ─────────────────────────────────────────────────────────

class TestRefusalDetection:
    def test_sources_cleared_when_ai_declines(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        engine._service.generate.return_value = "I couldn't find any books about photography."
        response = engine.query("photography")
        assert response.sources == []

    def test_answer_preserved_on_refusal(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        refusal = "I couldn't find any books matching your query."
        engine._service.generate.return_value = refusal
        response = engine.query("photography")
        assert response.answer == refusal

    def test_refusal_cached_with_empty_sources(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        engine._service.generate.return_value = "There are no books about photography."
        engine.query("photography")
        cached = engine._mock_cache.set.call_args[0][1]
        assert cached["sources"] == []

    def test_sources_returned_when_ai_answers_normally(self, engine):
        engine._mock_vs.search.return_value = [_make_result("Dune", "Herbert", 0.2)]
        engine._service.generate.return_value = "Dune is a great sci-fi book."
        response = engine.query("desert planets")
        assert len(response.sources) == 1


class TestIsRefusal:
    @pytest.mark.parametrize("phrase", [
        "I couldn't find any books",
        "could not find anything relevant",
        "there are no books about this",
        "no titles match your query",
        "nothing relevant in the catalogue",
        "not in the catalogue",
        "not in our catalogue",
        "don't have any books",
        "do not have any titles",
        "not available in our collection",
        "no results were found",
    ])
    def test_known_refusal_phrases(self, phrase):
        assert _is_refusal(phrase) is True

    def test_normal_answer_not_a_refusal(self):
        assert _is_refusal("Dune by Frank Herbert is a great science fiction novel.") is False

    def test_case_insensitive(self):
        assert _is_refusal("I COULDN'T FIND any books.") is True
