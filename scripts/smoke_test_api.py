"""
HTTP smoke test — hits every endpoint and reports pass/fail.

Start the server first:
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Then run:
    python scripts/smoke_test_api.py
"""

import sys

import httpx

BASE_URL = "http://localhost:8000"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    status = PASS if ok else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def run(client: httpx.Client) -> None:
    # ── Health ────────────────────────────────────────────────────────────────
    print("\n=== Health ===")
    r = client.get("/health")
    data = r.json()
    check("GET /health returns 200", r.status_code == 200)
    check("/health status=ok", data.get("status") == "ok")
    check("/health has daily_cost_usd", "daily_cost_usd" in data)
    check("/health has total_requests_today", "total_requests_today" in data)

    # ── Books ─────────────────────────────────────────────────────────────────
    print("\n=== Books ===")
    r = client.get("/api/v1/books/")
    check("GET /books/ returns 200", r.status_code == 200)
    books = r.json()
    check("Catalogue has ≥20 books", len(books) >= 20, f"got {len(books)}")

    r = client.get("/api/v1/books/book-001")
    check("GET /books/book-001 returns 200", r.status_code == 200)

    r = client.get("/api/v1/books/nonexistent-id")
    check("GET /books/nonexistent returns 404", r.status_code == 404)

    # ── Search / Books ────────────────────────────────────────────────────────
    print("\n=== Search / Books ===")
    r = client.post("/api/v1/search/books", json={"query": "space exploration adventure", "limit": 5})
    check("POST /search/books returns 200", r.status_code == 200)
    hits = r.json()
    check("/search/books returns a list", isinstance(hits, list))
    if hits:
        check("Result has relevance_score", "relevance_score" in hits[0])
        check("Result has title and author", "title" in hits[0] and "author" in hits[0])

    r = client.post("/api/v1/search/books", json={"query": "x"})
    check("/search/books rejects short query → 422", r.status_code == 422)

    # ── Search / Ask (RAG) ────────────────────────────────────────────────────
    print("\n=== Search / Ask ===")
    r = client.post(
        "/api/v1/search/ask",
        json={"question": "What science fiction books do you have about desert planets?"},
        timeout=60,
    )
    check("POST /search/ask returns 200", r.status_code == 200)
    data = r.json()
    check("/search/ask has answer", bool(data.get("answer")))
    check("/search/ask has sources list", isinstance(data.get("sources"), list))
    check("/search/ask has cached flag", "cached" in data)

    r = client.post(
        "/api/v1/search/ask",
        json={"question": "What is the meaning of life?"},
        timeout=60,
    )
    check("Off-topic ask returns 200", r.status_code == 200)
    check("Off-topic ask returns empty sources", r.json().get("sources") == [])

    # ── Query (legacy) ────────────────────────────────────────────────────────
    print("\n=== Query (legacy path) ===")
    r = client.post(
        "/api/v1/query/",
        json={"question": "Recommend a classic romance novel"},
        timeout=60,
    )
    check("POST /query/ returns 200", r.status_code == 200)
    check("POST /query/ has answer", bool(r.json().get("answer")))

    # ── Chat ──────────────────────────────────────────────────────────────────
    print("\n=== Chat ===")
    r = client.post("/api/v1/chat/", json={"message": "Hi!"}, timeout=60)
    check("POST /chat/ new conversation returns 200", r.status_code == 200)
    cid = r.json().get("conversation_id")
    check("Server generates conversation_id", bool(cid) and len(cid) == 36)

    r = client.post(
        "/api/v1/chat/",
        json={"message": "Recommend a thriller book", "conversation_id": cid},
        timeout=60,
    )
    check("Second turn with same ID returns 200", r.status_code == 200)

    r = client.post(
        "/api/v1/chat/",
        json={"message": "Tell me more about that one", "conversation_id": cid},
        timeout=60,
    )
    check("Third turn (memory test) returns 200", r.status_code == 200)

    r2 = client.post("/api/v1/chat/", json={"message": "Hello"}, timeout=60)
    new_cid = r2.json().get("conversation_id")
    check("Two conversations get different IDs", cid != new_cid)

    # ── Classify ──────────────────────────────────────────────────────────────
    print("\n=== Classify ===")
    r = client.post(
        "/api/v1/classify/ticket",
        json={"ticket": "My library card isn't working at the self-checkout and I'm very frustrated."},
        timeout=60,
    )
    check("POST /classify/ticket returns 200", r.status_code == 200)
    data = r.json()
    check("Has category", "category" in data)
    check("Has priority", "priority" in data)
    check("Has sentiment", "sentiment" in data)
    check("Has suggested_department", "suggested_department" in data)
    check("Has summary", "summary" in data)
    check(
        "Angry card ticket → sentiment=negative",
        data.get("sentiment") == "negative",
        f"got: {data.get('sentiment')}",
    )
    check(
        "Angry card ticket → category=technical",
        data.get("category") == "technical",
        f"got: {data.get('category')}",
    )

    r = client.post("/api/v1/classify/ticket", json={"ticket": "short"})
    check("/classify/ticket rejects short input → 422", r.status_code == 422)

    # ── Summarise ─────────────────────────────────────────────────────────────
    print("\n=== Summarise ===")
    r = client.post(
        "/api/v1/summarise/reviews",
        json={"reviews": [
            "Absolutely loved it. The world-building is stunning.",
            "Dense and slow at first but incredibly rewarding.",
            "A masterpiece that redefined the genre.",
            "Some parts dragged but the characters were unforgettable.",
            "Mixed feelings — brilliant ideas but hard to get through.",
        ]},
        timeout=60,
    )
    check("POST /summarise/reviews returns 200", r.status_code == 200)
    data = r.json()
    check("Has overall_sentiment", "overall_sentiment" in data)
    check("Has average_rating", "average_rating" in data)
    check("Has key_themes", "key_themes" in data)
    check("Has praise", "praise" in data)
    check("Has criticism", "criticism" in data)
    check("Has recommendation", "recommendation" in data)
    rating = data.get("average_rating", 0)
    check("average_rating is 1–5", isinstance(rating, (int, float)) and 1 <= rating <= 5, f"got: {rating}")

    r = client.post("/api/v1/summarise/reviews", json={"reviews": []})
    check("/summarise/reviews rejects empty list → 422", r.status_code == 422)

    # ── 422 validation ────────────────────────────────────────────────────────
    print("\n=== Validation ===")
    r = client.post("/api/v1/query/", json={"question": "x"})
    check("Short question → 422", r.status_code == 422)

    r = client.post("/api/v1/query/", json={})
    check("Missing question field → 422", r.status_code == 422)


def main() -> None:
    print(f"LibraryMind API smoke test → {BASE_URL}")
    try:
        with httpx.Client(base_url=BASE_URL, timeout=10) as client:
            run(client)
    except httpx.ConnectError:
        print(f"\nERROR: Could not connect to {BASE_URL}. Is the server running?")
        sys.exit(1)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{'='*50}")
    print(f"  {passed} passed  |  {failed} failed  |  {len(_results)} total")

    if failed:
        print("\nFailed:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  ✗ {name}" + (f" ({detail})" if detail else ""))
        sys.exit(1)
    else:
        print("  All checks passed.")


if __name__ == "__main__":
    main()
