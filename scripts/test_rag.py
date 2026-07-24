"""
Manual smoke test for the RAG engine.

Run from the project root after seeding ChromaDB:
    python scripts/seed.py
    python scripts/test_rag.py

Expected results:
  1. Desert planet question  → answer citing Dune / science fiction books
  2. Meaning of life        → polite refusal, no fabricated titles
  3. Same question (repeat) → cached: True, near-instant response
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from app.services.rag_engine import rag_engine


def _print_response(label: str, response, elapsed: float) -> None:
    print(f"\n{'='*60}")
    print(f"QUERY: {label}")
    print(f"Cached: {response.cached}  |  Time: {elapsed:.2f}s")
    print(f"\nAnswer:\n{response.answer}")
    if response.sources:
        print("\nSources:")
        for s in response.sources:
            print(f"  • {s.title} by {s.author}  (relevance: {s.relevance_score:.4f})")
    else:
        print("\nSources: none")


def main() -> None:
    tests = [
        "What science fiction books do you have about desert planets?",
        "What is the meaning of life?",
        "What science fiction books do you have about desert planets?",
    ]

    for question in tests:
        t0 = time.perf_counter()
        response = rag_engine.query(question)
        elapsed = time.perf_counter() - t0
        _print_response(question, response, elapsed)

    print(f"\n{'='*60}")
    print("Test complete.")


if __name__ == "__main__":
    main()
