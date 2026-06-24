"""
Seed script — embeds every book in data/books.json and upserts it into ChromaDB.

Run from the project root:
    python scripts/seed.py

Safe to re-run: upsert is idempotent, so existing documents are updated,
not duplicated. On first run, sentence-transformers downloads ~80 MB of model
weights automatically — an internet connection is required.
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from app.services.embedding_service import embedding_service
from app.infrastructure.vector_store import vector_store


def _build_embed_text(book: dict) -> str:
    """Combine title, author, and description into one string for embedding.

    Including all three fields ensures the resulting vector represents the whole
    book: a query for 'Frank Herbert' matches on author, 'space travel' matches
    on description, and 'Dune' matches on title. Embedding description alone
    breaks author-name and title queries.
    """
    return f"{book['title']} by {book['author']}. {book['description']}"


def main() -> None:
    books_path = Path("data/books.json")
    if not books_path.exists():
        print(f"ERROR: {books_path} not found. Run from the project root.")
        sys.exit(1)

    books: list[dict] = json.loads(books_path.read_text(encoding="utf-8"))
    total = len(books)
    print(f"Seeding {total} books...\n")

    for i, book in enumerate(books, 1):
        vector = embedding_service.embed(_build_embed_text(book))
        vector_store.upsert(
            id=book["id"],
            vector=vector,
            metadata={
                "title": book["title"],
                "author": book["author"],
                "year": book["year"],
                "genre": book["genre"],
                "description": book["description"],
            },
        )
        print(f"  [{i:>2}/{total}] {book['title']}  ({book['genre']})")

    print(f"\nDone. {vector_store.count()} documents in ChromaDB.")

    print("\n--- search test: 'space travel adventure' ---")
    query = embedding_service.embed("space travel adventure")
    for r in vector_store.search(query, top_k=3):
        m = r["metadata"]
        print(f"  {r['distance']:.4f}  {m['title']} by {m['author']}  [{m['genre']}]")


if __name__ == "__main__":
    main()
