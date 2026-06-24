import logging

import chromadb

logger = logging.getLogger(__name__)

_CHROMA_PATH = ".chroma"


class VectorStore:
    """ChromaDB-backed vector store for book embeddings.

    Distance note — ChromaDB returns cosine *distance*, not similarity:
        distance 0.0  →  identical vectors (most relevant)
        distance 1.0  →  maximally dissimilar (least relevant)

    Any relevance threshold in Part 4 must be a maximum distance, e.g.
    ``distance < 0.5``, not a minimum similarity score.
    """

    def __init__(self, collection_name: str = "books") -> None:
        self._client = chromadb.PersistentClient(path=_CHROMA_PATH)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "[vector_store] collection '%s' ready — %d documents",
            collection_name,
            self._collection.count(),
        )

    def upsert(self, id: str, vector: list[float], metadata: dict) -> None:
        """Insert or update a single document. Safe to call repeatedly."""
        self._collection.upsert(
            ids=[id],
            embeddings=[vector],
            metadatas=[metadata],
        )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Return the top_k closest documents to query_vector.

        Each result: {"id": str, "distance": float, "metadata": dict}
        Results are sorted ascending by distance (most relevant first).
        """
        count = self._collection.count()
        if count == 0:
            return []
        n = min(top_k, count)

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n,
            include=["metadatas", "distances"],
        )

        return [
            {
                "id": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]

    def count(self) -> int:
        return self._collection.count()


vector_store = VectorStore()
