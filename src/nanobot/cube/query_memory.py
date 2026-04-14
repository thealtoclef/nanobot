"""QueryMemory: ChromaDB-backed store for (nl_question, payload) pairs."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import EmbedderConfig, RerankerConfig


class QueryMemory:
    """ChromaDB-backed store for (question, payload) pairs.

    Collection: cube_memory
    Persists to: {workspace}/chroma/
    """

    def __init__(
        self,
        persist_dir: Path,
        max_results: int = 5,
        embedder: "EmbedderConfig | None" = None,
        reranker: "RerankerConfig | None" = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.max_results = max_results
        self.embedder = embedder
        self.reranker = reranker
        self._collection = None
        self._client = None
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def initialize(self) -> None:
        """Setup ChromaDB with CubeEmbeddingFunction, create collection."""
        import chromadb
        from chromadb.config import Settings

        from nanobot.cube.embedding_function import CubeEmbeddingFunction

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        settings = Settings(is_persistent=True, anonymized_telemetry=False)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=settings,
        )
        ef = CubeEmbeddingFunction(config=self.embedder) if self.embedder else None
        self._collection = self._client.get_or_create_collection(
            name="cube_memory",
            embedding_function=ef,
        )
        self._available = True

    async def store(self, question: str, payload: str) -> None:
        """Save pair with embedding."""
        import asyncio
        import uuid

        if not self._collection:
            return

        def _sync_store():
            self._collection.add(
                ids=[str(uuid.uuid7())],
                documents=[question],
                metadatas=[{"payload": payload}],
            )

        await asyncio.to_thread(_sync_store)

    async def search(self, question: str, limit: int | None = None) -> list[tuple[str, str, float]]:
        """Semantic search → list of (question, payload, score) tuples."""
        import asyncio

        if not self._collection:
            return []

        limit = limit or self.max_results

        def _sync_search():
            results = self._collection.query(
                query_texts=[question],
                n_results=limit,
            )
            pairs = []
            if results and results.get("documents") and results["documents"][0]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0] if results.get("metadatas") else [],
                    results.get("distances", [[]])[0] if results.get("distances") else [],
                ):
                    payload = metadata.get("payload", "") if metadata else ""
                    score = 1.0 - distance if distance is not None else 0.0
                    pairs.append((doc, payload, score))
            return pairs

        return await asyncio.to_thread(_sync_search)

    async def clear(self) -> None:
        """Clear all stored pairs."""
        import asyncio

        if not self._collection:
            return

        def _sync_clear():
            self._collection.delete(where={})

        await asyncio.to_thread(_sync_clear)
