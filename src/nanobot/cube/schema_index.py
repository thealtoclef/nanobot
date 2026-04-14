"""CubeSchemaIndex: ChromaDB-backed schema element indexing for context retrieval."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import EmbedderConfig, RerankerConfig


class CubeSchemaIndex:
    """ChromaDB-backed index of Cube schema elements for context retrieval.

    Collection: cube_schema
    Persists to: {workspace}/chroma/
    """

    def __init__(
        self,
        persist_dir: Path,
        max_results: int = 10,
        embedder: "EmbedderConfig | None" = None,
        reranker: "RerankerConfig | None" = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.max_results = max_results
        self.embedder = embedder
        self.reranker = reranker
        self._client = None
        self._collection = None
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def initialize(self) -> None:
        """Setup ChromaDB with CubeEmbeddingFunction, create cube_schema collection."""
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
            name="cube_schema",
            embedding_function=ef,
        )
        self._available = True

    async def index_cubes(self, cubes: list[dict]) -> None:
        """Index (or re-index) all cubes from /v1/meta response.

        Each cube becomes one record:
          - id: "cube::{cube_name}"
          - text: TOON representation of the full cube
          - metadata: {"cube_name": "...", "item_type": "cube"}
        """
        import asyncio

        import toons

        if not self._collection:
            return

        def _sync_index():
            # Clear existing records (full rebuild)
            existing = self._collection.get()
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])

            # Index each cube as a separate record
            for cube in cubes:
                cube_name = cube.get("name", "unknown")
                toon_text = toons.dumps(cube)
                self._collection.add(
                    ids=[f"cube::{cube_name}"],
                    documents=[toon_text],
                    metadatas=[{"cube_name": cube_name, "item_type": "cube"}],
                )

        await asyncio.to_thread(_sync_index)

    async def search(
        self,
        question: str,
        limit: int | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Semantic search → list of (text, metadata, score) tuples."""
        import asyncio

        if not self._collection:
            return []

        limit = limit or self.max_results

        def _sync_search():
            results = self._collection.query(
                query_texts=[question],
                n_results=limit,
            )
            search_results = []
            if results and results.get("documents") and results["documents"][0]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0] if results.get("metadatas") else [],
                    results.get("distances", [[]])[0] if results.get("distances") else [],
                ):
                    meta = metadata or {}
                    score = 1.0 - distance if distance is not None else 0.0
                    search_results.append((doc, meta, score))
            return search_results

        return await asyncio.to_thread(_sync_search)

    async def clear(self) -> None:
        """Clear all indexed schema items."""
        import asyncio

        if not self._collection:
            return

        def _sync_clear():
            self._collection.delete(where={})

        await asyncio.to_thread(_sync_clear)
