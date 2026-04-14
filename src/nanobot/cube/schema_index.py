"""CubeSchemaIndex: ChromaDB-backed schema element indexing for context retrieval."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import EmbedderConfig, RerankerConfig


# ── Module-level describe functions ─────────────────────────────────


def describe_cube(cube: dict) -> str:
    """Convert a single cube/view dict to markdown description.

    Covers fields relevant for NL→SQL:
    - name, type, title, description
    - connectedComponent (join relationship info)
    - measures, dimensions, segments
    """
    lines: list[str] = []

    name = cube.get("name", "unknown")
    cube_type = cube.get("type", "cube")
    title = cube.get("title", name)
    connected = cube.get("connectedComponent")

    # Cube/View header
    header = f"### {cube_type.capitalize()}: {name}"
    if title and title != name:
        header += f" (Title: {title})"
    lines.append(header)

    # Description
    description = cube.get("description", "")
    if description:
        lines.append(f"Description: {description}")

    # Connected component (join group)
    if connected is not None:
        lines.append(f"ConnectedComponent: {connected} (cubes with same value can be joined)")

    # Measures
    measures = cube.get("measures", [])
    if measures:
        lines.append("\n#### Measures:")
        for m in measures:
            m_name = m.get("name", "?")
            m_type = m.get("type", "?")
            m_agg = m.get("aggType", m_type)
            m_desc = m.get("description", "")
            line = f"  - {m_name} ({m_agg})"
            if m_desc:
                line += f" — {m_desc}"
            lines.append(line)

    # Dimensions
    dimensions = cube.get("dimensions", [])
    if dimensions:
        lines.append("\n#### Dimensions:")
        for d in dimensions:
            d_name = d.get("name", "?")
            d_type = d.get("type", "?")
            d_desc = d.get("description", "")
            line = f"  - {d_name} ({d_type})"
            if d_desc:
                line += f" — {d_desc}"
            lines.append(line)

    # Segments
    segments = cube.get("segments", [])
    if segments:
        lines.append("\n#### Segments:")
        for s in segments:
            s_name = s.get("name", "")
            if not s_name:
                continue
            s_desc = s.get("description", "")
            line = f"  - {s_name}"
            if s_desc:
                line += f" — {s_desc}"
            lines.append(line)

    return "\n".join(lines)


def describe_full_schema(cubes: list[dict]) -> str:
    """Generate full schema text for all cubes."""
    parts = [describe_cube(cube) for cube in cubes]
    return "\n\n".join(parts)


# ── CubeSchemaIndex class ─────────────────────────────────────────────────────


class CubeSchemaIndex:
    """ChromaDB-backed index of Cube schema elements for context retrieval.

    Collection: cube_schema
    Persists to: {workspace}/chroma/

    Dual strategy:
    - Small schemas (total text < CubeSchemaIndexConfig.threshold chars): skip index,
      return full schema via describe_full_schema()
    - Large schemas: use element-level embedding search with ChromaDB
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
        self._cube_json_cache: dict[str, dict] = {}  # cube_name -> cube dict (for retrieval)

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

    def describe_full_schema(self, cubes: list[dict]) -> str:
        """Return full schema as structured text for small schemas."""
        return describe_full_schema(cubes)

    async def index_cubes(self, cubes: list[dict]) -> None:
        """Index (or re-index) all cubes from /v1/meta response.

        Element-level indexing:
          - Cube-level:  id="cube::{name}",     item_type="cube",        text=describe_cube(cube)
          - Measure-level: id="measure::{cube}::{name}", item_type="measure", text="Measure '{name}' in cube '{cube}': {description}. Type: {type}."
          - Dimension-level: id="dim::{cube}::{name}", item_type="dimension", text="Dimension '{name}' in cube '{cube}': {description}. Type: {type}."
          - Segment-level: id="segment::{cube}::{name}", item_type="segment", text="Segment '{name}' in cube '{cube}': {description}."
        """
        import asyncio

        if not self._collection:
            return

        def _sync_index():
            # Clear existing records (full rebuild)
            existing = self._collection.get()
            if existing and existing.get("ids"):
                self._collection.delete(ids=existing["ids"])

            self._cube_json_cache.clear()

            # Index each cube and its elements
            for cube in cubes:
                cube_name = cube.get("name", "unknown")

                # Store full JSON for retrieval
                self._cube_json_cache[cube_name] = cube

                # 1. Cube-level record
                cube_text = describe_cube(cube)
                self._collection.add(
                    ids=[f"cube::{cube_name}"],
                    documents=[cube_text],
                    metadatas=[{"cube_name": cube_name, "item_type": "cube"}],
                )

                # 2. Measure-level records
                for m in cube.get("measures", []):
                    m_name = m.get("name", "")
                    if not m_name:
                        continue
                    m_type = m.get("type", "?")
                    m_agg = m.get("aggType", m_type)
                    m_desc = m.get("description", "")
                    measure_text = (
                        f"Measure '{m_name}' in cube '{cube_name}': {m_desc}. Type: {m_agg}."
                    )
                    self._collection.add(
                        ids=[f"measure::{cube_name}::{m_name}"],
                        documents=[measure_text],
                        metadatas=[{"cube_name": cube_name, "item_type": "measure"}],
                    )

                # 3. Dimension-level records
                for d in cube.get("dimensions", []):
                    d_name = d.get("name", "")
                    if not d_name:
                        continue
                    d_type = d.get("type", "?")
                    d_desc = d.get("description", "")
                    dim_text = (
                        f"Dimension '{d_name}' in cube '{cube_name}': {d_desc}. Type: {d_type}."
                    )
                    self._collection.add(
                        ids=[f"dim::{cube_name}::{d_name}"],
                        documents=[dim_text],
                        metadatas=[{"cube_name": cube_name, "item_type": "dimension"}],
                    )

                # 4. Segment-level records
                for s in cube.get("segments", []):
                    s_name = s.get("name", "")
                    if not s_name:
                        continue
                    s_desc = s.get("description", "")
                    seg_text = f"Segment '{s_name}' in cube '{cube_name}': {s_desc}."
                    self._collection.add(
                        ids=[f"segment::{cube_name}::{s_name}"],
                        documents=[seg_text],
                        metadatas=[{"cube_name": cube_name, "item_type": "segment"}],
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

    def get_cube_json(self, cube_name: str) -> dict | None:
        """Return the full cube JSON for a given cube name (for retrieval)."""
        return self._cube_json_cache.get(cube_name)

    async def clear(self) -> None:
        """Clear all indexed schema items."""
        import asyncio

        if not self._collection:
            return

        def _sync_clear():
            self._collection.delete(where={})

        await asyncio.to_thread(_sync_clear)
        self._cube_json_cache.clear()
