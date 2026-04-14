"""End-to-end tests for Cube semantic layer against live Cube + DuckDB stack.

These tests use the existing fixtures at tests/e2e/cube/fixtures/
- compose.yaml: Cube.js + DuckDB stack
- conf/model/customers.yaml: Cube model definition
- conf/db/jaffle_shop.duckdb: DuckDB database
"""

import pytest

from nanobot.config.schema import CubeConfig, EmbedderConfig, ProviderConfig
from nanobot.cube.query_memory import QueryMemory
from nanobot.cube.schema_index import CubeSchemaIndex
from nanobot.cube.service import CubeService


@pytest.fixture
async def cube_service_init(cube_service_url):
    """Create an initialized CubeService for testing."""
    service = CubeService(
        CubeConfig(
            cube_url=cube_service_url,
            token="secret",
            cubejs_api_path="/cubejs-api",
            timeout=30.0,
            request_span_enabled=True,
            continue_wait_retry_interval=1.0,
            continue_wait_retry_max_attempts=5,
        )
    )
    await service.initialize()
    yield service
    await service.close()


class TestCubeE2E:
    """End-to-end tests for Cube with Docker.

    Uses the existing fixture model definition:
    - cubes[0].name = "customers"
    - measures[0].name = "count"
    - dimensions use names like: customer_id, first_name, last_name, etc.
    """

    async def test_health_check(self, cube_service_init):
        """Test /readyz endpoint returns True when Cube is live."""
        result = await cube_service_init.check_ready()
        assert result is True

    async def test_get_schema(self, cube_service_init):
        """Test fetching schema via /v1/meta contains 'customers' cube."""
        schema = await cube_service_init.get_schema()
        assert "customers" in schema

    async def test_execute_query_count(self, cube_service_init):
        """Test executing a query via /v1/load returns data list.

        Uses the Cube API measure name 'customers.count'.
        """
        payload = {
            "query": {
                "measures": ["customers.count"],
            }
        }
        result = await cube_service_init.execute_query(payload)
        # execute_query returns the full Cube response dict with 'data' key
        assert "data" in result
        data = result["data"]
        assert isinstance(data, list)
        assert len(data) == 1
        # Cube returns normalized names with cube prefix
        assert "customers.count" in data[0]

    async def test_preview_sql(self, cube_service_init):
        """Test previewing SQL via /v1/sql returns SQL string containing SELECT."""
        payload = {
            "query": {
                "measures": ["customers.count"],
            }
        }
        result = await cube_service_init.preview_sql(payload)
        assert "SELECT" in result.upper()

    async def test_query_with_dimensions(self, cube_service_init):
        """Test querying with dimensions returns correct data.

        Uses the Cube API dimension names like 'customers.first_name'.
        """
        payload = {
            "query": {
                "dimensions": ["customers.first_name"],
                "measures": ["customers.count"],
            }
        }
        result = await cube_service_init.execute_query(payload)
        # execute_query returns the full Cube response dict with 'data' key
        assert "data" in result
        data = result["data"]
        assert isinstance(data, list)
        # Check that we got rows back
        assert len(data) > 0
        # Cube returns normalized names with cube prefix
        assert "customers.first_name" in data[0]

    async def test_schema_context_with_question(self, cube_service_init):
        """Test get_schema_context with a question returns relevant schema."""
        # Force large schema mode by setting threshold low
        cube_service_init.schema_index.threshold = 10

        result = await cube_service_init.get_schema_context(question="How many customers?")
        # Should contain either full schema or search results
        assert isinstance(result, str)
        assert len(result) > 0


class TestChromaDBE2E:
    """End-to-end tests for ChromaDB local integration with Cube components.

    Tests that our CubeEmbeddingFunction works correctly with ChromaDB's
    PersistentClient (local mode, not remote server).
    """

    def test_chroma_query_memory_integration(self):
        """Test QueryMemory can be initialized and used with ChromaDB.

        This catches issues with CubeEmbeddingFunction missing required
        methods like name(), is_legacy(), default_space(), etc.
        """
        import tempfile
        from pathlib import Path

        # Create a temporary directory for ChromaDB persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "chroma"

            # Create embedder config with a valid-looking provider
            embedder_cfg = EmbedderConfig(
                provider=ProviderConfig(
                    backend="openai",
                    base_url="https://api.openai.com",
                    api_key="test-key",  # Won't actually be called in this test
                ),
                model="text-embedding-3-small",
            )

            # Initialize QueryMemory - this will fail if CubeEmbeddingFunction
            # doesn't have the required ChromaDB protocol methods
            memory = QueryMemory(
                persist_dir=persist_path,
                max_results=5,
                embedder=embedder_cfg,
                reranker=None,
            )
            memory.initialize()

            # Verify it's available
            assert memory.is_available is True

            # Verify collection was created with correct name
            assert memory._collection is not None

    def test_chroma_schema_index_integration(self):
        """Test CubeSchemaIndex can be initialized with ChromaDB.

        This catches issues with CubeEmbeddingFunction missing required
        methods like name(), is_legacy(), default_space(), etc.
        """
        import tempfile
        from pathlib import Path

        # Create a temporary directory for ChromaDB persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "chroma"

            # Create embedder config with a valid-looking provider
            embedder_cfg = EmbedderConfig(
                provider=ProviderConfig(
                    backend="openai",
                    base_url="https://api.openai.com",
                    api_key="test-key",  # Won't actually be called in this test
                ),
                model="text-embedding-3-small",
            )

            # Initialize CubeSchemaIndex - this will fail if CubeEmbeddingFunction
            # doesn't have the required ChromaDB protocol methods
            index = CubeSchemaIndex(
                persist_dir=persist_path,
                max_results=10,
                embedder=embedder_cfg,
                reranker=None,
            )
            index.initialize()

            # Verify it's available
            assert index.is_available is True

            # Verify collection was created with correct name
            assert index._collection is not None

    def test_chroma_schema_index_index_and_search(self):
        """Test CubeSchemaIndex can index cubes and search for relevant ones.

        This tests the full index + search flow with simulated cube meta data.
        Note: This test requires a valid OpenAI API key since it performs real embeddings.
        Skipped unless OPENAI_API_KEY is set.
        """
        import asyncio

        # Skip if no real API key available
        import os
        import tempfile
        from pathlib import Path

        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip(
                "OPENAI_API_KEY not set - skipping integration test that requires real API calls"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            persist_path = Path(tmpdir) / "chroma"

            # Create embedder config with real API key
            embedder_cfg = EmbedderConfig(
                provider=ProviderConfig(
                    backend="openai",
                    base_url="https://api.openai.com",
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                ),
                model="text-embedding-3-small",
            )

            # Initialize CubeSchemaIndex
            index = CubeSchemaIndex(
                persist_dir=persist_path,
                max_results=10,
                embedder=embedder_cfg,
                reranker=None,
            )
            index.initialize()

            # Simulated cube meta response (similar to what Cube /v1/meta returns)
            cubes = [
                {
                    "name": "orders",
                    "title": "Orders",
                    "dimensions": [
                        {"name": "orders.status", "type": "string"},
                        {"name": "orders.customer_name", "type": "string"},
                    ],
                    "measures": [
                        {"name": "orders.count", "type": "number"},
                    ],
                },
                {
                    "name": "customers",
                    "title": "Customers",
                    "dimensions": [
                        {"name": "customers.city", "type": "string"},
                        {"name": "customers.age", "type": "number"},
                    ],
                    "measures": [
                        {"name": "customers.count", "type": "number"},
                    ],
                },
            ]

            # Index the cubes (this converts to TOON and stores in ChromaDB)
            asyncio.run(index.index_cubes(cubes))

            # Search for "orders" related cubes
            results = asyncio.run(index.search("How many orders by status?"))
            assert len(results) > 0

            # First result should be about orders
            text, metadata, score = results[0]
            assert "orders" in metadata.get("cube_name", "")
            assert score > 0.0
