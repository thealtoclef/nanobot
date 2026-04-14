"""Tests for Cube tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import (
    CubeConfig,
    CubeMemoryConfig,
    CubeSchemaIndexConfig,
    EmbedderConfig,
    ProviderConfig,
)
from nanobot.cube.query_memory import QueryMemory
from nanobot.cube.service import CubeService
from nanobot.tools.cube import CubeQueryTool, CubeSchemaTool, CubeSearchTool, _generate_span_id


@pytest.fixture
def cube_config():
    # Create valid embedder config for schema_index and memory
    embedder_cfg = EmbedderConfig(
        provider=ProviderConfig(
            backend="openai",
            base_url="https://api.openai.com",
            api_key="test-key",
        ),
        model="text-embedding-3-small",
    )
    schema_index_cfg = CubeSchemaIndexConfig(
        enabled=True,
        threshold=30_000,
        max_results=10,
        embedder=embedder_cfg,
    )
    memory_cfg = CubeMemoryConfig(
        enabled=True,
        max_results=5,
        embedder=embedder_cfg,
    )
    return CubeConfig(
        cube_url="https://cube.example.com",
        token="test-token",
        cubejs_api_path="/cubejs-api",
        timeout=30.0,
        request_span_enabled=True,
        continue_wait_retry_interval=0.01,
        continue_wait_retry_max_attempts=3,
        schema_index=schema_index_cfg,
        memory=memory_cfg,
    )


@pytest.fixture
def cube_service(cube_config):
    return CubeService(cube_config)


class TestGenerateSpanId:
    def test_generate_span_id(self):
        span_id = _generate_span_id()
        assert isinstance(span_id, str)
        assert len(span_id) > 0


class TestCubeSchemaTool:
    @pytest.mark.asyncio
    async def test_returns_schema(self, cube_service):
        tool = CubeSchemaTool(cube_service)
        cube_service._available = True
        cube_service._schema = "orders\ncustomers"

        result = await tool.execute()

        assert "orders" in result

    @pytest.mark.asyncio
    async def test_unavailable_returns_error(self, cube_service):
        tool = CubeSchemaTool(cube_service)
        cube_service._available = False
        cube_service._init_error = "Cube service failed readiness check"

        result = await tool.execute()

        assert "Error" in result
        assert "Cube service failed readiness check" in result

    @pytest.mark.asyncio
    async def test_calls_get_schema_context(self, cube_service):
        tool = CubeSchemaTool(cube_service)
        cube_service._available = True

        with patch.object(cube_service, "get_schema_context", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = "schema result"

            result = await tool.execute(question="How many orders?")

            mock_get.assert_called_once_with(question="How many orders?")


class TestCubeQueryTool:
    @pytest.mark.asyncio
    async def test_query_mode_returns_markdown(self, cube_service):
        tool = CubeQueryTool(cube_service)
        cube_service._available = True
        cube_service.request_span_enabled = False

        payload = {"query": {"measures": ["orders.count"]}}

        with patch.object(cube_service, "execute_query", new_callable=AsyncMock) as mock_exec:
            # execute_query returns full Cube response dict with "data" key
            mock_exec.return_value = {
                "data": [
                    {"orders.status": "completed", "orders.order_count": 150},
                ]
            }

            result = await tool.execute(payload=payload)

            assert "|" in result  # Markdown table
            assert "orders.order_count" in result

    @pytest.mark.asyncio
    async def test_dry_run_returns_sql(self, cube_service):
        tool = CubeQueryTool(cube_service)
        cube_service._available = True

        payload = {"query": {"measures": ["orders.count"]}}

        with patch.object(cube_service, "preview_sql", new_callable=AsyncMock) as mock_preview:
            mock_preview.return_value = "SELECT * FROM orders"

            result = await tool.execute(payload=payload, dry_run=True)

            assert "SELECT" in result

    @pytest.mark.asyncio
    async def test_error_returns_error_string(self, cube_service):
        tool = CubeQueryTool(cube_service)
        cube_service._available = True

        payload = {"query": {"measures": ["orders.count"]}}

        with patch.object(cube_service, "execute_query", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Query failed")

            result = await tool.execute(payload=payload)

            assert "Error" in result
            assert "Query failed" in result

    @pytest.mark.asyncio
    async def test_auto_save_on_success(self, cube_service):
        # Create a mock memory
        memory = MagicMock(spec=QueryMemory)
        memory.is_available = True
        memory.store = AsyncMock()

        tool = CubeQueryTool(cube_service, memory)
        cube_service._available = True
        cube_service.request_span_enabled = False

        payload = {"query": {"measures": ["orders.count"]}}

        with patch.object(cube_service, "execute_query", new_callable=AsyncMock) as mock_exec:
            # execute_query returns full Cube response dict with "data" key
            mock_exec.return_value = {"data": [{"orders.count": 100}]}

            result = await tool.execute(payload=payload, question="How many orders?")

            memory.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_save_without_question(self, cube_service):
        memory = MagicMock(spec=QueryMemory)
        memory.is_available = True
        memory.store = AsyncMock()

        tool = CubeQueryTool(cube_service, memory)
        cube_service._available = True
        cube_service.request_span_enabled = False

        payload = {"query": {"measures": ["orders.count"]}}

        with patch.object(cube_service, "execute_query", new_callable=AsyncMock) as mock_exec:
            # execute_query returns full Cube response dict with "data" key
            mock_exec.return_value = {"data": [{"orders.count": 100}]}

            result = await tool.execute(payload=payload)  # No question

            memory.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_save_on_error(self, cube_service):
        memory = MagicMock(spec=QueryMemory)
        memory.is_available = True
        memory.store = AsyncMock()

        tool = CubeQueryTool(cube_service, memory)
        cube_service._available = True

        payload = {"query": {"measures": ["orders.count"]}}

        with patch.object(cube_service, "execute_query", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = Exception("Query failed")

            result = await tool.execute(payload=payload, question="How many orders?")

            memory.store.assert_not_called()


class TestCubeSearchTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self):
        memory = MagicMock(spec=QueryMemory)
        memory.is_available = True
        memory.search = AsyncMock()
        memory.search.return_value = [
            ("How many orders?", '{"query": {"measures": ["orders.count"]}}', 0.95),
        ]

        tool = CubeSearchTool(memory)

        result = await tool.execute(question="How many orders?", limit=5)

        assert "How many orders?" in result
        assert "Similar Past Queries" in result

    @pytest.mark.asyncio
    async def test_no_results_message(self):
        memory = MagicMock(spec=QueryMemory)
        memory.is_available = True
        memory.search = AsyncMock()
        memory.search.return_value = []

        tool = CubeSearchTool(memory)

        result = await tool.execute(question="nonexistent", limit=5)

        assert "No similar queries" in result

    @pytest.mark.asyncio
    async def test_unavailable_memory(self):
        memory = MagicMock(spec=QueryMemory)
        memory.is_available = False

        tool = CubeSearchTool(memory)

        result = await tool.execute(question="How many orders?", limit=5)

        assert "Error" in result
        assert "not available" in result
