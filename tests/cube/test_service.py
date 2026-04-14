"""Tests for CubeService."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.config.schema import (
    CubeConfig,
    CubeMemoryConfig,
    CubeSchemaIndexConfig,
    EmbedderConfig,
    ProviderConfig,
)
from nanobot.cube.service import CubeService


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


class TestInitialize:
    async def test_initialize_success(self, cube_service, mock_health_ok):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_health_ok
            mock_get_client.return_value = mock_client

            await cube_service.initialize()

            assert cube_service.is_available is True
            assert cube_service.init_error is None

    async def test_initialize_connection_error(self, cube_service, mock_health_fail):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.side_effect = mock_health_fail
            mock_get_client.return_value = mock_client

            await cube_service.initialize()

            assert cube_service.is_available is False
            assert cube_service.init_error is not None

    async def test_initialize_down(self, cube_service, mock_health_down):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_health_down
            mock_get_client.return_value = mock_client

            await cube_service.initialize()

            assert cube_service.is_available is False

    async def test_initialize_concurrent_serialization(self, cube_service, mock_health_ok):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_health_ok
            mock_get_client.return_value = mock_client

            # Run multiple initializes concurrently
            await asyncio.gather(
                cube_service.initialize(),
                cube_service.initialize(),
                cube_service.initialize(),
            )

            # Should only call check_ready once
            assert mock_client.get.call_count == 1


class TestHealthChecks:
    async def test_check_ready(self, cube_service, mock_health_ok):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_health_ok
            mock_get_client.return_value = mock_client

            result = await cube_service.check_ready()
            assert result is True

    async def test_check_live(self, cube_service, mock_live_ok):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_live_ok
            mock_get_client.return_value = mock_client

            result = await cube_service.check_live()
            assert result is True

    async def test_health_no_auth_header(self, cube_service, mock_health_ok):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_health_ok
            mock_get_client.return_value = mock_client

            await cube_service.check_ready()

            # Check no auth header was passed to /readyz
            call_args = mock_client.get.call_args
            assert "headers" not in call_args.kwargs or "Authorization" not in call_args.kwargs.get(
                "headers", {}
            )


class TestSchema:
    async def test_get_schema_parses_meta(self, cube_service, mock_meta_response):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_meta_response
            mock_response.text = '{"cubes": [...]}'
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            schema = await cube_service.get_schema()

            assert "orders" in schema
            assert "customers" in schema

    async def test_get_schema_empty_cubes(self, cube_service):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"cubes": []}
            mock_response.text = '{"cubes": []}'
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            schema = await cube_service.get_schema()

            # Schema is the raw JSON string
            assert '"cubes": []' in schema

    async def test_get_schema_caching(self, cube_service, mock_meta_response):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_meta_response
            mock_response.text = '{"cubes": [...]}'
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            schema1 = await cube_service.get_schema()
            schema2 = await cube_service.get_schema()

            # Should only make one HTTP call
            assert mock_client.get.call_count == 1
            assert schema1 == schema2

    async def test_get_schema_context_no_question(self, cube_service, mock_meta_response):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_meta_response
            mock_response.text = '{"cubes": [...]}'
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await cube_service.get_schema_context(None)

            assert "orders" in result

    async def test_get_schema_context_small_schema(self, cube_service, mock_meta_response):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_meta_response
            mock_response.text = '{"cubes": [...]}'
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Set threshold very high so schema is "small"
            cube_service.schema_index.threshold = 100_000_000

            result = await cube_service.get_schema_context("How many orders?")

            assert "orders" in result

    async def test_get_schema_context_large_schema_no_index(self, cube_service, mock_meta_response):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_meta_response
            mock_response.text = '{"cubes": [...]}'
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            # Set threshold very low to trigger "large schema" path
            cube_service.schema_index.threshold = 10
            # But no schema index available
            cube_service._schema_index = None

            result = await cube_service.get_schema_context("How many orders?")

            assert "orders" in result

    async def test_reload_stores_compiler_id(self, cube_service, mock_meta_response):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_meta_response
            mock_response.text = '{"cubes": [...]}'
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_get_client.return_value = mock_client

            await cube_service.reload()

            assert cube_service._compiler_id == "test-compiler-id-123"


class TestQueryExecution:
    async def test_execute_query_success(self, cube_service, mock_load_success):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_load_success
            mock_get_client.return_value = mock_client

            payload = {"query": {"measures": ["orders.count"]}}
            result = await cube_service.execute_query(payload)

            # execute_query returns the full response dict
            assert "data" in result
            assert len(result["data"]) == 2
            assert result["data"][0]["orders.order_count"] == 150

    async def test_execute_query_error(self, cube_service, mock_load_error):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_load_error
            mock_get_client.return_value = mock_client

            payload = {"query": {"measures": ["orders.count"]}}

            with pytest.raises(Exception):
                await cube_service.execute_query(payload)

    async def test_execute_query_continue_wait(self, cube_service, mock_load_success):
        # Create continue wait mock with status_code 202 (triggers ContinueWaitError)
        continue_wait_response = MagicMock()
        continue_wait_response.status_code = 202
        continue_wait_response.json.return_value = {"error": "Continue wait"}

        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            # First call returns 202 (continue wait), second returns success
            mock_client.post.side_effect = [continue_wait_response, mock_load_success]
            mock_get_client.return_value = mock_client

            payload = {"query": {"measures": ["orders.count"]}}
            result = await cube_service.execute_query(payload)

            # Should have made 2 calls (retry after continue wait)
            assert mock_client.post.call_count == 2

    async def test_execute_query_span_id(self, cube_service, mock_load_success):
        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_load_success
            mock_get_client.return_value = mock_client

            payload = {"query": {"measures": ["orders.count"]}}
            data = await cube_service.execute_query(payload, span_id="test-span-123")

            # Check x-request-id header
            call_args = mock_client.post.call_args
            headers = call_args.kwargs.get("headers", {})
            assert "x-request-id" in headers
            assert "test-span-123-span-" in headers["x-request-id"]

    async def test_preview_sql_returns_sql(self, cube_service):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sql": {"sql": ["SELECT * FROM orders"]}}

        with patch.object(cube_service, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            payload = {"query": {"measures": ["orders.count"]}}
            result = await cube_service.preview_sql(payload)

            assert "SELECT" in result


class TestAuth:
    def test_auth_header_bearer(self, cube_service):
        headers = cube_service._auth_headers()
        assert headers == {"Authorization": "Bearer test-token"}

    def test_auth_header_empty_token(self):
        # Create a service directly with empty token by bypassing CubeConfig validation
        # We test _auth_headers which only uses self.token
        service = CubeService.__new__(CubeService)
        service.token = ""
        headers = service._auth_headers()
        assert headers == {}
