"""End-to-end tests for Cube semantic layer against live Cube + DuckDB stack.

These tests use the existing fixtures at tests/e2e/cube/fixtures/
- compose.yaml: Cube.js + DuckDB stack
- conf/model/customers.yaml: Cube model definition
- conf/db/jaffle_shop.duckdb: DuckDB database
"""

import pytest

from nanobot.config.schema import CubeConfig
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
