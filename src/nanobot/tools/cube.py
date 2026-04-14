"""Cube semantic layer tools."""

from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.cube.service import CubeService
from nanobot.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.cube.query_memory import QueryMemory


def _generate_span_id() -> str:
    """Generate a span ID using uuid7 (Python 3.14+)."""
    import uuid

    return str(uuid.uuid7())


class CubeSchemaTool(Tool):
    name = "cube_schema"
    api_endpoint = "GET /cubejs-api/v1/meta"

    description = (
        "Retrieve the Cube data model schema — the catalog of available cubes, views, "
        "dimensions, measures, and segments. Call this FIRST to discover what data is "
        "available before crafting queries.\n\n"
        "The response is a JSON object with a top-level `cubes` array. Each cube object "
        "contains:\n"
        "  - `name` (str): Codename of the cube/view (e.g. 'Users', 'Orders').\n"
        "  - `type` (str): Either 'cube' or 'view'.\n"
        "  - `title` (str): Human-readable name.\n"
        "  - `measures` (array): Aggregatable metrics. Each measure has:\n"
        "      - `name` (str): Fully-qualified name in 'cube_name.measure_name' format.\n"
        "      - `type` (str): Always 'number' for measures.\n"
        "      - `aggType` (str): Aggregation type — 'count', 'sum', 'avg', "
        "'countDistinct', 'countDistinctApprox', 'min', 'max', etc.\n"
        "      - `title` / `shortTitle` (str): Human-readable labels.\n"
        "      - `drillMembers` (array of str): Dimension names available for drill-down.\n"
        "  - `dimensions` (array): Groupable attributes. Each dimension has:\n"
        "      - `name` (str): Fully-qualified 'cube_name.dimension_name'.\n"
        "      - `type` (str): One of 'string', 'number', 'time', 'boolean', or 'geo'.\n"
        "      - `title` / `shortTitle` (str): Human-readable labels.\n"
        "  - `segments` (array): Named pre-defined filters (each has `name`, `title`).\n"
        "  - `connectedComponent` (int): Cubes sharing the same value have a join path "
        "between them.\n\n"
        "USAGE TIP: For large schemas, pass a `question` to use embedding-based search "
        "and return only the most relevant cubes instead of the full catalog."
    )

    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Optional natural language question (e.g. 'How many orders were placed "
                    "last month by region?'). When provided AND the schema is large, triggers "
                    "embedding search to return only the most relevant cubes/dimensions/measures. "
                    "Omit to retrieve the full schema."
                ),
            },
        },
        "required": [],
    }

    def __init__(self, service: "CubeService") -> None:
        self._service = service

    async def execute(self, question: str | None = None, **kwargs: Any) -> str:
        # Lazy init (matches Mem0Client pattern)
        if self._service._client is None:
            await self._service.initialize()
        if not self._service.is_available:
            return f"Error: Cube is not available: {self._service.init_error}"

        result = await self._service.get_schema_context(question=question)
        logger.debug("cube_schema tool returning {} chars", len(result))
        return result


class CubeQueryTool(Tool):
    name = "cube_query"
    api_endpoint = "POST /cubejs-api/v1/load"

    description = (
        "Execute a Cube query to retrieve aggregated analytics data. Use cube_schema first "
        "to discover available measures and dimensions, then construct the payload here.\n\n"
        "## Query payload structure\n"
        "The `payload` must be an object with a required `query` key:\n"
        "```\n"
        "{\n"
        '  "query": {\n'
        '    "measures": ["orders.count", "orders.revenue"],\n'
        '    "dimensions": ["orders.status", "users.region"],\n'
        '    "filters": [{ "member": "orders.status", "operator": "equals", '
        '"values": ["shipped"] }],\n'
        '    "timeDimensions": [{ "dimension": "orders.created_at", '
        '"dateRange": ["2024-01-01", "2024-12-31"], "granularity": "month" }],\n'
        '    "segments": ["orders.completedOrders"],\n'
        '    "order": { "orders.revenue": "desc" },\n'
        '    "limit": 100,\n'
        '    "offset": 0,\n'
        '    "timezone": "UTC"\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "## Key query properties\n"
        "  - `measures` (str[]): Aggregatable metrics from the schema (e.g. "
        "'orders.count', 'orders.revenue').\n"
        "  - `dimensions` (str[]): Group-by attributes (e.g. 'users.city', "
        "'orders.status').\n"
        "  - `segments` (str[]): Named pre-defined filters from the data model.\n"
        "  - `timeDimensions` (array): Objects with `dimension`, `dateRange`, and "
        "optional `granularity` (year|quarter|month|week|day|hour|minute|second).\n"
        "  - `order` (object): Keys are measure/dimension names, values are 'asc' or "
        "'desc'. Pass {} to disable default ordering.\n"
        "  - `limit` (int): Max rows (default 10000, max 50000).\n"
        "  - `offset` (int): Rows to skip for pagination.\n"
        "  - `timezone` (str): TZ database name (e.g. 'America/Los_Angeles').\n"
        "  - `total` (bool): If true, returns total row count ignoring limit/offset.\n"
        "  - `ungrouped` (bool): If true, returns raw rows without GROUP BY.\n\n"
        "## Filter format\n"
        'Each filter is `{ "member": "<name>", "operator": "<op>", "values": [...] }`.\n'
        "Available operators:\n"
        "  - For string dims: equals, notEquals, contains, notContains, startsWith, "
        "notStartsWith, endsWith, notEndsWith, set, notSet.\n"
        "  - For number dims & measures: equals, notEquals, gt, gte, lt, lte, set, notSet.\n"
        "  - For time dims: inDateRange, notInDateRange, beforeDate, beforeOrOnDate, "
        "afterDate, afterOrOnDate, set, notSet.\n"
        '  - Logical combinators: wrap filters in `{ "and": [...] }` or `{ "or": [...] }`. '
        "Cannot mix dimension and measure filters within the same logical group.\n\n"
        "## dry_run mode\n"
        "When `dry_run` is true, calls /v1/sql instead of /v1/load. Returns the generated "
        "SQL query without executing it. Useful for debugging, validating queries, or "
        "understanding how Cube translates the payload into SQL.\n\n"
        "## Member naming convention\n"
        "All measure and dimension names use the format 'cube_name.member_name' "
        "(e.g. 'orders.count', 'users.email'). Time dimensions can optionally append "
        "granularity: 'orders.created_at.month'."
    )

    parameters = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "description": (
                    "The full request payload. Must contain a 'query' object. "
                    "Example:\n"
                    '{"query": {"measures": ["orders.count"], "dimensions": '
                    '["orders.status"], "limit": 50}}\n'
                    "Optional top-level keys: 'queryType' (set to 'multi' for data "
                    "blending / compare date range queries), 'cache' (cache control)."
                ),
            },
            "dry_run": {
                "type": "boolean",
                "description": (
                    "When true, returns the generated SQL via /v1/sql without executing "
                    "the query. Use this to validate a payload or inspect the SQL Cube "
                    "would generate before actually running it."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "Optional natural language description of what this query answers "
                    "(e.g. 'Total revenue by month in 2024'). Used to automatically save "
                    "the query to history so it can be found later via cube_search. "
                    "Recommended to always provide this."
                ),
            },
        },
        "required": ["payload"],
    }

    def __init__(self, service: "CubeService", query_memory: "QueryMemory | None" = None) -> None:
        self._service = service
        self._query_memory = query_memory

    async def execute(
        self,
        payload: dict,
        dry_run: bool = False,
        question: str | None = None,
        **kwargs: Any,
    ) -> str:
        # Lazy init (matches Mem0Client pattern)
        if self._service._client is None:
            await self._service.initialize()
        if not self._service.is_available:
            return f"Error: Cube is not available: {self._service.init_error}"

        # Generate span_id at tool level
        span_id = _generate_span_id() if self._service.request_span_enabled else None

        if dry_run:
            return await self._service.preview_sql(payload)

        try:
            result = await self._service.execute_query(payload, span_id=span_id)
            # Success — save to history (only if memory is enabled and question provided)
            if question and self._query_memory and self._query_memory.is_available:
                import json

                await self._query_memory.store(question, json.dumps(payload))
            # Cube /v1/load returns {"data": [...]} or similar structure
            data = result.get("data", []) if isinstance(result, dict) else result
            return self._service._format_as_markdown_table(data)
        except Exception as e:
            return f"Error: {e}"


class CubeSearchTool(Tool):
    name = "cube_search"

    description = (
        "Search the history of previously executed Cube queries by natural language "
        "similarity. Returns matching question–payload pairs ranked by semantic relevance.\n\n"
        "## When to use\n"
        "  - Before writing a new query from scratch, check if a similar query was already "
        "crafted successfully. Reusing a proven payload avoids schema mistakes.\n"
        "  - When a user asks a question that closely resembles a past question, the stored "
        "payload can be reused or adapted.\n\n"
        "## What it returns\n"
        "A list of matches, each containing:\n"
        "  - The original natural language question.\n"
        "  - The full Cube query payload that was used.\n"
        "  - A similarity score (higher = more relevant).\n\n"
        "## Workflow\n"
        "  1. Call cube_search with the user's question to find similar past queries.\n"
        "  2. If a good match is found, use its payload as-is or adapt it.\n"
        "  3. If no match is found, call cube_schema to explore the data model, then "
        "craft a new payload for cube_query.\n"
        "  4. Always pass a `question` to cube_query so successful queries are saved for "
        "future reuse."
    )

    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "Natural language question to search for semantically similar past "
                    "queries (e.g. 'What were total sales last quarter by region?'). "
                    "The search uses embedding similarity, so paraphrases and related "
                    "phrasing will also match."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of similar past queries to return. Default is 5. "
                    "Increase if you want more candidates to choose from."
                ),
            },
        },
        "required": ["question"],
    }

    def __init__(self, query_memory: "QueryMemory") -> None:
        self._query_memory = query_memory

    async def execute(self, question: str, limit: int = 5, **kwargs: Any) -> str:
        if not self._query_memory or not self._query_memory.is_available:
            return "Error: Cube query history is not available."

        pairs = await self._query_memory.search(question, limit=limit)

        if not pairs:
            return "No similar queries found."

        lines = ["## Similar Past Queries\n"]
        for i, (q, payload, score) in enumerate(pairs, 1):
            lines.append(f"### Query {i} (score: {score:.2f})")
            lines.append(f"**Question:** {q}")
            lines.append(f"**Payload:** ```json\n{payload}\n```")
            lines.append("")

        return "\n".join(lines)
