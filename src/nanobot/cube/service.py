"""Cube semantic layer service."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx
import toons
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

if TYPE_CHECKING:
    from nanobot.config.schema import CubeConfig


class ContinueWaitError(Exception):
    """Raised when Cube API returns a continue wait response."""


class CubeService:
    """Service for interacting with Cube semantic layer."""

    def __init__(self, config: CubeConfig) -> None:
        self.cube_url = config.cube_url.rstrip("/")
        self.token = config.token
        self.cubejs_api_path = config.cubejs_api_path
        self.timeout = config.timeout
        self.schema_index = config.schema_index
        self.memory = config.memory
        self.request_span_enabled = config.request_span_enabled
        self.continue_wait_retry_interval = config.continue_wait_retry_interval
        self.continue_wait_retry_max_attempts = config.continue_wait_retry_max_attempts

        self._available: bool = False
        self._init_error: str | None = None
        self._schema: str | None = None
        self._compiler_id: str | None = None  # staleness check via /v1/meta compilerId
        self._client: httpx.AsyncClient | None = None
        self._schema_index: Any = None  # set by runner
        self._init_lock: asyncio.Lock = asyncio.Lock()

    @property
    def is_available(self) -> bool:
        """Return whether the Cube service is available."""
        return self._available

    @property
    def init_error(self) -> str | None:
        """Return the initialization error if any."""
        return self._init_error

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create and return the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def _api_url(self, path: str) -> str:
        """Build the full API URL for a given path."""
        return f"{self.cube_url}{self.cubejs_api_path}{path}"

    async def check_ready(self) -> bool:
        """Check if Cube is ready. Returns True if ready, False otherwise."""
        try:
            client = self._get_client()
            response = await client.get(f"{self.cube_url}/readyz")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Cube readiness check failed: {e}")
            return False

    async def check_live(self) -> bool:
        """Check if Cube is live. Returns True if live, False otherwise."""
        try:
            client = self._get_client()
            response = await client.get(f"{self.cube_url}/livez")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Cube liveness check failed: {e}")
            return False

    async def initialize(self) -> None:
        """Initialize the Cube service by checking readiness."""
        async with self._init_lock:
            if self._available:
                return  # already initialized (race winner)
            self._available = await self.check_ready()
            if not self._available:
                self._init_error = "Cube service failed readiness check"

    async def get_schema(self) -> str:
        """Return cached schema or reload if not cached."""
        if self._schema is None:
            await self.reload()
        return self._schema or ""

    async def get_schema_context(
        self,
        question: str | None,
    ) -> str:
        """Get schema context, using hybrid retrieval based on schema size.

        - No question → full schema
        - No schema index / not enabled / schema < threshold → full schema
        - Large schema + question → use schema_index.search and format results
        - If search returns empty, fall back to full schema
        """
        if question is None:
            return await self.get_schema()

        schema = await self.get_schema()
        schema_len = len(schema)

        if self._schema_index is None:
            return schema
        if not self.schema_index.enabled:
            return schema
        if schema_len < self.schema_index.threshold:
            return schema

        # Large schema + question → semantic search
        limit = self.schema_index.max_results
        results = await self._schema_index.search(question, limit=limit)
        if not results:
            return schema

        # Format results as context
        lines = ["## Schema Context (semantic search results)"]
        for i, (text, metadata, score) in enumerate(results, 1):
            lines.append(f"\n### Result {i}")
            lines.append(text)
        return "\n".join(lines)

    async def reload(self) -> None:
        """Reload the schema from Cube API."""
        try:
            client = self._get_client()
            headers = self._auth_headers()
            response = await client.get(
                self._api_url("/v1/meta"),
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            # Store schema as TOON (lossless compact format) with hash
            cubes = data.get("cubes", [])
            raw_json = json.dumps(data, sort_keys=True)
            self._schema_hash = hashlib.sha256(raw_json.encode()).hexdigest()
            self._schema = self._to_toon(cubes) if cubes else raw_json

            # Re-index if schema_index is available
            if self._schema_index is not None:
                cubes = data.get("cubes", [])
                await self._schema_index.index(cubes)

        except Exception as e:
            logger.error(f"Failed to reload Cube schema: {e}")
            self._schema = ""
            self._schema_hash = None
            raise

    async def execute_query(
        self,
        payload: dict,
        span_id: str | None = None,
    ) -> dict:
        """Execute a query against Cube API with retry on ContinueWaitError."""
        client = self._get_client()
        headers = self._auth_headers()
        retry_count = 0

        async def _do_request() -> dict:
            nonlocal retry_count
            # Build request headers
            request_headers = dict(headers)
            if span_id is not None and self.request_span_enabled:
                request_headers["x-request-id"] = f"{span_id}-span-{retry_count}"

            response = await client.post(
                self._api_url("/v1/load"),
                json=payload,
                headers=request_headers,
            )

            if response.status_code == 202:
                body = response.json()
                raise ContinueWaitError(body.get("error", "Continue wait"))

            response.raise_for_status()
            return response.json()

        retry = AsyncRetrying(
            wait=wait_fixed(self.continue_wait_retry_interval),
            stop=stop_after_attempt(self.continue_wait_retry_max_attempts),
            retry=retry_if_exception_type(ContinueWaitError),
        )

        async for attempt in retry:
            with attempt:
                retry_count = attempt.retry_state.attempt_number
                return await _do_request()

        # Should not reach here, but just in case
        raise RuntimeError("Exhausted retries for Cube query")

    async def preview_sql(self, payload: dict) -> str:
        """Preview SQL for a query payload."""
        client = self._get_client()
        headers = self._auth_headers()
        response = await client.post(
            self._api_url("/v1/sql"),
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        sql_obj = data.get("sql", {})
        sql_parts = sql_obj.get("sql", [])
        if sql_parts:
            return f"Generated SQL:\n```sql\n{sql_parts[0]}\n```"
        return "No SQL generated."

    @staticmethod
    def _format_as_markdown_table(data: list[dict]) -> str:
        """Format a list of dicts as a markdown table."""
        if not data:
            return ""

        headers = list(data[0].keys())
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in data:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _to_toon(cubes: list[dict]) -> str:
        """Convert cubes list to TOON format string."""
        return toons.dumps(cubes)
