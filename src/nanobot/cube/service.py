"""Cube semantic layer service."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

if TYPE_CHECKING:
    from nanobot.config.schema import CubeConfig


class CubeError(Exception):
    """Raised when Cube API returns an error."""

    def __init__(self, message: str, payload: dict, response_body: str = ""):
        self.payload = payload
        self.response_body = response_body
        super().__init__(message)


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
        self._schema_described: str | None = None  # Schema described text
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

    async def get_schema_context(
        self,
        question: str | None,
    ) -> str:
        """Get schema context, using hybrid retrieval based on schema size.

        - No question → full described schema
        - No schema index / not enabled / described schema < threshold → full described schema
        - Large described schema + question → use schema_index.search and format results
        - If search returns empty, fall back to full described schema
        """
        schema_described = self._schema_described or ""
        schema_len = len(schema_described)

        if question is None:
            return schema_described

        if self._schema_index is None:
            return schema_described
        if not self.schema_index.enabled:
            return schema_described
        if schema_len < self.schema_index.threshold:
            return schema_described

        # Large described schema + question → semantic search
        limit = self.schema_index.max_results
        results = await self._schema_index.search(question, limit=limit)
        if not results:
            return schema_described

        # Format results as context
        lines = ["## Schema Context (semantic search results)"]
        for i, (text, metadata, score) in enumerate(results, 1):
            lines.append(f"\n### Result {i}")
            lines.append(text)
        return "\n".join(lines)

    async def reload(self) -> None:
        """Reload the schema from Cube API if stale (compilerId changed)."""
        try:
            client = self._get_client()
            headers = self._auth_headers()
            response = await client.get(
                self._api_url("/v1/meta"),
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()

            # Check staleness via compilerId
            new_compiler_id = data.get("compilerId")
            if new_compiler_id and new_compiler_id == self._compiler_id:
                return

            cubes = data.get("cubes", [])

            # Compute schema described schema
            from nanobot.cube.schema_index import describe_full_schema

            self._schema_described = describe_full_schema(cubes) if cubes else ""

            self._compiler_id = new_compiler_id

            # Re-index if schema_index is available
            if self._schema_index is not None:
                await self._schema_index.index_cubes(cubes)

        except Exception as e:
            logger.error(f"Failed to reload Cube schema: {e}")
            self._compiler_id = None
            raise

    def _raise_error(self, response: httpx.Response, payload: dict) -> None:
        """Raise CubeError with parsed error details from response."""
        response_body = response.text
        try:
            error_body = response.json()
            error_msg = error_body.get("error", str(error_body))
        except Exception:
            error_msg = response_body or f"HTTP {response.status_code}"
        raise CubeError(error_msg, payload, response_body)

    async def execute_query(
        self,
        payload: dict,
        span_id: str | None = None,
    ) -> dict:
        """Execute a query against Cube API with retry on ContinueWaitError."""
        logger.debug(f"Cube query payload: {json.dumps(payload)}")

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

            # 4xx/5xx = error
            if response.status_code >= 400:
                self._raise_error(response, payload)

            body = response.json()
            # Continue wait payload = Continue wait
            if body.get("error") == "Continue wait":
                raise ContinueWaitError()

            return body

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
        logger.debug(f"Cube SQL preview payload: {json.dumps(payload)}")

        client = self._get_client()
        headers = self._auth_headers()
        response = await client.post(
            self._api_url("/v1/sql"),
            json=payload,
            headers=headers,
        )
        if response.status_code >= 400:
            self._raise_error(response, payload)
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
