# Cube Semantic Layer Implementation Plan

## Overview

Implement a Cube-based text-to-SQL feature for nanobot. LLM sees Cube schema via `cube_schema`, executes queries via `cube_query`.

```
User: "How many orders by status?"
    │
    │ cube_schema(question="How many orders by status?")
    │   → GET /cubejs-api/v1/meta
    │   → if schema < threshold → return full flat text (best for small schemas)
    │   → if schema ≥ threshold → ChromaDB embedding search for relevant cubes
    │
    │ LLM builds JSON query
    │
    │ cube_query(payload, dry_run=False)
    │   → POST /cubejs-api/v1/load
    │   → Returns markdown table
    │
    │ (On success + question)
    │ → SqlMemory stores (nl_question, payload)
```

---

## Tool Inventory

| Tool | Name | API Endpoint | Purpose |
|------|------|-------------|---------|
| Schema | `cube_schema` | `GET /cubejs-api/v1/meta` | Feed LLM schema (full or context-searched) |
| Query | `cube_query` | `POST /cubejs-api/v1/load` | Execute query |
| Search | `cube_search` | — | Search past queries from ChromaDB |

**Hybrid schema retrieval:** `cube_schema` accepts an optional `question` parameter. When the schema is large (≥ `schema_index.threshold` chars), the question is used for embedding-based retrieval of relevant cubes via `CubeSchemaIndex`. For small schemas, the full text is returned.

---

## Phase 1: Config

**File:** `src/nanobot/config/schema.py`

### EmbedderConfig

```python
class EmbedderConfig(Base):
    """Embedder configuration — holds full ProviderConfig for api_key resolution."""

    provider: ProviderConfig  # full ProviderConfig with backend + get_api_key()
    model: str = "text-embedding-3-small"
```

### RerankerConfig

```python
class RerankerConfig(Base):
    """Optional reranker configuration — holds full ProviderConfig for api_key resolution."""

    provider: ProviderConfig | None = None   # full ProviderConfig
    model: str | None = None
```

### CubeSchemaIndexConfig

```python
class CubeSchemaIndexConfig(Base):
    """Cube schema index configuration — ChromaDB-backed schema element indexing.

    Uses the same nanobot provider config format as memory.embedder/memory.reranker.
    Both memory and schema_index can use different embedder/reranker providers
    (they share the same ChromaDB but use different collections with different vector dimensions).
    """

    enabled: bool = True
    threshold: int = 30_000                  # Schema size (chars) above which embedding search replaces full dump
    max_results: int = 10                    # Max results for schema embedding search
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    reranker: RerankerConfig | None = None
```

### CubeMemoryConfig

```python
class CubeMemoryConfig(Base):
    """Cube memory configuration — NL→SQL query history via ChromaDB.

    Uses the same nanobot provider config format for embedder/reranker.
    Can use different embedder/reranker from schema_index (different collections, different vector dimensions).
    """

    enabled: bool = True
    max_results: int = 5
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    reranker: RerankerConfig | None = None
```

### CubeConfig

```python
class CubeConfig(Base):
    """Cube semantic layer configuration.

    Auth: a single `token` field used as a Bearer token.
    Header format: Authorization: Bearer {token}

    memory and schema_index each have their own embedder/reranker config in nanobot
    provider format. They both use ChromaDB but with separate collections, so they
    can use different embedding models (e.g., different vector dimensions).
    """

    enabled: bool = False
    cube_url: str = ""                       # e.g. "https://cube.example.com"
    token: str = ""                          # Bearer token for Cube API auth
    cubejs_api_path: str = "/cubejs-api"     # default Cube API base path
    timeout: float = 30.0                    # HTTP request timeout in seconds
    request_span_enabled: bool = True                # Generate x-request-id headers for tracing
    continue_wait_retry_interval: float = 1.0        # Seconds to wait between continue-wait retries
    continue_wait_retry_max_attempts: int = 5        # Max retry attempts before giving up
    memory: CubeMemoryConfig = Field(default_factory=CubeMemoryConfig)
    schema_index: CubeSchemaIndexConfig = Field(default_factory=CubeSchemaIndexConfig)
```

### Integration into root Config class

Add `cube` field to the existing `Config(BaseSettings)` class in the same file:

```python
class Config(BaseSettings):
    """Root configuration for nanobot."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cube: CubeConfig = Field(default_factory=CubeConfig)          # ← NEW

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agent.workspace).expanduser()

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
```

**YAML config example:**

```yaml
cube:
  enabled: true
  cube_url: "https://cube.example.com"
  token: "${CUBE_API_TOKEN}"            # Bearer token for Authorization header
  timeout: 30.0
  request_span_enabled: true            # generate x-request-id headers for tracing
  continue_wait_retry_interval: 1.0     # seconds to wait between continue-wait retries
  continue_wait_retry_max_attempts: 5   # max retry attempts before giving up
  memory:
    enabled: true
    max_results: 5
    embedder:
      provider: "openai"
      model: "text-embedding-3-small"
    reranker:
      provider: "openai"
      model: "bge-reranker"
  schema_index:
    enabled: true
    threshold: 30000                    # chars — above this, embedding search replaces full dump
    max_results: 10                     # max results for schema embedding search
    embedder:
      provider: "openai"
      model: "text-embedding-3-small"  # can differ from memory.embedder
    reranker:
      provider: "openai"
      model: "bge-reranker"           # can differ from memory.reranker
```

> **Note:** This is WIP development — everything can be recreated. No migration needed for mem0 or any other component.

**CLI integration:** The `cube` config is loaded automatically via the existing YAML/env loading path. No CLI commands needed — Cube is always accessed through LLM tools, not directly by the user.

---

## Phase 2: CubeService

**File:** `src/nanobot/cube/service.py`

HTTP client wrapping Cube REST API. **Plain class** (not a BaseModel). All HTTP methods are `async` using `httpx.AsyncClient`. Lazy import `httpx`. Initialization (health check) is deferred to first tool use, matching the `Mem0Client` pattern in the codebase.

### Authentication

All requests use Bearer token auth: `Authorization: Bearer {token}`. The `token` config field holds the bearer token.

### Class: CubeService

```python
import asyncio
import json
import time
import uuid
from typing import Any

import httpx
from loguru import logger
from tenacity import AsyncRetrying, wait_fixed, retry_if_exception_type, stop_after_attempt

class CubeService:
    """HTTP client for Cube REST API. Plain class — not a BaseModel."""

    def __init__(self, config: "CubeConfig") -> None:
        self.cube_url: str = config.cube_url.rstrip("/")
        self.token: str = config.token
        self.cubejs_api_path: str = config.cubejs_api_path
        self.timeout: float = config.timeout
        # Store sub-configs directly (no need to unpack individual fields)
        self.schema_index: "CubeSchemaIndexConfig" = config.schema_index
        self.memory: "CubeMemoryConfig" = config.memory
        # Request span tracking (span_id/request_seq are per-call, not stored here)
        self.request_span_enabled: bool = config.request_span_enabled
        self.continue_wait_retry_interval: float = config.continue_wait_retry_interval
        self.continue_wait_retry_max_attempts: int = config.continue_wait_retry_max_attempts

        # Internal state
        self._available: bool = False
        self._init_error: str | None = None
        self._schema: str | None = None          # full flat schema
        self._schema_hash: str | None = None     # SHA-256 of raw meta response (for staleness detection)
        self._client: httpx.AsyncClient | None = None
        self._schema_index: "CubeSchemaIndex | None" = None  # set by runner if enabled
        self._init_lock: asyncio.Lock = asyncio.Lock()  # prevent concurrent initialize()

    # --- Public properties ---------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether Cube is reachable and initialized."""
        return self._available

    @property
    def init_error(self) -> str | None:
        """Initialization error message, if any."""
        return self._init_error

    # --- Private helpers -----------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-create async httpx client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        """Build auth headers. Health endpoints skip auth."""
        if not self.token:
            return {}
        return {"Authorization": f"Bearer {self.token}"}

    def _api_url(self, path: str) -> str:
        """Build full URL for API endpoints (includes base path)."""
        return f"{self.cube_url}{self.cubejs_api_path}{path}"

    # --- Health check --------------------------------------------------------

    async def check_ready(self) -> bool:
        """Check /readyz — no auth required, no base path prefix."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.cube_url}/readyz", timeout=10.0)
            if resp.status_code == 200 and resp.json().get("health") == "HEALTH":
                return True
            return False
        except Exception as e:
            logger.warning("Cube /readyz check failed: {}", e)
            return False

    async def check_live(self) -> bool:
        """Check /livez — no auth required, no base path prefix."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.cube_url}/livez", timeout=10.0)
            if resp.status_code == 200 and resp.json().get("health") == "HEALTH":
                return True
            return False
        except Exception as e:
            logger.warning("Cube /livez check failed: {}", e)
            return False

    # --- Lifecycle -----------------------------------------------------------

    async def initialize(self) -> None:
        """Verify Cube is reachable and set _available. Never raises.

        Uses asyncio.Lock to prevent concurrent initialization from multiple
        tool calls racing to the health check endpoint.

        Note: _schema_index is set by the runner before tools use the service.
        This method only performs the health check.
        """
        async with self._init_lock:
            try:
                start = time.monotonic()
                ready = await self.check_ready()
                elapsed = time.monotonic() - start
                if ready:
                    self._available = True
                    self._init_error = None
                    logger.info("Cube ready ({}ms)", int(elapsed * 1000))
                else:
                    self._init_error = "Cube /readyz returned non-HEALTH status"
                    logger.warning("Cube not healthy: {}", self._init_error)
            except Exception as e:
                self._init_error = str(e)
                logger.error("Cube init failed: {}", e)

    # --- Schema --------------------------------------------------------------

    async def get_schema(self) -> str:
        """Fetch and cache schema as flat text. Returns cached version if available."""
        return self._schema or await self.reload()

    async def get_schema_context(self, question: str | None = None) -> str:
        """Hybrid schema retrieval — full text or embedding-searched context.

        Uses a single threshold: schema_index.threshold (default 30,000 chars).
        When the full schema text is >= this threshold AND a question is provided,
        embedding search retrieves only relevant cubes. Otherwise the full text
        is returned (the LLM context window handles sizing).

        Decision logic:
          1. No question → return full schema text
          2. No schema index available → return full schema text
          3. Schema index disabled → return full schema text
          4. Schema < threshold → return full schema text
          5. Schema ≥ threshold + question + index available → embedding search

        **Note:** The threshold check uses the cached schema length. If the schema grows
        past `schema_index.threshold` between calls (e.g., after a Cube model update),
        the user should call `reload()` manually to re-fetch and re-index the schema.
        """
        schema = await self.get_schema()

        # Conditions for returning full schema
        if not question:
            return schema
        if not self._schema_index or not self._schema_index.is_available:
            return schema
        if not self.schema_index.enabled:
            return schema
        if len(schema) < self.schema_index.threshold:
            return schema

        # Large schema + question + index available → embedding search
        results = self._schema_index.search(question, limit=self.schema_index.max_results)
        if not results:
            logger.warning("Schema index search returned no results; falling back to full schema")
            return schema

        context_parts = [f"Relevant schema elements for: \"{question}\"\n"]
        for text, metadata, score in results:
            context_parts.append(f"### {metadata.get('cube_name', 'unknown')} ({metadata.get('item_type', 'cube')})")
            context_parts.append(text)
            context_parts.append("")
        return "\n".join(context_parts)

    async def reload(self) -> str:
        """Re-fetch schema from /v1/meta and cache it. Also re-indexes if schema index is enabled."""
        try:
            start = time.monotonic()
            client = await self._get_client()
            resp = await client.get(
                self._api_url("/v1/meta"),
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            elapsed = time.monotonic() - start
            logger.info("Cube /v1/meta fetched ({}ms)", int(elapsed * 1000))

            data = resp.json()
            cubes = data.get("cubes", [])
            if not cubes:
                self._schema = "No cubes found in Cube schema."
                return self._schema

            # Store hash for future staleness detection
            raw_json = resp.text
            import hashlib
            self._schema_hash = hashlib.sha256(raw_json.encode()).hexdigest()

            self._schema = self._to_toon(cubes)

            # Re-index schema elements if index is available
            if self._schema_index and self._schema_index.is_available:
                self._schema_index.index_cubes(cubes)
                logger.info("Schema index updated ({} cubes)", len(cubes))

            return self._schema
        except Exception as e:
            logger.error("Failed to fetch Cube schema: {}", e)
            return f"Error fetching Cube schema: {e}"

    @staticmethod
    def _to_toon(cubes: list[dict]) -> str:
        """Convert Cube meta response to TOON format for LLM context.

        TOON is a compact, lossless format designed for LLM contexts.
        Provides 30-60% token reduction vs JSON.
        """
        import toons
        return toons.dumps(cubes)

    # --- Query execution -----------------------------------------------------

    class ContinueWaitError(Exception):
        """Raised when Cube returns 'Continue wait' response, signaling a retry."""
        pass

    async def execute_query(self, payload: dict, span_id: str | None = None) -> list[dict]:
        """Execute a Cube query via POST /v1/load. Returns data list or raises."""

        async def do_post() -> list[dict]:
            resp = await client.post(
                self._api_url("/v1/load"),
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            body = resp.json()
            if body == {"error": "Continue wait"}:
                raise self.ContinueWaitError()
            return body.get("data", [])

        retry_count = 0
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.continue_wait_retry_max_attempts),
            wait=wait_fixed(self.continue_wait_retry_interval),
            retry=retry_if_exception_type(self.ContinueWaitError),
            reraise=True,
        ):
            with attempt:
                retry_count = attempt.retry_state.attempt_number
                start = time.monotonic()
                headers = self._auth_headers()
                if span_id:
                    headers["x-request-id"] = f"{span_id}-span-{retry_count}"
                client = await self._get_client()
                return await do_post()

    async def preview_sql(self, payload: dict) -> str:
        """Get generated SQL via POST /v1/sql. Returns SQL string or error."""
        try:
            start = time.monotonic()
            headers = self._auth_headers()

            client = await self._get_client()
            resp = await client.post(
                self._api_url("/v1/sql"),
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            elapsed = time.monotonic() - start
            logger.info("Cube /v1/sql completed ({}ms)", int(elapsed * 1000))

            body = resp.json()
            sql_obj = body.get("sql", {})
            sql_parts = sql_obj.get("sql", [])
            if sql_parts:
                return f"Generated SQL:\n```sql\n{sql_parts[0]}\n```"
            return "No SQL generated."
        except Exception as e:
            logger.error("Cube SQL preview failed: {}", e)
            return f"Error previewing SQL: {e}"

    # --- Output formatting ---------------------------------------------------

    @staticmethod
    def _format_as_markdown_table(data: list[dict]) -> str:
        """Format query results as a markdown table."""
        if not data:
            return "No data."

        keys = list(data[0].keys())
        header = "| " + " | ".join(keys) + " |"
        separator = "| " + " | ".join("---" for _ in keys) + " |"
        rows = []
        for row in data:
            cells = [str(row.get(k, "")) for k in keys]
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join([header, separator] + rows)

    # --- Cleanup -------------------------------------------------------------

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
```

### Method summary

| Method | API | Purpose |
|--------|-----|---------|
| `async check_ready()` | `GET /readyz` | Health check — no auth, no base path |
| `async check_live()` | `GET /livez` | Liveness check — no auth, no base path |
| `async initialize()` | `GET /readyz` | Verify Cube reachable, set `_available` (lazy — called on first tool use) |
| `async get_schema()` | `GET /cubejs-api/v1/meta` | Fetch schema → full flat text (cached) |
| `async get_schema_context(question)` | — | Hybrid: full text or embedding search based on schema size |
| `async execute_query(payload, span_id)` | `POST /cubejs-api/v1/load` | Run query → `list[dict]`; **retries Continue-wait with fixed delay, raises on timeout/error** |
| `async preview_sql(payload)` | `POST /cubejs-api/v1/sql` | Get generated SQL via /v1/sql endpoint |
| `async reload()` | `GET /cubejs-api/v1/meta` | Re-fetch schema + re-index |
| `async close()` | — | Close async HTTP client |

### Schema output format (TOON — lossless)

```toon
cubes: [3,]{name,title,description,dimensions,measures}:
  orders,Orders,Order transactions,
    [4,]{name,type,description}:
      orders.order_id,number,Unique order identifier
      orders.status,string,Current order status
      orders.customer_name,string,Customer name
      orders.created_at,time,Order creation timestamp
    ,
    [2,]{name,aggType,description}:
      orders.order_count,count,Total number of orders
      orders.total_revenue,sum,Total revenue
    ,
  ,
  customers,Customers,Customer master data,
    [3,]{name,type,description}:
      customers.customer_id,number,Unique customer identifier
      customers.name,string,Customer full name
      customers.city,string,Customer city
    ,
    [1,]{name,aggType,description}:
      customers.customer_count,count,Total number of customers
    ,
  ,
  products,Products,Product catalog,
    [2,]{name,type,description}:
      products.product_id,number,Unique product identifier
      products.category,string,Product category
    ,
    [1,]{name,aggType,description}:
      products.product_count,count,Total number of products
    ,
  ,
```

### Query result output format

`POST /v1/load` response → formatted markdown:

```
| orders.status | orders.order_count |
|---------------|--------------------|
| completed     | 150                |
| pending       | 42                 |
```

### Error handling

- `execute_query` raises exceptions on error (caught and converted to error string by tool layer)
- `preview_sql` returns error string on failure (never raises)
- `reload()` returns error string on failure (never raises)
- `initialize()` sets `_init_error` on failure — startup never crashes

### Logging

All HTTP calls are logged via loguru:
- **Request latency** — measured with `time.monotonic()`, logged in milliseconds
- **Errors** — logged at `error` level with full context
- **Warnings** — health check failures logged at `warning` level
- **Info** — successful fetches with timing

---

## Phase 3: Memory & Indexing

Two separate ChromaDB-backed stores with distinct responsibilities:

| Store | Purpose | Collection |
|-------|---------|------------|
| `SqlMemory` | NL→SQL query history (existing plan) | `cube_query_history` |
| `CubeSchemaIndex` | Schema element indexing for context retrieval | `cube_schema_items` |

---

### Reranking

ChromaDB has **no built-in reranker**. Reranking is a **post-retrieval step** done after ChromaDB returns initial results. Only OpenAI backend is supported.

### Embedding Adapter

ChromaDB's built-in `OpenAIEmbeddingFunction` hardcodes OpenAI's endpoint — it doesn't support custom `base_url`. We need our own adapter that respects the nanobot `ProviderConfig` (which has `base_url` for OpenAI-compatible providers like DeepSeek, Groq, etc.).

```python
import asyncio
import openai
from chromadb.api.types import Embeddings

class CubeEmbeddingFunction(Embeddings):
    """Adapter that bridges nanobot EmbedderConfig to ChromaDB's Embeddings protocol.

    Supports OpenAI-compatible backends via base_url. ChromaDB calls embed_documents()
    for indexing and embed_query() for queries.
    """

    def __init__(self, config: "EmbedderConfig") -> None:
        self.config = config
        self._client: "openai.AsyncOpenAI | None" = None

    def _get_client(self) -> "openai.AsyncOpenAI":
        """Lazy-create OpenAI-compatible async client."""
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.provider.get_api_key(),
                base_url=self.config.provider.base_url,
            )
        return self._client

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed a list of texts for indexing. Blocking — wrap with asyncio.to_thread if needed."""
        return asyncio.run(self._embed_async(input))

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Blocking — wrap with asyncio.to_thread if needed."""
        return asyncio.run(self._embed_async([query]))[0]

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        """Internal async embed implementation."""
        if self.config.provider.backend != "openai":
            raise ValueError(f"Unsupported embedding backend: {self.config.provider.backend}. Only 'openai' backend is supported.")
        client = self._get_client()
        resp = await client.embeddings.create(model=self.config.model, input=texts)
        return [item.embedding for item in resp.data]
```

### Reranking

```python
async def _rerank(
    reranker: "RerankerConfig",
    query: str,
    texts: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Call rerank API via OpenAI SDK. Only OpenAI backend is supported."""
    provider_cfg = reranker.provider
    if provider_cfg is None:
        raise ValueError("Reranker provider not configured")
    if provider_cfg.backend != "openai":
        raise ValueError(f"Unsupported reranker backend: {provider_cfg.backend}. Only 'openai' backend is supported.")
    client = openai.AsyncOpenAI(api_key=provider_cfg.get_api_key(), base_url=provider_cfg.base_url)
    resp = await client.ranking.rerank(
        query=query,
        documents=texts,
        top=top_k,
        model=reranker.model,
    )
    return [(result.document.text, result.relevance_score) for result in resp.results]
```

---

### SqlMemory

**File:** `src/nanobot/cube/sql_memory.py`

ChromaDB-backed store for `(nl_question, payload)` pairs.

```python
class SqlMemory:
    """ChromaDB-backed store for (question, payload) pairs."""

    def __init__(
        self,
        persist_dir: Path,
        max_results: int = 5,
        embedder: EmbedderConfig | None = None,
        reranker: RerankerConfig | None = None,
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
        """Setup ChromaDB with CubeEmbeddingFunction, create collection.

        Uses CubeEmbeddingFunction which respects ProviderConfig base_url for OpenAI-compatible providers.
        ChromaDB calls are blocking — wrap with asyncio.to_thread() in async context.
        """
        import chromadb

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        ef = CubeEmbeddingFunction(config=self.embedder)
        self._collection = self._client.get_or_create_collection(
            name="cube_query_history",
            embedding_function=ef,
        )
        self._available = True

    async def store(self, question: str, payload: str) -> None:
        """Save pair with embedding. ChromaDB call wrapped with asyncio.to_thread."""
        ...

    async def search(self, question: str, limit: int | None = None) -> list[tuple[str, str, float]]:
        """Semantic search → list of (question, payload, score) tuples. ChromaDB call wrapped with asyncio.to_thread."""
        ...

    async def clear(self) -> None:
        """Clear all stored pairs. ChromaDB call wrapped with asyncio.to_thread."""
        ...
```

| Method | Description |
|--------|-------------|
| `initialize()` | Setup ChromaDB with embedder config, create collection |
| `store(question, payload)` | Save pair with embedding (async, uses asyncio.to_thread) |
| `search(question, limit)` | Semantic search → list of `(question, payload, score)` tuples (async, uses asyncio.to_thread) |
| `clear()` | Clear all stored pairs (async, uses asyncio.to_thread) |

- Collection: `cube_query_history`
- Embeddings: Use embedder config (provider + model) for ChromaDB embeddings
- Persist to `{workspace}/chroma/` (directory containing `chroma.db` — shared with mem0)
- **Note:** mem0 is being updated to use `{workspace}/chroma/` and `chroma.db`, so there is no conflict with Cube's ChromaDB usage.
- `is_available` property for external access

---

### CubeSchemaIndex

**File:** `src/nanobot/cube/schema_index.py`

Indexes each cube as a separate ChromaDB record for embedding-based context retrieval when schemas are too large for full-text injection.

```python
class CubeSchemaIndex:
    """ChromaDB-backed index of Cube schema elements for context retrieval.

    Each cube is indexed as one record containing its dimensions, measures,
    and segments. This allows semantic search to retrieve only the relevant
    cubes when the full schema is too large for the LLM context window.
    """

    def __init__(
        self,
        persist_dir: Path,
        max_results: int = 10,
        embedder: EmbedderConfig | None = None,
        reranker: RerankerConfig | None = None,
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
        """Setup ChromaDB with CubeEmbeddingFunction, create `cube_schema_items` collection.

        Uses CubeEmbeddingFunction which respects ProviderConfig base_url for OpenAI-compatible providers.
        ChromaDB calls are blocking — wrap with asyncio.to_thread() in async context.
        """
        import chromadb

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        ef = CubeEmbeddingFunction(config=self.embedder)
        self._collection = self._client.get_or_create_collection(
            name="cube_schema_items",
            embedding_function=ef,
        )
        self._available = True

    async def index_cubes(self, cubes: list[dict]) -> None:
        """Index (or re-index) all cubes from /v1/meta response.

        Each cube becomes one record:
          - id: "cube::{cube_name}"
          - text: TOON representation of the full cube (lossless — preserves all attributes)
          - metadata: {"cube_name": "...", "item_type": "cube"}
        Clears existing records before re-indexing (full rebuild on each reload).
        ChromaDB calls wrapped with asyncio.to_thread.
        """
        import toons
        ...

    async def search(
        self,
        question: str,
        limit: int | None = None,
    ) -> list[tuple[str, dict, float]]:
        """Semantic search → list of (text, metadata, score) tuples.

        ChromaDB search is wrapped with asyncio.to_thread.
        If reranker is configured, post-processes results through rerank API.
        """
        ...

    async def clear(self) -> None:
        """Clear all indexed schema items. ChromaDB call wrapped with asyncio.to_thread."""
        ...
```

| Method | Description |
|--------|-------------|
| `initialize()` | Setup ChromaDB with embedder config, create `cube_schema_items` collection |
| `index_cubes(cubes)` | Index each cube as a separate record (full rebuild, async with asyncio.to_thread) |
| `search(question, limit)` | Semantic search → list of `(text, metadata, score)` tuples (async with asyncio.to_thread, optionally reranks) |
| `clear()` | Clear all indexed schema items (async with asyncio.to_thread) |

- Collection: `cube_schema_items`
- Embeddings: Use embedder config (provider + model) for ChromaDB embeddings
- Persist to `{workspace}/chroma/` (directory containing `chroma.db` — shared with mem0)
- **Note:** mem0 is being updated to use `{workspace}/chroma/` and `chroma.db`, so there is no conflict with Cube's ChromaDB usage.
- Each record: one cube with all its dimensions, measures, and segments as a single text block
- `index_cubes` performs a full rebuild — clears and re-inserts all records

**Record structure example:**

```
id:       "cube::orders"
text:     |
  name: orders
  title: Orders
  description: This table tracks customer orders.
  dimensions: [2,]{name,type,description}:
    orders.order_id,number,Unique order identifier
    orders.status,string,Current order status
  measures: [1,]{name,aggType,description}:
    orders.order_count,count,Total number of orders
metadata: {"cube_name": "orders", "item_type": "cube"}
```

**Why one record per cube (not per dimension/measure):**
- A question like "how many orders by status?" needs both the `orders.status` dimension AND the `orders.order_count` measure — they're useless in isolation
- Cubes typically have 5–30 members — still small enough for one embedding to capture
- Fewer records = faster search and simpler index management

---

## Phase 4: Tools

**File:** `src/nanobot/tools/cube.py`

### Helper

```python
def _generate_span_id() -> str:
    """Generate a span ID using uuid7 (Python 3.14+)."""
    import uuid
    return str(uuid.uuid7())
```

### CubeSchemaTool

```python
class CubeSchemaTool(Tool):
    name = "cube_schema"
    description = "Get the Cube schema: cubes, dimensions, measures, and descriptions. Optionally provide a question to retrieve only relevant schema elements for large schemas."
    api_endpoint = "GET /cubejs-api/v1/meta"

    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Optional NL question. When provided AND schema is large, triggers embedding search for relevant cubes.",
            },
        },
        "required": [],
    }

    async def execute(self, question: str | None = None, **kwargs) -> str:
        # Lazy init (matches Mem0Client pattern)
        if self._service._client is None:
            await self._service.initialize()
        if not self._service.is_available:
            return f"Error: Cube is not available: {self._service.init_error}"

        return await self._service.get_schema_context(question=question)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str \| None` | `None` | Optional NL question. When provided AND schema is large, triggers embedding search for relevant cubes. |

**Behavior:**
- No `question` → always returns full schema text
- `question` + small schema → returns full schema text (embedding search not needed)
- `question` + large schema → returns only relevant cubes via `CubeSchemaIndex`

### CubeQueryTool

```python
class CubeQueryTool(Tool):
    name = "cube_query"
    description = (
        "Execute a query against Cube via the /v1/load endpoint. "
        "Set dry_run=True to call /v1/sql and return the generated SQL without executing via /v1/load."
    )
    api_endpoint = "POST /cubejs-api/v1/load"

    parameters = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "description": "Full payload for /v1/load: {query: {...}, queryType?: string, cache?: string}. "
                               "query is required; queryType and cache are optional.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "When True, calls /v1/sql to return generated SQL instead of executing via /v1/load.",
            },
            "question": {
                "type": "string",
                "description": "Optional natural language question for query history auto-save.",
            },
        },
        "required": ["payload"],
    }

    async def execute(
        self,
        payload: dict,
        dry_run: bool = False,
        question: str | None = None,
        **kwargs
    ) -> str:
        # Lazy init (matches Mem0Client pattern)
        if self._service._client is None:
            await self._service.initialize()
        if not self._service.is_available:
            return f"Error: Cube is not available: {self._service.init_error}"

        # Generate span_id at tool level (per-request, no mutable service state)
        # Uses uuid.uuid7
        span_id = _generate_span_id() if self._service.request_span_enabled else None

        if dry_run:
            return await self._service.preview_sql(payload)

        try:
            data = await self._service.execute_query(payload, span_id=span_id)
            # Success — save to history (only if memory is enabled)
            if question and self._sql_memory:
                self._sql_memory.store(question, json.dumps(payload))
            return self._service._format_as_markdown_table(data)
        except Exception as e:
            # Error — no history save
            return f"Error: {e}"
```

**Parameters:**

```json
{
  "payload": {
    "query": {
      "dimensions": ["orders.status"],
      "measures": ["orders.order_count"],
      "filters": [{"member": "orders.customer_name", "operator": "equals", "values": ["Alice"]}],
      "limit": 100
    },
    "queryType": "multi",
    "cache": "force"
  },
  "dry_run": false,
  "question": "How many orders per status?"
}
```

> **Note:** The payload structure supports `queryType` and `cache` for future expansion, even though we don't use them yet. `queryType: "multi"` is reserved for compareDateRange or multi-query support. `cache` controls cache behavior.

**Auto-save:** On success + `question` provided → `sql_memory.store(question, payload)`

### CubeSearchTool

```python
class CubeSearchTool(Tool):
    name = "cube_search"
    description = "Search past successful Cube queries by natural language similarity."

    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Natural language question to search for similar past queries.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
            },
        },
        "required": ["question"],
    }

    async def execute(self, question: str, limit: int = 5, **kwargs) -> str:
        pairs = self._sql_memory.search(question, limit=limit)
        # Format as markdown list
```

---

## Phase 5: Runner Integration

**File:** `src/nanobot/runner.py`

### Constructor changes

Add `cube_config` parameter to `AgentRunner.__init__`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanobot.config.schema import CubeConfig

class AgentRunner:
    def __init__(
        self,
        workspace: Path,
        models: list[Any],
        bus: MessageBus,
        providers: ProvidersConfig,  # ← Already in existing signature, keep for agent models
        *,
        # ... existing params ...
        cube_config: CubeConfig | None = None,    # ← CubeConfig (not full Config)
        **kwargs: Any,
    ) -> None:
        # ... existing init ...

        # Cube semantic layer
        self._cube_service: Any = None
        self._cube_memory: Any = None
        self._cube_schema_index: Any = None
        if cube_config and cube_config.enabled:
            from nanobot.cube.service import CubeService

            self._cube_service = CubeService(cube_config)
            # initialize() is async — deferred to first tool use (matches Mem0Client pattern)

            if cube_config.memory.enabled:
                from nanobot.cube.sql_memory import SqlMemory

                # Common ChromaDB persist directory: {workspace}/chroma/
                persist_path = workspace / "chroma"

                self._cube_memory = SqlMemory(
                    persist_dir=persist_path,
                    max_results=cube_config.memory.max_results,
                    embedder=cube_config.memory.embedder,
                    reranker=cube_config.memory.reranker,
                )
                self._cube_memory.initialize()

                # Schema index uses the same ChromaDB directory ({workspace}/chroma/)
                # with its own embedder/reranker (can differ from memory's)
                # Set _schema_index directly before tools use it (health check still deferred to first tool use)
                if cube_config.schema_index.enabled:
                    from nanobot.cube.schema_index import CubeSchemaIndex

                    self._cube_schema_index = CubeSchemaIndex(
                        persist_dir=persist_path,
                        embedder=cube_config.schema_index.embedder,
                        reranker=cube_config.schema_index.reranker,
                    )
                    self._cube_schema_index.initialize()
                    self._cube_service._schema_index = self._cube_schema_index
```

### Lazy async initialization (matches Mem0Client pattern)

`CubeService.__init__` is synchronous, but `initialize()` is async. Following the same pattern as `Mem0Client`, the health check is deferred to the first tool invocation:

```python
# In tool execute() methods:
if self._service._client is None:
    await self._service.initialize()
if not self._service.is_available:
    return f"Error: Cube is not available: {self._service.init_error}"
```

### Tool registration (in `_register_default_tools`)

```python
# Register Cube tools (always register if config enabled — lazy init in tools)
if self._cube_service:
    from nanobot.tools.cube import CubeSchemaTool, CubeQueryTool, CubeSearchTool

    self.tools.register(CubeSchemaTool(self._cube_service))
    self.tools.register(CubeQueryTool(self._cube_service, self._cube_memory))
    if self._cube_memory and self._cube_memory.is_available:
        self.tools.register(CubeSearchTool(self._cube_memory))
```

> **Note:** `CubeSchemaIndex` is not passed to tools — it's set on `CubeService._schema_index` directly during runner initialization. `CubeSchemaTool` accesses it through `service.get_schema_context()`.

### Shutdown cleanup

Add to `shutdown()` method:

```python
if self._cube_service:
    await self._cube_service.close()
    # Note: ChromaDB does not require explicit cleanup — clients are lightweight
    # and persist to disk; no close() call needed for SqlMemory or CubeSchemaIndex.
```

### Bootstrapper changes

Find all `AgentRunner` instantiation sites in the codebase (e.g., `cli/commands.py`, `src/nanobot/gateway.py`, etc.) and pass `cube_config=config.cube` to each:

```python
runner = AgentRunner(
    workspace=config.workspace_path,
    models=...,
    bus=...,
    cube_config=config.cube,      # ← pass CubeConfig (not full Config)
    # ... other params ...
)
```

> **Known call sites to check** (verify during implementation):
> - `cli/commands.py` — likely in CLI command handlers
> - `src/nanobot/gateway.py` — if used by gateway service
> - Any other place `AgentRunner` is constructed

---

## Phase 6: Tests

### `tests/cube/conftest.py`

Fixtures for mock httpx responses:
- `mock_health_ok` — `GET /readyz` → 200 `{"health": "HEALTH"}`
- `mock_health_fail` — `GET /readyz` → connection error
- `mock_health_down` — `GET /readyz` → 500 `{"health": "DOWN"}`
- `mock_live_ok` — `GET /livez` → 200 `{"health": "HEALTH"}`
- `mock_meta_response` — `GET /v1/meta` → sample cube schema JSON
- `mock_load_success` — `POST /v1/load` → sample data response
- `mock_load_error` — `POST /v1/load` → error response
- `mock_load_continue_wait` — `POST /v1/load` → `{"error": "Continue wait"}`

### `tests/cube/test_service.py` — 23 tests

| Test | Description |
|------|-------------|
| `test_initialize_success` | Mock `/readyz` 200 → `is_available=True` |
| `test_initialize_connection_error` | Mock `/readyz` → connection error → `init_error` set |
| `test_initialize_down` | Mock `/readyz` → 500 `{"health": "DOWN"}` → `is_available=False` |
| `test_check_live` | Mock `/livez` → returns True |
| `test_get_schema_parses_meta` | Mock `/v1/meta` → flat text output |
| `test_get_schema_empty_cubes` | Mock `/v1/meta` no cubes → graceful message |
| `test_get_schema_caching` | Second call uses cached schema (no HTTP call) |
| `test_get_schema_context_no_question` | Returns full schema when no question provided |
| `test_get_schema_context_small_schema` | Returns full schema when schema < threshold |
| `test_get_schema_context_large_schema_no_index` | Returns full schema when index not available |
| `test_get_schema_context_large_schema_with_search` | Returns search results when schema ≥ threshold + question + index available |
| `test_get_schema_context_fallback_on_empty_search` | Falls back to full schema when search returns no results |
| `test_reload_stores_schema_hash` | Reload computes and stores SHA-256 hash |
| `test_reload_reindexes` | Reload calls `index_cubes` when schema index is available |
| `test_reload_empty_cubes` | Mock `/v1/meta` no cubes → graceful message returned |
| `test_initialize_concurrent_serialization` | Concurrent `initialize()` calls are properly serialized via asyncio.Lock |
| `test_execute_query_success` | Mock `/v1/load` → returns `list[dict]`, formats as markdown table |
| `test_execute_query_error` | Mock `/v1/load` error → raises exception |
| `test_execute_query_continue_wait` | Mock `/v1/load` → "Continue wait" triggers tenacity retry with fixed delay; raises on timeout |
| `test_execute_query_span_id` | span_id passed through → x-request-id header is `{span_id}-span-{retry_count}` |
| `test_preview_sql_returns_sql` | Mock /v1/sql → SQL string output |
| `test_auth_header_bearer` | `token` set → `Authorization: Bearer {token}` |
| `test_auth_header_empty_token` | `token` empty → no Authorization header |
| `test_health_no_auth_header` | `/readyz` request does NOT include Authorization header |

### `tests/cube/test_sql_memory.py` — 16 tests

| Category | Tests |
|----------|-------|
| Init | directory created, collection exists, version check |
| Store | stores question + sql, handles empty, handles special chars |
| Search | returns results with scores, limit respected, no results message |
| Clear | clears all, handles empty |
| Available | unavailable on init error, graceful degradation |

### `tests/cube/test_schema_index.py` — 12 tests

| Category | Tests |
|----------|-------|
| Init | directory created, `cube_schema_items` collection exists |
| Index | indexes cubes from meta response, re-index clears old data, handles empty cubes |
| Search | returns relevant cubes with scores, limit respected, no results returns empty |
| Clear | clears all items, handles empty |
| Hybrid | `get_schema_context` returns full schema when < threshold, uses search when ≥ threshold, falls back on empty search results |

### `tests/tools/test_cube_tool.py` — 18 tests

| Category | Tests |
|----------|-------|
| CubeSchemaTool | returns schema, calls `service.get_schema()` |
| CubeQueryTool | query mode → markdown, dry_run → SQL, error → raises exception string, auto-save on success, no save without question, no save on error |
| CubeSearchTool | returns formatted results, no results message, unavailable memory |

---

## Phase 7: E2E Tests

### `tests/e2e/cube/test_cube_e2e.py`

End-to-end tests against a live Cube + DuckDB stack using the fixtures at `tests/e2e/cube/fixtures/`.

**Fixtures location:** `tests/e2e/cube/fixtures/`
- `compose.yaml` — Docker Compose with Cube.js on port 4000, DuckDB, dev mode, secret "secret"
- `conf/model/customers.yaml` — Cube model with customers table (dimensions: customer_id, first_name, last_name, first_order, most_recent_order, number_of_orders, customer_lifetime_value; measure: count)
- `conf/db/jaffle_shop.duckdb` — DuckDB database

### pytest-docker fixture

Use `pytest-docker` for declarative container management:

```python
@pytest.fixture(scope="module")
def cube_service_url(docker_ip):
    """Return the Cube service URL from docker network."""
    return f"http://{docker_ip}:4000"

@pytest.fixture(scope="module")
async def cube_container(docker_services):
    """Start and wait for Cube container."""
    docker_services.start("cube")
    docker_services.wait_for_endpoint("cube", timeout=30)
    yield
    docker_services.stop("cube")
```


### Test Cases

```python
import pytest
import httpx
from nanobot.cube.service import CubeService
from nanobot.config.schema import CubeConfig

# Fixtures
@pytest.fixture
def cube_service():
    config = CubeConfig(
        cube_url="http://localhost:4000",
        token="secret",
        cubejs_api_path="/cubejs-api",
    )
    return CubeService(config)

@pytest.fixture
async def cube_service_init(cube_service):
    await cube_service.initialize()
    yield cube_service
    await cube_service.close()

# Tests
class TestCubeE2E:
    
    async def test_health_check(self, cube_service):
        """Test /readyz endpoint."""
        assert await cube_service.check_ready() is True
    
    async def test_get_schema(self, cube_service_init):
        """Test fetching schema via /v1/meta."""
        schema = await cube_service_init.get_schema()
        assert "customers" in schema
    
    async def test_execute_query(self, cube_service_init):
        """Test executing a query via /v1/load."""
        payload = {
            "query": {
                "measures": ["customers.count"],
            }
        }
        data = await cube_service_init.execute_query(payload)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "customers.count" in data[0]
    
    async def test_preview_sql(self, cube_service_init):
        """Test previewing SQL via /v1/sql."""
        payload = {
            "query": {
                "measures": ["customers.count"],
            }
        }
        result = await cube_service_init.preview_sql(payload)
        assert "SELECT" in result
```

| Test | Description |
|------|-------------|
| `test_health_check` | `GET /readyz` returns True when Cube is live |
| `test_get_schema` | `GET /v1/meta` returns schema containing "customers" cube |
| `test_execute_query` | `POST /v1/load` with `customers.count` measure returns data list with the measure key |
| `test_preview_sql` | `POST /v1/sql` returns SQL string containing "SELECT" |

---

## File Map

```
src/nanobot/
├── config/
│   └── schema.py              [modify] add CubeConfig + add cube field to Config
├── cube/                      [new]
│   ├── __init__.py            [new] exports CubeService, SqlMemory, CubeSchemaIndex
│   ├── service.py             [new] CubeService (plain class) — hybrid schema retrieval
│   ├── schema_index.py        [new] CubeSchemaIndex — ChromaDB schema element indexing
│   └── sql_memory.py          [new] SqlMemory — NL→SQL query history
└── tools/
    ├── __init__.py           [modify] export CubeSchemaTool, CubeQueryTool, CubeSearchTool
    └── cube.py                [new] 3 tools (cube_schema accepts optional question)

tests/
├── cube/                      [new]
│   ├── conftest.py
│   ├── test_service.py
│   ├── test_schema_index.py   [new] CubeSchemaIndex tests
│   └── test_sql_memory.py
├── tools/
│   └── test_cube_tool.py      [new]
└── e2e/
    └── cube/
        └── test_cube_e2e.py   [new] E2E tests against live Cube
```

---

## Dependencies

- `httpx>=0.28.0` — already a core dependency (see `pyproject.toml`)
- `openai>=2.3.1,<3.0.0` — for OpenAI embeddings and reranking via official SDK (NEW); add to `pyproject.toml`
- `tenacity>=9.1.4,<10.0.0` — for continue-wait retry logic (uses `AsyncRetrying` for async context); update `pyproject.toml` with this version (NEW)
- `toons>=0.5.4,<1.0.0` — for lossless compact schema output (NEW); from PyPI — **verify this is on PyPI before use**
- `pytest-docker` — for E2E test container management (NEW); add to test dependency group in `pyproject.toml`
- ChromaDB — already present (uses openai internally for OpenAIEmbeddingFunction)

---

## API Endpoint Summary

| Tool | Method | Endpoint | Auth | Notes |
|------|--------|----------|------|-------|
| Health (init) | GET | `{cube_url}/readyz` | **None** | No base path, no auth |
| Health (liveness) | GET | `{cube_url}/livez` | **None** | No base path, no auth |
| `cube_schema` | GET | `{cube_url}{cubejs_api_path}/v1/meta` | `Authorization: Bearer {token}` | Cached |
| `cube_query` (query) | POST | `{cube_url}{cubejs_api_path}/v1/load` | `Authorization: Bearer {token}` | Body: `{"query": {...}, "queryType"?: string, "cache"?: string}`; supports `x-request-id` header for tracing continue-wait cycles |
| `cube_query` (dry_run) | POST | `{cube_url}{cubejs_api_path}/v1/sql` | `Authorization: Bearer {token}` | Returns generated SQL |

**Key details from Cube REST API docs:**
- `/readyz` and `/livez` are NOT prefixed with the base path — they are root-level endpoints
- Health endpoints have no auth scope — accessible without token
- Auth uses Bearer token: `Authorization: Bearer {token}`
- POST `/v1/load` body: `{"query": {...}}` with optional `queryType` and `cache` parameters
- `query`: required, the Cube query object
- `queryType`: optional, for multi queries (e.g., `"multi"` for compareDateRange)
- `cache`: optional, cache control (e.g., `"force"` to bypass cache)
- Long-running queries return `{"error": "Continue wait"}` with HTTP 200
- All numerical values in responses are strings (driver-dependent)

> **Note on payload expansion:** The payload structure supports `queryType` and `cache` for future expansion. Currently only `query` is used. `queryType: "multi"` will be needed for `compareDateRange` or multi-query support when those features are required.

---

## Future Enhancement: Schema Staleness Detection

### When Cube has a schema version endpoint:
1. On each `cube_schema` call, check the schema version endpoint
2. If version changed → re-fetch `/v1/meta`, update `_schema` and `_schema_hash`
3. Also update the ChromaDB vector embeddings (delete old, insert new)
4. Serve from cache thereafter

### When Cube does NOT have a schema version endpoint (current situation):
1. No automatic staleness detection available
2. `reload()` must be called manually to refresh schema
3. ChromaDB embeddings must be manually rebuilt via `reload()`
4. For now: every `get_schema_context()` call fetches fresh meta if schema is large enough to need embedding search
5. Consider: polling `/v1/meta` periodically if Cube doesn't change often

### What we need from Cube:
- A lightweight endpoint (e.g., `GET /v1/meta/version` or similar) that returns just a version/hash
- This avoids downloading the full schema just to check if it changed
- Alternative: Cube could add `ETag` or `Last-Modified` headers to `/v1/meta`

### What's already in place:
- `CubeService._schema_hash` stores SHA-256 of the raw `/v1/meta` response
- `reload()` already computes this hash on every fetch

---

## Implementation Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
   ↓          ↓         ↓         ↓         ↓         ↓         ↓
  Config    Service   Memory    Tools    Runner    Tests    E2E Tests
```

Phases 3a and 3b (SqlMemory and CubeSchemaIndex) are independent and can run in parallel.

### Implementation checklist

- [ ] **Phase 1:** Add `CubeConfig` to `schema.py` (with `token` for bearer auth, `request_span_enabled`, `continue_wait_retry_interval` (seconds), `continue_wait_retry_max_attempts`, nested `memory` with `CubeMemoryConfig` containing `enabled`, `max_results`, `embedder`, `reranker`, nested `schema_index` with `CubeSchemaIndexConfig` containing `enabled`, `threshold`, `max_results`, `embedder`, `reranker`), add `cube` field to `Config` class. Add `EmbedderConfig` and `RerankerConfig` classes following nanobot provider config format.
- [ ] **Phase 2:** Implement `CubeService` as plain class with `get_schema_context()` hybrid method, `AsyncRetrying` from tenacity for continue-wait retry with fixed delay (`continue_wait_retry_interval`) and `continue_wait_retry_max_attempts` exit, `asyncio.Lock` for concurrent init protection, Bearer token auth. No mutable request state — span_id generated at tool layer and passed to `execute_query(query, span_id=...)` only (not preview_sql). Uses plain dicts throughout — no Pydantic models.
- [ ] **Phase 3a:** Implement `SqlMemory` with ChromaDB backend (query history), persist to `{workspace}/chroma/` (shared with mem0, uses `chroma.db` SQLite file), use ChromaDB's built-in `OpenAIEmbeddingFunction` with resolved api_key, async methods with `asyncio.to_thread()`. Guard `store()` call with `self._sql_memory` falsy check. Reranking via `_rerank()` helper (post-retrieval step).
- [ ] **Phase 3b:** Implement `CubeSchemaIndex` with ChromaDB backend (schema indexing), same persist directory, use ChromaDB's built-in `OpenAIEmbeddingFunction` with resolved api_key, async methods with `asyncio.to_thread()`, supports reranking via `_rerank()` helper.
- [ ] **Phase 4:** Implement 3 tools with PydanticAI `parameters` JSON schema (`cube_schema` with optional `question`, `cube_query` generates span_id at tool level and passes to service only (not preview_sql), `cube_search`). Ensure consistent `Error:` prefix for all error returns.
- [ ] **Phase 5:** Wire into `AgentRunner.__init__` (service + memory + schema index with workspace-derived path using `cube_config.memory.enabled`, `cube_config.memory.max_results`, `cube_config.memory.embedder`, `cube_config.memory.reranker`), set `service._schema_index` directly in runner (before tools use service), health check deferred to first tool use, `_register_default_tools`, `shutdown`; update all `AgentRunner` instantiation sites in bootstrapper
- [ ] **Phase 6:** Write all tests (including `test_schema_index.py`, `AsyncRetrying` retry tests, bearer auth tests, span_id pass-through tests, memory=None crash test)
- [ ] **Phase 7:** Write E2E tests against live Cube + DuckDB stack using fixtures at `tests/e2e/cube/fixtures/`
