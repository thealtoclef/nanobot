# Cube Semantic Layer Implementation Plan

## Overview

Implement a Cube-based text-to-SQL feature for nanobot. LLM sees Cube schema via `cube_meta`, executes queries via `cube_load`.

```
User: "How many orders by status?"
    │
    │ cube_meta()
    │   → GET /cubejs-api/v1/meta
    │   → Returns flat schema text
    │
    │ LLM builds JSON query
    │
    │ cube_load(query_json, dry_run=False)
    │   → POST /cubejs-api/v1/load
    │   → Returns markdown table
    │
    │ (On success + question)
    │ → Memory stores (nl_question, query_json)
```

---

## Tool Inventory

| Tool | Name | API Endpoint | Purpose |
|------|------|-------------|---------|
| Schema | `cube_meta` | `GET /cubejs-api/v1/meta` | Feed LLM full schema |
| Query | `cube_load` | `POST /cubejs-api/v1/load` | Execute query |
| Search | `cube_search` | — | Search past queries from ChromaDB |

---

## Phase 1: Config

**File:** `src/nanobot/config/schema.py`

```python
class CubeConfig(BaseModel):
    """Cube semantic layer configuration."""
    enabled: bool = False
    cube_url: str = ""                      # e.g. "https://cube.example.com"
    api_key: str = ""                      # Bearer token
    cubejs_api_path: str = "/cubejs-api"   # default Cube API path
    memory_enabled: bool = True
    memory_max_results: int = 5
```

---

## Phase 2: CubeService

**File:** `src/nanobot/cube/service.py`

HTTP client wrapping Cube REST API. Lazy import `httpx`.

### Class: `CubeService`

```python
class CubeService(BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}

    cube_url: str = ""
    api_key: str = ""
    cubejs_api_path: str = "/cubejs-api"
    memory_enabled: bool = True
    memory_max_results: int = 5

    _available: bool = False
    _init_error: str | None = None
    _schema: str | None = None

    def model_post_init(self, _context: Any) -> None:
        if self.enabled:
            self.initialize()
```

| Method | API | Purpose |
|--------|-----|---------|
| `initialize()` | `GET /readyz` | Verify Cube reachable, set `_available` |
| `get_schema()` | `GET /cubejs-api/v1/meta` | Fetch schema → flat text |
| `execute_query()` | `POST /cubejs-api/v1/load` | Run query → markdown table |
| `dry_plan()` | `POST /cubejs-api/v1/sql` | Show generated SQL |
| `reload()` | — | Re-fetch schema |

### Schema output format (flat per-table)

```
## orders
Dimensions: order_id(number, pk), status(string), customer_name(string), created_at(time)
Measures: order_count(count), total_revenue(sum)
## customers
Dimensions: customer_id(number, pk), name(string), city(string)
Measures: customer_count(count)
```

### Query result output format

`POST /v1/load` response → formatted markdown:

```
| status     | order_count |
|------------|-------------|
| completed  | 150         |
| pending    | 42          |
```

### Error handling

- All exceptions caught → `_init_error` set, startup never crashes
- Query errors return formatted error string (don't raise)

---

## Phase 3: SqlMemory

**File:** `src/nanobot/cube/sql_memory.py`

ChromaDB-backed store for `(nl_question, query_json)` pairs.

| Method | Description |
|--------|-------------|
| `initialize()` | Setup ChromaDB, create collection |
| `store(question, query_json)` | Save pair with embedding |
| `search(question, limit)` | Semantic search → list of pairs with scores |
| `clear()` | Clear all stored pairs |

- Collection: `cube_pairs`
- Embeddings: `all-MiniLM-L6-v2` via ChromaDB default
- Persist to workspace `sql_memory_chroma/`

---

## Phase 4: Tools

**File:** `src/nanobot/tools/cube.py`

### CubeMetaTool

```python
class CubeMetaTool(Tool):
    name = "cube_meta"
    description = "Get the Cube schema: cubes, dimensions, measures, and descriptions."
    api_endpoint = "GET /cubejs-api/v1/meta"

    async def execute(self, **kwargs) -> str:
        return self._service.get_schema()
```

### CubeLoadTool

```python
class CubeLoadTool(Tool):
    name = "cube_load"
    description = "Execute a query against Cube via JSON query format. Set dry_run=True to preview SQL without executing."
    api_endpoint = "POST /cubejs-api/v1/load (or /v1/sql if dry_run=True)"

    async def execute(
        self,
        query_json: dict,
        dry_run: bool = False,
        question: str | None = None,
        **kwargs
    ) -> str:
        if dry_run:
            return await self._service.dry_plan(query_json)
        result = await self._service.execute_query(query_json)
        if question and self._sql_memory:
            self._sql_memory.store(question, json.dumps(query_json))
        return result
```

**Parameters:**

```json
{
  "query_json": {
    "dimensions": ["orders.status"],
    "measures": ["orders.order_count"],
    "filters": [{"member": "orders.customer_name", "operator": "equals", "values": ["Alice"]}],
    "limit": 100
  },
  "dry_run": false,
  "question": "How many orders per status?"
}
```

**Auto-save:** On success + `question` provided → `sql_memory.store(question, query_json)`

### CubeSearchTool

```python
class CubeSearchTool(Tool):
    name = "cube_search"
    description = "Search past successful Cube queries by natural language similarity."

    async def execute(self, question: str, limit: int = 5, **kwargs) -> str:
        pairs = self._sql_memory.search(question, limit=limit)
        # Format as markdown list
```

---

## Phase 5: Runner Integration

**File:** `src/nanobot/runner.py`

```python
# Init
self._cube_service: Any = None
self._cube_memory: Any = None
if cube_config and cube_config.enabled:
    self._cube_service = CubeService.model_validate(cube_config.model_dump())
    if cube_config.memory_enabled:
        from nanobot.cube.sql_memory import SqlMemory
        self._cube_memory = SqlMemory(
            persist_dir=workspace / "sql_memory_chroma",
            max_results=cube_config.memory_max_results,
        )
        self._cube_memory.initialize()

# Register tools
if self._cube_service and self._cube_service._available:
    tools.register(CubeMetaTool(self._cube_service))
    tools.register(CubeLoadTool(self._cube_service, self._cube_memory))
    if self._cube_memory:
        tools.register(CubeSearchTool(self._cube_memory))
```

---

## Phase 6: Tests

### `tests/cube/conftest.py`

Fixtures for mock httpx responses:
- `mock_health_ok` — `/readyz` → 200
- `mock_health_fail` — `/readyz` → connection error
- `mock_meta_response` — `/v1/meta` → sample cube schema JSON
- `mock_load_success` — `/v1/load` → sample data response
- `mock_load_error` — `/v1/load` → error response
- `mock_sql_response` — `/v1/sql` → generated SQL

### `tests/cube/test_service.py` — 9 tests

| Test | Description |
|------|-------------|
| `test_initialize_success` | Mock `/readyz` 200 → `_available=True` |
| `test_initialize_connection_error` | Mock `/readyz` → connection error → `_init_error` set |
| `test_initialize_unauthorized` | Mock `/readyz` → 401 → `_init_error` set |
| `test_get_schema_parses_meta` | Mock `/v1/meta` → flat text output |
| `test_get_schema_empty_cubes` | Mock `/v1/meta` no cubes → graceful message |
| `test_execute_query_success` | Mock `/v1/load` → markdown table |
| `test_execute_query_error` | Mock `/v1/load` error → formatted error string |
| `test_dry_plan_returns_sql` | Mock `/v1/sql` → SQL string |
| `test_reload_refreshes_schema` | Re-fetch `/v1/meta` |

### `tests/cube/test_sql_memory.py` — 16 tests

| Category | Tests |
|----------|-------|
| Init | directory created, collection exists, version check |
| Store | stores question + sql, handles empty, handles special chars |
| Search | returns results with scores, limit respected, no results message |
| Clear | clears all, handles empty |
| Available | unavailable on init error, graceful degradation |

### `tests/tools/test_cube_tool.py` — 18 tests

| Category | Tests |
|----------|-------|
| CubeMetaTool | returns schema, calls `service.get_schema()` |
| CubeLoadTool | query mode → markdown, dry_run → SQL, error → string, auto-save on success, no save without question, no save on error |
| CubeSearchTool | returns formatted results, no results message, unavailable memory |

---

## File Map

```
src/nanobot/
├── config/
│   └── schema.py              [modify] add CubeConfig
├── cube/                      [new]
│   ├── __init__.py
│   ├── service.py             [new] CubeService
│   └── sql_memory.py         [new] SqlMemory
└── tools/
    └── cube.py                [new] 3 tools

tests/
├── cube/                      [new]
│   ├── conftest.py
│   ├── test_service.py
│   └── test_sql_memory.py
└── tools/
    └── test_cube_tool.py     [new]
```

---

## Dependencies

```toml
# pyproject.toml
[project.optional-dependencies]
cube = ["httpx>=0.27.0"]
```

No new dependencies. `httpx` via existing deps. ChromaDB already present.

---

## API Endpoint Summary

| Tool | Method | Endpoint | Auth |
|------|--------|----------|------|
| `cube_meta` | GET | `{cube_url}{cubejs_api_path}/v1/meta` | Bearer `api_key` |
| `cube_load` (query) | POST | `{cube_url}{cubejs_api_path}/v1/load` | Bearer `api_key` |
| `cube_load` (dry_run) | POST | `{cube_url}{cubejs_api_path}/v1/sql` | Bearer `api_key` |
| Health check | GET | `{cube_url}/readyz` | Bearer `api_key` |

---

## Implementation Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
   ↓          ↓          ↓          ↓         ↓         ↓
 Config    Service    Memory     Tools    Runner    Tests
```

Phases 3, 4 can start after Phase 2 is complete.
