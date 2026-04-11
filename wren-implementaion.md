# Wren Engine Adoption Plan for Nanobot

## Context

Nanobot needs a Text-to-SQL / "SQL Talk" capability. Wren Engine was selected as the foundation — it's a complementary library (not a competing framework) that provides a semantic SQL layer (MDL), SQL transformation, and 20+ datasource connectors.

**Codebase state:** Based on commit `f48d33b` which migrated mem0 to PydanticAI-native capability:
- Memory is now a PydanticAI native tool (`search_memory`), not system prompt injection
- `AgentDeps` dataclass provides per-run context (`session_key`, `channel`, `chat_id`, `mem0_client`)
- `ContextBuilder.build_system_prompt()` is dead code — instructions flow through `build_instructions()` → `Agent(instructions=...)`
- Only ONE `TalkerAgent` instantiation in runner.py (line 131)

**Constraints:**
- Use Python API directly (not CLI)
- dbt sync via Go binary (supported tool, call binary directly)
- MDL Schema Maintenance is Phase 3 (not needed for PoC)
- Memory bridge uses existing mem0 + ChromaDB (skip LanceDB)

---

## Phase 1: PoC (Go/No-Go)

### 1.1 Add wren-engine as optional dependency

**File:** `pyproject.toml`

Add to `[project.optional-dependencies]`, following the discord/langfuse pattern:

```toml
[project.optional-dependencies]
api = ["aiohttp>=3.9.0,<4.0.0"]
discord = ["discord.py>=2.5.2,<3.0.0"]
langfuse = ["langfuse>=3.0.0,<4.0.0"]
wren = ["wren-engine>=0.2.1,<0.3.0"]
wren-postgres = ["nanobot[wren]", "wren-engine[postgres]>=0.2.1,<0.3.0"]
wren-duckdb = ["nanobot[wren]", "wren-engine[duckdb]>=0.2.1,<0.3.0"]
wren-bigquery = ["nanobot[wren]", "wren-engine[bigquery]>=0.2.1,<0.3.0"]
wren-snowflake = ["nanobot[wren]", "wren-engine[snowflake]>=0.2.1,<0.3.0"]

all = ["nanobot[api,discord,langfuse,wren]"]
```

Not in core dependencies — wren-engine pulls in pyarrow, sqlglot, ibis-framework, and a Rust wheel.

### 1.2 Add WrenConfig to config schema

**File:** `src/nanobot/config/schema.py`

New config models following existing patterns (`WebSearchConfig`, `ExecToolConfig`):

```python
class WrenConnectionConfig(Base):
    """Connection parameters for a Wren data source."""
    host: str = ""
    port: int = 0
    database: str = ""
    user: str = ""
    password: str = ""
    password_env: str = ""
    extras: dict[str, Any] = Field(default_factory=dict)

class WrenDataSourceConfig(Base):
    """A Wren data source definition."""
    type: str = ""  # "postgres", "duckdb", "bigquery", etc.
    connection: WrenConnectionConfig = Field(default_factory=WrenConnectionConfig)

class WrenConfig(Base):
    """Wren Engine Text-to-SQL configuration."""
    enabled: bool = False
    project_path: str = ""       # Path to MDL project dir
    manifest_path: str = ""      # Alternative: pre-built mdl.json
    data_source: WrenDataSourceConfig = Field(default_factory=WrenDataSourceConfig)
    strict_mode: bool = False
    denied_functions: list[str] = Field(default_factory=list)
```

Add to `ToolsConfig` (line ~192):
```python
class ToolsConfig(Base):
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    wren: WrenConfig = Field(default_factory=WrenConfig)  # NEW
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
```

### 1.3 Create WrenEngineService

**New file:** `src/nanobot/wren/__init__.py` (empty)
**New file:** `src/nanobot/wren/service.py`

Backing service that holds engine instance, manifest, and schema description. Follows CronService pattern — stateful service that tools call into.

All `from wren import ...` must be lazy (inside methods), never at module top-level. A top-level import would crash at import time before the runner's `try/except ImportError` can catch it.

```python
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nanobot.config.schema import WrenConfig as NanobotWrenConfig


class WrenEngineService:
    """Managed wrapper around wren-engine WrenEngine.

    All wren-engine imports are lazy (inside methods) because wren-engine
    is an optional dependency. The module must be importable even when
    wren-engine is not installed.
    """

    def __init__(self, config: NanobotWrenConfig):
        self._config = config
        self._engine: Any = None  # WrenEngine (lazy import)
        self._manifest_str: str | None = None
        self._manifest_dict: dict | None = None
        self._schema_description: str | None = None
        self._initialized: bool = False
        self._init_error: str | None = None

    def initialize(self) -> None:
        """Load manifest and create WrenEngine.

        All wren imports happen here (lazy). Catches all errors — bad
        creds/manifest won't crash nanobot startup.
        """
        # Early validation — catch misconfiguration before importing wren
        if not self._config.data_source.type:
            self._init_error = "data_source.type not configured"
            logger.warning("WrenEngineService: {}", self._init_error)
            return
        if not self._config.project_path and not self._config.manifest_path:
            self._init_error = "Either project_path or manifest_path must be set"
            logger.warning("WrenEngineService: {}", self._init_error)
            return

        try:
            from wren import WrenEngine
            from wren.config import WrenConfig as WrenEngineConfig
            from wren.mdl import to_json_base64

            manifest_dict = self._load_manifest()
            self._manifest_dict = manifest_dict
            self._manifest_str = to_json_base64(manifest_dict)
            connection_info = self._build_connection_info()
            self._engine = WrenEngine(
                manifest_str=self._manifest_str,
                data_source=self._config.data_source.type,
                connection_info=connection_info,
                config=WrenEngineConfig(
                    strict_mode=self._config.strict_mode,
                    denied_functions=frozenset(self._config.denied_functions),
                ),
            )
            self._initialized = True
            logger.info("WrenEngineService initialized successfully")
        except ImportError:
            self._init_error = (
                "wren-engine not installed. Install with: pip install nanobot[wren]"
            )
            logger.warning(self._init_error)
        except Exception as e:
            self._init_error = str(e)
            logger.warning("WrenEngineService init failed (non-fatal): {}", e)

    def _load_manifest(self) -> dict:
        """Load MDL manifest. Returns camelCase dict."""
        cfg = self._config
        if cfg.project_path:
            from wren.context import build_json
            return build_json(Path(cfg.project_path).expanduser())
        else:
            path = Path(cfg.manifest_path).expanduser()
            return json.loads(path.read_text(encoding="utf-8"))

    def _build_connection_info(self) -> dict:
        """Build connection dict from config (resolve password/password_env)."""
        cfg = self._config.data_source.connection
        info: dict[str, Any] = {}
        if cfg.host:
            info["host"] = cfg.host
        if cfg.port:
            info["port"] = cfg.port
        if cfg.database:
            info["database"] = cfg.database
        if cfg.user:
            info["user"] = cfg.user
        password = cfg.password or (
            os.getenv(cfg.password_env, "") if cfg.password_env else ""
        )
        if password:
            info["password"] = password
        info.update(cfg.extras)
        return info

    @property
    def available(self) -> bool:
        return self._initialized and self._engine is not None

    # -- Sync internals (called via run_in_executor) --------------------------

    def _query_sync(self, sql: str, limit: int) -> str:
        """Synchronous query — called via run_in_executor."""
        table = self._engine.query(sql, limit=limit)
        df = table.to_pandas()
        if df.empty:
            return "Query returned no results."
        return df.to_markdown(index=False)

    def _dry_plan_sync(self, sql: str) -> str:
        """Synchronous dry_plan — called via run_in_executor."""
        return self._engine.dry_plan(sql)

    def _dry_run_sync(self, sql: str) -> str:
        """Synchronous dry_run — called via run_in_executor."""
        self._engine.dry_run(sql)
        return "SQL validation passed."

    # -- Async public API (event-loop safe) -----------------------------------

    async def query(self, sql: str, limit: int = 100) -> str:
        """Execute SQL, return formatted markdown table.

        Runs in executor to avoid blocking the event loop — WrenEngine
        calls into Rust/PyO3 and performs database I/O synchronously.
        """
        if not self.available:
            return f"Wren Engine not available: {self._init_error or 'not initialized'}"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._query_sync, sql, limit)

    async def dry_plan(self, sql: str) -> str:
        """Return expanded SQL in target dialect. Runs in executor."""
        if not self.available:
            return f"Wren Engine not available: {self._init_error or 'not initialized'}"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._dry_plan_sync, sql)

    async def dry_run(self, sql: str) -> str:
        """Validate SQL without execution. Runs in executor."""
        if not self.available:
            return f"Wren Engine not available: {self._init_error or 'not initialized'}"
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._dry_run_sync, sql)

    def get_schema_description(self) -> str:
        """Cached full-text schema description for prompt injection.

        CPU-only (no I/O), so synchronous is fine.
        """
        if self._schema_description is None and self._manifest_dict:
            try:
                from wren.memory.schema_indexer import describe_schema
                self._schema_description = describe_schema(self._manifest_dict)
            except ImportError:
                return ""
        return self._schema_description or ""

    def get_manifest_dict(self) -> dict | None:
        return self._manifest_dict

    def reload(self) -> None:
        """Re-load manifest and recreate engine."""
        self._schema_description = None
        self._initialized = False
        self._init_error = None
        self.initialize()
```

### 1.4 Create SqlQueryTool

**New file:** `src/nanobot/tools/wren.py`

Implements nanobot `Tool` ABC:

```python
class SqlQueryTool(Tool):
    """Execute SQL queries through the Wren semantic layer.

    Note: While Wren Engine generates SELECT-only queries through
    its semantic model, this tool uses the default read_only=False
    since the underlying database connection could theoretically
    allow writes depending on configuration.
    """

    def __init__(self, engine_service: WrenEngineService):
        self._engine = engine_service

    @property
    def name(self) -> str:
        return "sql_query"

    @property
    def description(self) -> str:
        return (
            "Execute SQL queries against the data warehouse via the Wren semantic layer. "
            "Three modes: 'query' executes and returns results, "
            "'dry_plan' shows the expanded SQL without executing, "
            "'dry_run' validates SQL syntax. Use dry_plan first for complex queries."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The SQL query to execute"},
                "mode": {
                    "type": "string",
                    "enum": ["query", "dry_plan", "dry_run"],
                    "description": "Execution mode (default: query)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 100, max 1000)",
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "required": ["sql"],
        }

    async def execute(self, sql: str, mode: str = "query", limit: int = 100, **kwargs: Any) -> str:
        try:
            if mode == "dry_plan":
                return await self._engine.dry_plan(sql)
            elif mode == "dry_run":
                return await self._engine.dry_run(sql)
            else:
                # Conservative for PoC: auto dry_plan before every query to catch
                # errors before hitting the DB. Doubles latency (2 executor round-trips).
                # Phase 2 will make this configurable (skip_dry_plan option).
                try:
                    await self._engine.dry_plan(sql)
                except Exception as plan_err:
                    return (
                        f"SQL planning failed: {plan_err}\n\n"
                        "Fix the SQL and retry. Check model/column names against the schema."
                    )
                return await self._engine.query(sql, limit=min(limit, 1000))
        except Exception as e:
            phase = getattr(e, "phase", None)
            phase_str = f" (phase: {phase.name})" if phase else ""
            return f"Error{phase_str}: {e}"
```

### 1.5 Wire into AgentRunner

**File:** `src/nanobot/runner.py`

**1. Add to TYPE_CHECKING imports (line ~39):**
```python
if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        ExecToolConfig,
        MCPServerConfig,
        MemoryConfig,
        WrenConfig,          # NEW
        WebToolsConfig,
    )
```

**2. Add `wren_config` parameter to `__init__()` (after `memory_config`, line ~91):**
```python
    memory_config: MemoryConfig | None = None,
    wren_config: WrenConfig | None = None,   # NEW
    **kwargs: Any,
```

**3. Initialize WrenEngineService before `_register_default_tools()` (~line 128, after mem0 init):**
```python
    # Wren Engine (lazy, non-fatal on failure)
    self._wren_engine: Any = None
    if wren_config and wren_config.enabled:
        try:
            from nanobot.wren.service import WrenEngineService
            self._wren_engine = WrenEngineService(wren_config)
            self._wren_engine.initialize()
        except ImportError:
            logger.warning("wren-engine not installed. Install with: pip install nanobot[wren]")
        except Exception as e:
            logger.warning("Wren Engine init failed (non-fatal): {}", e)
```

**4. In `_register_default_tools()`, add after cron tool block (~line 226):**
```python
    if self._wren_engine and self._wren_engine.available:
        from nanobot.tools.wren import SqlQueryTool
        self.tools.register(SqlQueryTool(self._wren_engine))
```

### 1.6 Wire into CLI call sites

**File:** `src/nanobot/cli/commands.py`

Add `wren_config=config.tools.wren` (or `runtime_config.tools.wren`) to all three `AgentRunner(...)` calls at lines ~568, ~656, ~876. Same pattern as existing `memory_config=config.memory`.

### 1.7 Inject schema into system prompt

Schema description is semi-static (changes only when MDL changes). The correct injection point is `build_messages()` via `SystemPromptPart` prepend — matching the existing pattern for history summary at `context.py:110-116`.

**File:** `src/nanobot/context.py`

Add `wren_schema` parameter to `build_messages()`:

```python
def build_messages(
    self,
    history: list[ModelMessage],
    current_message: str,
    skill_names: list[str] | None = None,
    media: list[str] | None = None,
    channel: str | None = None,
    chat_id: str | None = None,
    session_key: str | None = None,
    wren_schema: str | None = None,        # NEW
) -> tuple[list[ModelMessage], str | list[dict[str, Any]]]:
    runtime_ctx = self._build_runtime_context(channel, chat_id, self.timezone)
    user_content = self._build_user_content(current_message, media)
    # ... existing merge logic ...

    enriched_history = list(history)

    # Inject schema context as SystemPromptPart (same pattern as history summary)
    if wren_schema:
        enriched_history = [
            ModelRequest(parts=[SystemPromptPart(content=wren_schema)]),
            *enriched_history,
        ]

    if session_key:
        summary = self._db.get_latest_history_summary(session_key)
        if summary:
            enriched_history = [
                ModelRequest(parts=[SystemPromptPart(content=summary)]),
                *enriched_history,
            ]

    return enriched_history, merged
```

**File:** `src/nanobot/runner.py`

Extract a helper to build wren schema context, then pass it at **both** `build_messages()` call sites within `_process_message()`:

```python
def _get_wren_schema(self) -> str | None:
    """Build wren schema context string, or None if unavailable."""
    if not self._wren_engine or not self._wren_engine.available:
        return None
    schema_desc = self._wren_engine.get_schema_description()
    if not schema_desc:
        return None
    return f"# Data Warehouse Schema\n\n{schema_desc}"
```

**System message path (~line 439):**
```python
    model_history, prompt_content = self.context.build_messages(
        history=unconsolidated,
        current_message=msg.content,
        channel=channel,
        chat_id=chat_id,
        session_key=key,
        wren_schema=self._get_wren_schema(),     # NEW
    )
```

**Normal message path (~line 487):**
```python
    model_history, prompt_content = self.context.build_messages(
        history=unconsolidated,
        current_message=msg.content,
        media=msg.media if msg.media else None,
        channel=msg.channel,
        chat_id=msg.chat_id,
        session_key=session.key,
        wren_schema=self._get_wren_schema(),     # NEW
    )
```

Both call sites are wired — no TODOs left.

### 1.8 Create sql-talk skill

**New file:** `src/nanobot/skills/sql-talk/SKILL.md`

```markdown
---
name: sql-talk
description: Answer data questions using natural language to SQL translation.
always: true
metadata: {"nanobot":{"emoji":"🗄️"}}
---

# SQL Talk

You can query the connected data warehouse using the `sql_query` tool.
The schema is provided in your context as "Data Warehouse Schema".

## Workflow

1. **Understand** the user's data question
2. **Plan** — call `sql_query(sql="...", mode="dry_plan")` to validate first
3. **Execute** — call `sql_query(sql="...", mode="query")` to get results
4. **Present** — show results clearly with context and interpretation

## SQL Rules

- Reference model and column names exactly as shown in the schema
- Wren Engine translates semantic model references to actual database SQL
- Start with simple queries, add complexity incrementally
- Default LIMIT 100 unless the user asks for more
- For aggregations, include GROUP BY for all non-aggregated columns

## Error Recovery

If a query fails:
1. Check the error for correct model/column names against schema
2. Use dry_plan to validate the corrected query
3. Only execute after dry_plan succeeds
```

### 1.9 Phase 1 file summary

| Action | File |
|--------|------|
| Modify | `pyproject.toml` — add `wren` optional extra |
| Modify | `src/nanobot/config/schema.py` — add WrenConfig + add to ToolsConfig |
| Create | `src/nanobot/wren/__init__.py` — empty package |
| Create | `src/nanobot/wren/service.py` — WrenEngineService |
| Create | `src/nanobot/tools/wren.py` — SqlQueryTool |
| Modify | `src/nanobot/runner.py` — add wren_config param, init service, register tool, pass schema at both call sites |
| Modify | `src/nanobot/cli/commands.py` — pass wren_config at 3 call sites |
| Modify | `src/nanobot/context.py` — add wren_schema param to build_messages() |
| Create | `src/nanobot/skills/sql-talk/SKILL.md` — always-on skill |
| Create | `tests/tools/test_wren_tool.py` — SqlQueryTool tests |
| Create | `tests/wren/test_service.py` — WrenEngineService tests |

### 1.10 PoC verification

1. `pip install -e ".[wren-duckdb]"` — verify dependency installs
2. Create test MDL project: 1 model (orders), 3 columns, DuckDB file with sample data
3. Configure: `tools.wren.enabled=true`, `tools.wren.project_path=...`, `tools.wren.data_source.type=duckdb`
4. `nanobot run "How many orders do we have?"` — expect SQL generation + result
5. `nanobot run "Show me top 5 orders by amount"` — expect correct SQL + table
6. Unit tests: `pytest tests/tools/test_wren_tool.py tests/wren/test_service.py`

---

## Phase 2: Production Hardening

### 2.1 Memory bridge — index schema into mem0

**New file:** `src/nanobot/wren/memory_bridge.py`

Bridges Wren's pure-function extractors into nanobot's mem0/ChromaDB. Uses:
- `wren.memory.schema_indexer.extract_schema_items(manifest)` — structured items (pure function, no LanceDB)
- `wren.memory.schema_indexer.manifest_hash(manifest)` — change detection (pure function)
- `wren.memory.seed_queries.generate_seed_queries(manifest)` — NL→SQL pairs (pure function)

```python
class WrenMemoryBridge:
    """Indexes Wren schema items and seed queries into mem0/ChromaDB."""

    def __init__(self, mem0_client: Mem0Client, engine_service: WrenEngineService):
        self._mem0 = mem0_client
        self._engine = engine_service
        self._indexed_hash: str | None = None

    async def index_schema(self, session_key: str = "__wren__") -> dict:
        """Index schema items + seed queries into mem0. Skips if hash matches."""

    async def search_schema_context(self, session_key: str, query: str, limit: int = 5) -> str:
        """Search mem0 for schema items relevant to a user question."""

    async def search_similar_queries(self, session_key: str, query: str, limit: int = 3) -> str:
        """Search mem0 for similar past NL→SQL pairs as few-shot examples."""

    async def log_successful_query(self, session_key: str, nl: str, sql: str) -> None:
        """Store a successful NL→SQL pair for future recall."""
```

**NL question extraction for query logging:** Use `SessionManager.get_unconsolidated_messages(key)[-1]` to extract the user's original natural language question from the session.

### 2.2 Dynamic schema context at query time

For large schemas (>30K chars / ~8K tokens), switch from full schema injection to semantic search:

```python
# In runner.py _get_wren_schema() — replace full dump with targeted retrieval:
if self._wren_memory and len(schema_desc) > self._schema_threshold:
    relevant = await self._wren_memory.search_schema_context(key, msg.content)
    similar = await self._wren_memory.search_similar_queries(key, msg.content)
    wren_schema = f"# Relevant Schema\n\n{relevant}"
    if similar:
        wren_schema += f"\n\n# Similar Past Queries\n\n{similar}"
```

Make the 30K threshold configurable in WrenConfig: `schema_threshold: int = 30000`.

### 2.3 Make auto-dry-plan configurable

Add `skip_dry_plan: bool = False` to WrenConfig. When True, SqlQueryTool skips the pre-query dry_plan validation (saves latency for trusted schemas).

### 2.4 Concurrency guard for reload()

Add `threading.Lock` to WrenEngineService to protect `reload()` from concurrent callers when multi-source is in use.

### 2.5 WrenProjectTool — MDL management

**New file:** `src/nanobot/tools/wren_project.py`

```python
class WrenProjectTool(Tool):
    name = "wren_project"
    # actions: validate, build, info, reload
```

### 2.6 dbt sync tool (calls Go binary)

**New file:** `src/nanobot/tools/wren_dbt.py`

```python
class WrenDbtSyncTool(Tool):
    name = "wren_dbt_sync"
    # Calls wren-launcher Go binary as subprocess
```

Config addition: `dbt_binary: str = "wren-launcher"` in WrenConfig.

### 2.7 Multi-datasource support (optional)

Extend WrenConfig to support named sources:
```python
sources: dict[str, WrenSourceConfig] = Field(default_factory=dict)
```

SqlQueryTool gets optional `source` parameter.

### 2.8 Phase 2 file summary

| Action | File |
|--------|------|
| Create | `src/nanobot/wren/memory_bridge.py` |
| Modify | `src/nanobot/tools/wren.py` — configurable dry_plan, error recovery |
| Create | `src/nanobot/tools/wren_project.py` |
| Create | `src/nanobot/tools/wren_dbt.py` |
| Modify | `src/nanobot/wren/service.py` — threading.Lock for reload |
| Modify | `src/nanobot/runner.py` — memory bridge init, dynamic schema context |
| Modify | `src/nanobot/config/schema.py` — dbt_binary, schema_threshold, skip_dry_plan, multi-source |
| Create | `tests/wren/test_memory_bridge.py` |
| Create | `tests/tools/test_wren_project.py` |

---

## Phase 3: Schema Maintenance (Enhancement — Last)

Port `DataSourceSchemaDetector` from WrenAI/wren-ui TypeScript to Python.

### 3.1 Alembic migration

**New file:** `alembic/versions/xxxx_add_wren_schema_changes.py`

```sql
CREATE TABLE wren_schema_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    change_json TEXT NOT NULL,
    resolve_json TEXT NOT NULL,
    detected_at INTEGER NOT NULL,
    resolved_at INTEGER
);
```

### 3.2 ORM model + Database methods

**File:** `src/nanobot/db.py`

Add `WrenSchemaChangeRow` ORM model and Database methods:
- `add_schema_change()`
- `get_unresolved_changes(source_name)`
- `resolve_schema_change(change_id)`

### 3.3 Schema detector service

**New file:** `src/nanobot/wren/schema_detector.py`

Ported from: `WrenAI/wren-ui/src/apollo/server/managers/dataSourceSchemaDetector.ts`

```python
class SchemaChangeType(str, Enum):
    DELETED_TABLES = "deleted_tables"
    DELETED_COLUMNS = "deleted_columns"
    MODIFIED_COLUMNS = "modified_columns"

class SchemaDetector:
    """Detects schema drift between MDL manifest and live database."""

    async def detect(self, source_name: str) -> SchemaChange | None:
        """Compare stored MDL vs live DB. Returns None if no changes."""
        # 1. Extract current schema from manifest (models → columns)
        # 2. Introspect live DB (INFORMATION_SCHEMA or ibis-server /metadata)
        # 3. Diff: find deleted tables, deleted columns, modified column types

    async def resolve(self, source_name: str, change_type: SchemaChangeType) -> dict:
        """Apply resolution cascade:
        - DELETED_TABLES → remove model + relationships + calculated fields
        - DELETED_COLUMNS → remove columns + dependent calculated fields
        - MODIFIED_COLUMNS → read-only (user must manually update)
        """
```

### 3.4 Integration

Add `check_schema` and `resolve_schema` actions to `WrenProjectTool`.
Optional: schedule periodic detection via nanobot's cron service.
After resolution: `engine_service.reload()` then `memory_bridge.index_schema()`.

### 3.5 Phase 3 file summary

| Action | File |
|--------|------|
| Create | `alembic/versions/xxxx_add_wren_schema_changes.py` |
| Modify | `src/nanobot/db.py` — WrenSchemaChangeRow + methods |
| Create | `src/nanobot/wren/schema_detector.py` |
| Modify | `src/nanobot/tools/wren_project.py` — add check_schema/resolve_schema |
| Create | `tests/wren/test_schema_detector.py` |

---

## Reusable code from wren-engine (Python imports, all lazy)

| What | Import |
|------|--------|
| Core engine | `from wren import WrenEngine, DataSource, WrenError` |
| Schema description | `from wren.memory.schema_indexer import describe_schema` |
| Schema items for embedding | `from wren.memory.schema_indexer import extract_schema_items` |
| Change detection hash | `from wren.memory.schema_indexer import manifest_hash` |
| Seed NL→SQL pairs | `from wren.memory.seed_queries import generate_seed_queries` |
| Build MDL from project | `from wren.context import build_json, validate_project, validate_manifest` |
| Encode manifest | `from wren.mdl import to_json_base64` |
| Engine config | `from wren.config import WrenConfig as WrenEngineConfig` |

All are pure Python except `WrenEngine` which depends on the `wren-core-py` Rust wheel.
