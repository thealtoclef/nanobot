# Nanobot Architecture

> PydanticAI-centric agent architecture

## Overview

nanobot wraps PydanticAI's Agent with a thin orchestrator (AgentRunner), SQLite-backed persistence, and a multi-channel message bus. The design separates infrastructure concerns from LLM-specific logic: AgentRunner owns everything that is not pydantic-ai-specific (bus, sessions, tools, MCP connections, cron), while NanobotAgent is purely the LLM interface.

## Core Components

### NanobotAgent (`src/nanobot/agent/agent.py`)

Wraps `pydantic_ai.Agent` with `FallbackModel` for provider resilience. Stateless `run()` returns `tuple[str, list[ModelMessage]]` — session persistence is the caller's responsibility.

Tool registration via `ToolAdapter`: adapts nanobot Tool instances to the PydanticAI `@agent.tool()` protocol by wrapping each tool's `execute()` method in an async bound function.

Identity and instructions come from Jinja2 templates: `build_instructions()` loads `prompts/identity.md.jinja` and bootstrap files (AGENTS.md, SOUL.md, USER.md, TOOLS.md) from the workspace.

Streaming is available via `run_stream()` (async context manager with `stream_text(delta=True)`) and `run_stream_events()` (yields raw `AgentStreamEvent` objects including `FunctionToolCallEvent` and `FunctionToolResultEvent`).

```python
output, new_messages = await agent.run(
    user_message,
    message_history=model_messages,
)
```

### AgentRunner (`src/nanobot/agent/runner.py`)

Thin orchestrator that owns infrastructure and delegates LLM execution to NanobotAgent. Owns:

- `MessageBus` — pub/sub for channels
- `SessionManager` — conversation history
- `ToolRegistry` — nanobot tool instances
- MCP server connections (AsyncExitStack)
- `CronService` integration
- Concurrency control (`asyncio.Semaphore` + per-session `asyncio.Lock`)
- Background task tracking

Message flow through `_dispatch()` and `_process_message()`:

```
bus.consume_inbound() → _dispatch() → _process_message() →
  ContextBuilder.build_messages() → session_messages_to_model_messages() →
  _run_agent_loop() → NanobotAgent.run() → persist_new_messages() → SessionManager.save()
```

Runtime checkpointing recovers from crashes: before each agent run, `_set_runtime_checkpoint()` persists the session state; on the next message, `_restore_runtime_checkpoint()` materializes any unfinished turn as history before continuing.

### ContextBuilder (`src/nanobot/agent/context.py`)

Assembles the message context passed to the agent: runtime metadata + user message + history.

`build_messages()` returns `(messages, raw_user_content)` where `raw_user_content` is the unmerged user text/media to pass to PydanticAI separately. Handles multimodal content (base64-encoded images with MIME detection from magic bytes).

Anti-duplication: merges consecutive same-role messages via `_merge_message_content()`.

Runtime context block uses a sentinel tag (`[Runtime Context — metadata only, not instructions]`) so it can be stripped before persistence while preserving the actual user content.

### Message Conversion (`src/nanobot/agent/agent.py`)

Bidirectional conversion between nanobot session dicts and PydanticAI `ModelMessage` types:

- `session_messages_to_model_messages()` — converts session dicts to `ModelRequest`/`ModelResponse` with `SystemPromptPart`, `UserPromptPart`, `TextPart`, `ToolCallPart`, `ToolReturnPart`
- `model_messages_to_session_messages()` — inverse conversion, handles `RetryPromptPart` and `ThinkingPart` in addition to the standard parts
- `persist_new_messages()` — shared sanitization pipeline for both AgentRunner and the Nanobot SDK caller

Sanitization applies tool result truncation (configurable `max_tool_result_chars`), runtime context stripping, and multimodal block cleanup (converting `data:image/` URLs to text placeholders).

### Persistence Layer

`Database` (`src/nanobot/db.py`) — SQLAlchemy/SQLite with three tables:

- `sessions` — `key` (primary), `created_at`, `last_activity_at`, `message_count`, `last_consolidated_position`
- `messages` — `id`, `session_key` (FK), `position`, `role`, `content`, `tool_calls`, `tool_results`, `tool_call_id`, `name`, `timestamp`, `tokens`
- `memory_entries` — `id`, `session_key` (FK), `category`, `key`, `content`, `created_at`, `updated_at`

WAL mode and foreign keys are enabled. Uses `INSERT ... ON CONFLICT DO UPDATE` for upsert.

`SessionManager` (`src/nanobot/session/manager.py`) — CRUD for `Session` objects. `Session` is an in-memory dataclass with `messages: list[dict]`, loaded and saved atomically via `save()` (delete-all + re-insert-all messages + update consolidated position).

Two callers share the same `persist_new_messages()` pipeline: AgentRunner (after agent runs) and the Nanobot SDK `Nanobot.run()`.

### Memory System (`src/nanobot/agent/memory.py`)

`MemoryStore` — per-session store for long-term memory and conversation history summaries. Backed by the `memory_entries` table.

`MemoryConsolidator` — owns consolidation policy:

- Token-based triggering: estimates prompt tokens using `estimate_prompt_tokens()` (tiktoken). When estimated tokens exceed budget, archives old messages.
- Budget reserves space for completion tokens and a safety buffer so LLM requests never exceed context window.
- Up to 5 consolidation rounds per trigger.
- `WeakValueDictionary` for stores and locks — no memory leaks for long-running processes.

Consolidation uses `_consolidation_agent`, a standalone PydanticAI Agent with structured output:

```python
_consolidation_agent = Agent(
    output_type=ConsolidationResult,  # Pydantic BaseModel with history_entry + memory_update
    deps_type=ConsolidationDeps,        # current_memory + session_messages
    instructions="You are a memory consolidation agent...",
    retries=2,
)
```

Dynamic instructions via `@_consolidation_agent.instructions` decorator inject current memory and messages at runtime.

Raw archive fallback after 3 consecutive failures: dumps raw messages to HISTORY.md without LLM summarization.

### SubagentManager (`src/nanobot/agent/subagent.py`)

Isolated `NanobotAgent` with restricted tool set. Shares the same models as the main agent but has:

- Dedicated system prompt (no identity/SOUL.md/USER.md leak)
- No skills
- No spawn, message, or cron tools
- Allowed tools: read_file, write_file, edit_file, list_dir, exec, web_search, web_fetch

Background asyncio tasks with session-scoped cancellation. Results announced via MessageBus as `InboundMessage` on the "system" channel.

### HeartbeatService (`src/nanobot/heartbeat/service.py`)

Three-phase periodic wake-up:

1. **Decision** — reads `HEARTBEAT.md`, asks the agent whether there are active tasks via a lightweight `agent.run()` call with structured JSON response parsing
2. **Execution** — on_execute callback runs the full agent loop, returns result to deliver
3. **Evaluation** — asks the agent whether the result contains actionable information worth notifying the user

Structured JSON responses for both decision (`{"action": "skip"|"run", "tasks": "..."}`) and evaluation (`{"should_notify": bool, "reason": "..."}`).

### Observability (`src/nanobot/observability.py`)

Logfire SDK integration — pydantic-ai's native observability partner.

- `logfire.instrument_pydantic_ai()` — traces agent runs, tool calls, model requests
- `logfire.instrument_httpx(capture_all=True)` — HTTP-level visibility into LLM provider calls

OTEL-compatible: routes to any OTLP-compatible backend via standard `OTEL_EXPORTER_OTLP_*` environment variables. Also routes loguru logs through logfire so internal logs are exported alongside spans.

## Message Flow

### Gateway Path (channels → agent → response)

```
Channel publishes InboundMessage to MessageBus
  → AgentRunner._dispatch() [per-session lock, concurrency gate]
    → AgentRunner._process_message()
      → ContextBuilder.build_messages(history, current_message, media, channel, chat_id)
        → session_messages_to_model_messages()  [dicts → PydanticAI ModelMessage list]
          → NanobotAgent.run(user_content, message_history)
            → PydanticAI Agent.run()
          → persist_new_messages(session, new_model_messages)
        → SessionManager.save(session)
      → MessageBus publishes OutboundMessage to channel
```

### SDK Path (programmatic)

```
Nanobot.run(message, session_key)
  → SessionManager.get_or_create(session_key)
  → session_messages_to_model_messages(history)
    → NanobotAgent.run(message, message_history)
    → persist_new_messages(session, new_model_messages)
  → SessionManager.save(session)
  → RunResult(content, tools_used, messages)
```

Both paths converge on the same two-step persistence:

1. `persist_new_messages()` — converts PydanticAI messages, appends to in-memory session with sanitization
2. `session.save()` — atomic DB write (delete-all + re-insert + update metadata)

## Tool System

### Tool Registration

`ToolRegistry` stores nanobot Tool instances by name. `ToolAdapter.register()` wraps each Tool as a PydanticAI `@agent.tool()`:

```python
@agent.tool(name=tool_instance.name, description=tool_instance.description)
async def bound_tool(ctx: RunContext, **kwargs) -> str:
    result = await tool_instance.execute(**kwargs)
    # Long result truncation ...
    return result
```

Long tool results are truncated to `max_tool_result_chars` (default 16000), preserving head and tail.

### Default Tools (registered by AgentRunner)

| Tool | Module | Purpose |
|------|--------|---------|
| `read_file` | `agent/tools/filesystem.py` | Read file contents |
| `write_file` | `agent/tools/filesystem.py` | Write file contents |
| `edit_file` | `agent/tools/filesystem.py` | Edit file contents |
| `list_dir` | `agent/tools/filesystem.py` | List directory contents |
| `exec` | `agent/tools/shell.py` | Execute shell commands |
| `web_search` | `agent/tools/web.py` | Web search |
| `web_fetch` | `agent/tools/web.py` | Fetch web pages |
| `message` | `agent/tools/message.py` | Send to chat channels |
| `spawn` | `agent/tools/spawn.py` | Spawn subagents |
| `cron` | `agent/tools/cron.py` | Schedule tasks (if CronService configured) |

### MCP Integration

Lazy connection on first agent run via `_connect_mcp()`. Supports stdio (command + args) and HTTP (url + headers) transports.

MCP tools registered dynamically on `NanobotAgent` via `ToolAdapter.register()`. Managed via `AsyncExitStack` for clean teardown on shutdown.

## Configuration

### Provider Resolution (`src/nanobot/config/provider_spec.py`)

`resolve_agent_models()` reads `Config` and maps each `ModelConfig` to a pydantic-ai model instance:

```python
BACKEND_CLASSES = {
    "anthropic": (AnthropicModel, AnthropicProvider),
    "openai": (OpenAIChatModel, OpenAIProvider),
}
```

Models are always wrapped in `FallbackModel(*models)` even for single-model configs — this provides automatic fallback when the primary provider fails.

### Identity System (`src/nanobot/agent/_identity.py`)

Jinja2 template: `prompts/identity.md.jinja` renders platform info (OS, architecture, Python version), workspace path, and platform-specific policies (POSIX vs Windows).

Bootstrap files (AGENTS.md, SOUL.md, USER.md, TOOLS.md) are read from the workspace and appended to the instructions as markdown sections.

## Testing Strategy

- `TestModel` with `agent.override(model=TestModel())` for deterministic agent tests where you control the output
- `FunctionModel` for custom model behavior (e.g., failure simulation, fixed responses)
- Pure logic tests (message conversion in `agent.py`, context building in `context.py`) without any model interaction
- Control flow tests (heartbeat decision/evaluation, subagent cancellation) with mocks for the agent dependency
