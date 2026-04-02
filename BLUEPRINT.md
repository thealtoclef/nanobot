# Nanobot Production Hardening Blueprint

> **Purpose**: Full refactor of Nanobot to production-grade foundations.
> Stand on shoulders of giants — no reinventing wheels.
>
> **Stack**:
> - **Agent framework**: PydanticAI (wraps official OpenAI/Anthropic SDKs)
> - **Data models**: Pydantic BaseModel everywhere (no dataclasses)
> - **Base classes**: `BaseModel + ABC` — config validation + interface enforcement + auto-observability
> - **Logging**: Python stdlib `logging` (only first-class OTEL path)
> - **Observability**: OpenTelemetry (platform) + Langfuse (LLM) via OTEL
> - **Security**: LLM Guard (prompt injection) + detect-secrets (leak detection)
> - **Providers**: User-defined in config, resolved via factory. Zero provider logic in core.
> - **Packaging**: Modular optional extras (`nanobot[openai,telegram,langfuse]`)
>
> **Approach**: Full refactor — no backward compatibility. No feature loss.
> No Logfire (closed-source). No loguru (no OTEL support). No boilerplate provider registry.
>
> **Origin**: Nanobot is a rewrite from OpenClaw. Architecture follows OpenClaw's principle
> of a clean, provider/channel/tool-agnostic core with pluggable extensions.
>
> **Repositories**:
> - Nanobot: `/home/blue/repos/nanobot/`
> - ZeroClaw (reference only): `/home/blue/repos/zeroclaw/`

---

## Table of Contents

1. [Design Decisions](#1-design-decisions)
2. [PydanticAI Refactor Plan](#2-pydanticai-refactor-plan)
3. [Observability Architecture](#3-observability-architecture)
4. [Implementation Phases](#4-implementation-phases)
5. [Modularization](#5-modularization)
6. [Testing Strategy](#6-testing-strategy)
7. [Appendix](#7-appendix)

---

## 1. Design Decisions

### 1.1 PydanticAI as Core Agent Framework

Replace `AgentRunner` + `AgentLoop` with PydanticAI `Agent`. Full refactor.

PydanticAI v1.x:
- Wraps the **official** `openai.AsyncOpenAI` and `anthropic.AsyncAnthropic` SDKs internally.
  Zero custom HTTP.
- `Agent.iter()` API: intercept every node, inject hooks, security checks, observability.
- Native OTEL spans via `Agent.instrument_all()` — agent run, model request, tool execution.
- Prompt caching (Anthropic), extended thinking (Claude), streaming, structured output.
- Composable toolsets: `FunctionToolset`, `FilteredToolset`, `ApprovalRequiredToolset`.
- Dependency injection: `RunContext[DepsType]` carries session/workspace/channel state.
- Built-in model fallback: pass a list of `Model` objects, auto-failover on errors.
- `TestModel` for deterministic unit testing without real LLM calls.

### 1.2 Pydantic Everywhere — No Dataclasses

All data models use `pydantic.BaseModel`. No `@dataclass`.
Use `ConfigDict(frozen=True)` for immutability where needed.

### 1.3 Base Classes — `BaseModel + ABC` + Auto-Observability

Every base class (channels, tools, services) uses `BaseModel + ABC`:
- **Pydantic** validates config, provides serialization, enforces types
- **ABC** enforces the interface contract (`@abstractmethod`)
- **Public methods** auto-instrument OTEL tracing, metrics, logging
- **Subclasses** implement `_method()` with business logic only
- **Zero observability code** in any subclass, channel, tool, or plugin

### 1.4 stdlib logging — No loguru

loguru has **no OTEL support** (confirmed by OTEL maintainers). stdlib `logging` is the
**only first-class OTEL logging path**. Two handlers: `StreamHandler` (stderr) +
OTEL `LoggingHandler` (collector). `LoggingInstrumentor` auto-injects `trace_id`/`span_id`.

### 1.5 No Logfire

PydanticAI's native OTEL works without Logfire. `Agent.instrument_all()` emits standard
OTEL spans. Logfire is a closed-source SaaS — we don't need it.

### 1.6 Uniform Interfaces — Core Architecture Principle

Inspired by OpenClaw: the core is a clean, provider/channel/tool-agnostic engine.

```
┌──────────────────────────────────────────────────────────┐
│                     NANOBOT CORE                         │
│                                                          │
│  Agent loop (PydanticAI iter())                          │
│    ├── calls model.run()     ← Model ABC (uniform)      │
│    ├── calls toolset.call()  ← Toolset ABC (uniform)    │
│    └── calls channel.send()  ← BaseChannel (uniform)    │
│                                                          │
│  Session, Memory, Cron, Bus — all provider-agnostic      │
│  Observability — auto-instrumented in base classes        │
│  Security — plugged in at boundaries                     │
│                                                          │
│  ZERO if/else for provider type                          │
│  ZERO if/else for channel type                           │
│  ZERO if/else for tool type                              │
└──────────────────────────────────────────────────────────┘
         ▲               ▲               ▲
         │               │               │
    ┌────┴────┐    ┌─────┴─────┐   ┌─────┴─────┐
    │Providers│    │ Channels  │   │   Tools   │
    │(optional│    │ (optional │   │ (plugin   │
    │ extras) │    │  extras)  │   │  system)  │
    │         │    │           │   │           │
    │ openai  │    │ telegram  │   │ filesystem│
    │anthropic│    │ slack     │   │ shell     │
    │         │    │ discord   │   │ web       │
    └─────────┘    └───────────┘   └───────────┘
```

Each extension point:
- **Base class**: `BaseModel + ABC`
- **Public methods**: auto-instrumented (OTEL spans, metrics, logging)
- **Private methods**: `_method()` — subclass implements business logic only
- **Registration**: entry-point discovery or config-driven factory
- **Dependencies**: optional extras (`pip install nanobot[telegram]`)

### 1.7 Provider Simplification

Remove the 31-entry `PROVIDERS` tuple. User defines providers in config:

```json
{
  "providers": {
    "my-anthropic": {
      "backend": "anthropic",
      "apiKey": "$ANTHROPIC_API_KEY"
    },
    "my-deepseek": {
      "backend": "openai",
      "baseUrl": "https://api.deepseek.com/v1",
      "apiKey": "$DEEPSEEK_API_KEY"
    },
    "my-local": {
      "backend": "openai",
      "baseUrl": "http://localhost:8080/v1",
      "apiKey": "not-needed"
    }
  }
}
```

Two backends (for now): `openai` (all OpenAI-compatible), `anthropic`. Skip gemini.
Provider SDKs are optional extras: `nanobot[openai]`, `nanobot[anthropic]`.

**Model references** use `provider-name/model-name` format (e.g., `my-anthropic/claude-sonnet-4-5`).
Also supports auto-detection by model name (e.g., `claude-sonnet-4-5` → finds first `anthropic` provider).
Both formats work identically for primary and fallback models.

### 1.8 Modular Package — Optional Extras

Base install has no channels, no provider SDKs:
```
pip install nanobot                               # Core only
pip install nanobot[openai]                       # + OpenAI-compatible
pip install nanobot[anthropic]                    # + Anthropic
pip install nanobot[telegram]                     # + Telegram channel
pip install nanobot[langfuse]                     # + Langfuse LLM tracing
pip install nanobot[security]                     # + LLM Guard + detect-secrets
pip install nanobot[openai,anthropic,telegram,langfuse]  # Typical production
pip install nanobot[all]                          # Everything
```

### 1.9 Security — Established Libraries

| Need | Library | License |
|------|---------|---------|
| Prompt injection | **LLM Guard** (protectai/llm-guard) — 15 input scanners, 20 output scanners | MIT |
| Credential leaks | **detect-secrets** (Yelp) — plugin architecture, scans arbitrary text | Apache 2.0 |
| Sandboxing | **Docker SDK** (`docker` package) — industry standard | Apache 2.0 |
| Output guardrails | **Guardrails AI** (optional) — schema validation + retry | Apache 2.0 |

---

## 2. PydanticAI Refactor Plan

### 2.1 Feature Inventory — What Must Be Preserved

Every feature from the current codebase that must survive the refactor:

#### From `agent/runner.py` (AgentRunner):
| Feature | How to preserve in PydanticAI |
|---------|-------------------------------|
| Iteration loop with tool-call/final-response branching | `agent.iter()` — `ModelRequestNode` vs `CallToolsNode` vs `End` |
| Empty final response retry with finalization prompt | Post-`End` check: if empty, re-run with explicit "please respond" |
| Context snipping with token budgets + safety buffers | Pre-`ModelRequestNode` logic: trim `message_history` before each step |
| Tool result truncation (max_tool_result_chars) | Custom toolset wrapper that truncates return values |
| Tool batching by `concurrency_safe` attribute | `CallToolsNode` — PydanticAI runs tools concurrently |
| External lookup repeated-call detection | Track call hashes in inter-node state |
| Fatal error tracking that breaks loop | Exception handling around `agent_run.next(node)` |
| Checkpoint emission (3 phases) | Emit between nodes in `iter()` loop |
| Usage accumulation across iterations | `agent_run.result.usage()` — built-in |
| Tool preparation via `prepare_call` | PydanticAI `prepare_tools` callback |

#### From `agent/loop.py` (AgentLoop):
| Feature | How to preserve in PydanticAI |
|---------|-------------------------------|
| Per-session serial, cross-session concurrent execution | Keep: asyncio locks per session_key |
| Streaming with delta filtering + resuming flag | `node.stream(run.ctx)` with `TextPartDelta` events |
| Runtime checkpoint for crash recovery | Keep: persist state between nodes |
| MCP lazy connection + cleanup | Keep: `AsyncExitStack` lifecycle unchanged |
| Memory consolidation as background task | Keep: schedule after turn, use separate PydanticAI agent |
| Command dispatch before agent loop | Keep: route commands before `agent.run()` |
| System message vs regular message routing | Keep: routing logic stays in loop |
| Turn saving with sanitization + truncation | Keep: convert PydanticAI `ModelMessage` to storage format |
| Message tool send suppression | Keep: check after `End` node |
| `<think>` tag stripping | `finalize_content` → post-process `End` result |

#### From `agent/hook.py`:
| Hook | PydanticAI mapping |
|------|--------------------|
| `before_iteration` | Before each `agent_run.next(node)` call |
| `on_stream` / `on_stream_end` | Inside `node.stream(run.ctx)` event loop |
| `before_execute_tools` | Before `CallToolsNode` executes (intercept node) |
| `after_iteration` | After each `agent_run.next(node)` returns |
| `finalize_content` | After `End` node, transform `result.output` |

#### From `agent/memory.py`:
| Feature | Approach |
|---------|----------|
| Two-layer storage (MEMORY.md + HISTORY.md) | Keep as-is |
| LLM-driven consolidation with tool forcing | Separate PydanticAI Agent with `save_memory` tool |
| Failure threshold → raw archive fallback | Keep logic |
| Token-based consolidation with budget/target | Keep: budget = context_window - completion - safety |

#### From `agent/subagent.py`:
| Feature | Approach |
|---------|----------|
| Background task spawning with task_id | Keep: asyncio tasks with PydanticAI agents |
| Subagent tool registry (reduced toolset) | PydanticAI `FilteredToolset` |
| System announcement via message bus | Keep: bus publish on completion |
| Cancellation by session | Keep: cancel asyncio tasks |

#### From `providers/base.py`:
| Feature | PydanticAI handling |
|---------|---------------------|
| Message sanitization | PydanticAI handles internally |
| Transient error detection + retry | PydanticAI built-in; configure via `httpx` client |
| Image stripping fallback | Pre-process messages before passing to agent |
| Retry-after header parsing | PydanticAI respects SDK retry behavior |

### 2.2 Agent Architecture

```python
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict

class NanobotDeps(BaseModel):
    """Dependency injection — all per-request state."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    session_key: str
    workspace: Path
    channel: str | None = None
    chat_id: str | None = None
    memory_store: MemoryStore | None = None

# Main agent — replaces AgentRunner + AgentLoop
agent = Agent(
    model,                  # uniform Model interface — could be any provider
    deps_type=NanobotDeps,
    output_type=str,
    instrument=True,        # OTEL spans
    retries=3,
)

# Tools — plain functions, auto-instrumented by InstrumentedToolset
@agent.tool
async def read_file(ctx: RunContext[NanobotDeps], path: str) -> str:
    """Read a file from the workspace."""
    resolved = resolve_path(ctx.deps.workspace, path)
    return resolved.read_text()
```

### 2.3 Agent Iteration Loop (replaces runner.py)

```python
async def run_agent(
    agent: Agent[NanobotDeps, str],
    prompt: str,
    deps: NanobotDeps,
    hooks: list[AgentHook],
    message_history: list[ModelMessage] | None = None,
    max_iterations: int = 40,
) -> AgentRunResult:
    """Main agent execution — drives PydanticAI iter() with nanobot's features."""

    iteration = 0
    async with agent.iter(prompt, deps=deps, message_history=message_history) as run:
        async for node in run:
            iteration += 1
            if iteration > max_iterations:
                break

            for hook in hooks:
                await hook.before_iteration(context)

            if Agent.is_model_request_node(node):
                # Context snipping: trim message_history if over token budget
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                for hook in hooks:
                                    await hook.on_stream(ctx, event.delta.content_delta)

            elif Agent.is_call_tools_node(node):
                for hook in hooks:
                    await hook.before_execute_tools(context)
                # Security: prompt guard on tool arguments

            for hook in hooks:
                await hook.after_iteration(context)

    result = run.result
    content = result.output if result else ""
    for hook in hooks:
        content = hook.finalize_content(context, content) or content

    return AgentRunResult(
        final_content=content,
        usage=result.usage() if result else {},
        messages=result.all_messages() if result else [],
    )
```

### 2.4 Provider Factory + Model Resolution

```python
def resolve_model(config: Config, model_ref: str) -> Model:
    """Resolve a model reference to a PydanticAI Model.

    Supports two formats:
      - Explicit: "my-anthropic/claude-sonnet-4-5" → split at /, look up provider
      - Auto-detect: "claude-sonnet-4-5" → detect backend from model name
    """
    if "/" in model_ref:
        provider_name, model_name = model_ref.split("/", 1)
        provider_config = config.providers[provider_name]
    else:
        model_name = model_ref
        provider_config = detect_provider(config, model_name)

    return build_model(provider_config, model_name)


def detect_provider(config: Config, model_name: str) -> ProviderConfig:
    """Auto-detect provider from model name."""
    if model_name.startswith("claude"):
        return next(p for p in config.providers.values() if p.backend == "anthropic")
    return next(p for p in config.providers.values() if p.backend == "openai")


def build_model(provider_config: ProviderConfig, model_name: str) -> Model:
    """Build a PydanticAI Model. Raises ImportError if SDK not installed."""
    if provider_config.backend == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=provider_config.base_url, api_key=provider_config.api_key),
        )
    elif provider_config.backend == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        return AnthropicModel(
            model_name,
            provider=AnthropicProvider(api_key=provider_config.api_key),
        )
    raise ValueError(f"Unknown backend: {provider_config.backend}")


def build_model_chain(config: Config) -> Model | list[Model]:
    """Build primary + fallback models. PydanticAI handles failover internally."""
    primary = resolve_model(config, config.agents.defaults.model)
    fallbacks = config.agents.defaults.fallback_models
    if not fallbacks:
        return primary
    return [primary] + [resolve_model(config, ref) for ref in fallbacks]
```

### 2.5 Message Format

PydanticAI uses `ModelMessage` Pydantic models. For session persistence:
```python
def to_storage(messages: list[ModelMessage]) -> list[dict]:
    return [msg.model_dump() for msg in messages]

def from_storage(data: list[dict]) -> list[ModelMessage]:
    return [ModelMessage.model_validate(d) for d in data]
```

---

## 3. Observability Architecture

### 3.1 Instrumentation Coverage

PydanticAI `Agent.instrument_all()` auto-instruments the **agent core**.
Everything else uses **manual OTEL spans in base classes** — subclasses never touch observability.

#### Auto-instrumented by PydanticAI:
| What | Also flows to Langfuse? |
|------|-------------------------|
| Agent run lifecycle (start → end) | Yes |
| Every LLM API call (prompts, completions, tokens, latency) | Yes (shows cost) |
| Every tool execution (name, args, duration, result) | Yes |
| Streaming chunks | Yes |

#### Auto-instrumented by base classes (manual OTEL, but subclasses don't see it):
| What | Base class handles it |
|------|----------------------|
| Channel message receive/send | `BaseChannel.handle_message()`, `.send()` |
| Cron job execution | `BaseCronService.execute_job()` |
| Tool metrics + logging | `InstrumentedToolset.call_tool()` |
| Session, memory, MCP operations | Respective base service classes |

#### OTEL metrics (in base classes, not subclasses):
| Metric | Type |
|--------|------|
| `nanobot.channel.messages` | Counter [channel, direction] |
| `nanobot.tool.calls` | Counter [tool, success] |
| `nanobot.tool.duration_ms` | Histogram [tool] |
| `nanobot.sessions.active` | UpDownCounter |
| `nanobot.errors` | Counter [component] |
| `nanobot.cron.executions` | Counter [job, success] |

**Complete trace example:**
```
channel.receive (BaseChannel auto-span)
  └── agent.run (PydanticAI auto-span)
        ├── gen_ai.request → Langfuse
        ├── tool.execute: read_file → Langfuse
        ├── gen_ai.request → Langfuse
        └── tool.execute: exec → Langfuse
  └── session.save (BaseService auto-span)
  └── memory.consolidate (BaseService auto-span)
        └── agent.run (PydanticAI auto-span — separate consolidation agent)
```

### 3.2 OTEL Setup (No Logfire)

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import set_tracer_provider
from pydantic_ai import Agent

resource = Resource.create({"service.name": "nanobot"})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://collector:4318/v1/traces"))
)
set_tracer_provider(tracer_provider)
Agent.instrument_all()
```

### 3.3 Langfuse (via OTEL)

Langfuse v3+ is an OTEL-compatible backend. PydanticAI spans flow to Langfuse automatically.

```python
from langfuse import get_client
langfuse = get_client()    # registers as OTEL span processor
Agent.instrument_all()     # spans → Langfuse

# Optional metadata
from langfuse.otel import propagate_attributes
with propagate_attributes(user_id="user-123", session_id="session-456"):
    result = await agent.run(prompt, deps=deps)
```

Config via env vars:
```
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 3.4 Logging (stdlib + OTEL)

```python
import logging
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor

def setup_logging(otel_endpoint: str | None = None, service_name: str = "nanobot"):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Handler 1: stderr
    stderr = logging.StreamHandler()
    stderr.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(otelTraceID)s] %(name)s - %(message)s"
    ))
    root.addHandler(stderr)

    # Handler 2: OTEL export
    if otel_endpoint:
        resource = Resource.create({"service.name": service_name})
        log_provider = LoggerProvider(resource=resource)
        log_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{otel_endpoint}/v1/logs"))
        )
        root.addHandler(LoggingHandler(level=logging.INFO, logger_provider=log_provider))

    # Auto-inject trace_id, span_id into every log record
    LoggingInstrumentor().instrument(set_logging_format=False)
```

### 3.5 Metrics (OTEL)

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(endpoint=f"{otel_endpoint}/v1/metrics"),
    export_interval_millis=30000,
)
meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
meter = meter_provider.get_meter("nanobot")
```

### 3.6 Instrumented Base Classes

#### BaseChannel

```python
class BaseChannel(BaseModel, ABC):
    """Pydantic validates config, ABC enforces interface, public methods auto-instrument."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    enabled: bool = True
    allow_from: list[str] = []

    @abstractmethod
    async def _start(self) -> None: ...
    @abstractmethod
    async def _handle_message(self, raw_message: Any) -> InboundMessage: ...
    @abstractmethod
    async def _send(self, msg: OutboundMessage) -> None: ...

    async def start(self) -> None:
        with tracer.start_as_current_span("channel.start", attributes={"channel": self.name}):
            logger.info("Starting channel", extra={"channel": self.name})
            await self._start()

    async def handle_message(self, raw_message: Any) -> InboundMessage:
        with tracer.start_as_current_span("channel.receive", attributes={"channel": self.name}) as span:
            channel_messages.add(1, {"channel": self.name, "direction": "inbound"})
            try:
                msg = await self._handle_message(raw_message)
                span.set_attribute("chat_id", msg.chat_id)
                return msg
            except Exception:
                channel_errors.add(1, {"channel": self.name})
                logger.exception("Channel receive error", extra={"channel": self.name})
                raise

    async def send(self, msg: OutboundMessage) -> None:
        with tracer.start_as_current_span("channel.send", attributes={"channel": self.name}):
            channel_messages.add(1, {"channel": self.name, "direction": "outbound"})
            await self._send(msg)
```

Subclass — business logic only:
```python
class TelegramChannel(BaseChannel):
    name: str = "telegram"
    token: str                    # Pydantic validates
    parse_mode: str = "HTML"

    async def _start(self) -> None: ...
    async def _handle_message(self, update: Update) -> InboundMessage: ...
    async def _send(self, msg: OutboundMessage) -> None: ...
```

#### InstrumentedToolset

PydanticAI auto-instruments tool OTEL spans. This adds metrics + logging:
```python
class InstrumentedToolset(FunctionToolset):
    async def call_tool(self, call, run_context):
        start = time.monotonic()
        logger.info("Tool call: %s", call.tool_name, extra={"tool": call.tool_name})
        try:
            result = await super().call_tool(call, run_context)
            tool_calls_metric.add(1, {"tool": call.tool_name, "success": "true"})
            tool_duration_metric.record((time.monotonic() - start) * 1000, {"tool": call.tool_name})
            return result
        except Exception:
            tool_calls_metric.add(1, {"tool": call.tool_name, "success": "false"})
            logger.exception("Tool error: %s", call.tool_name)
            raise
```

#### BaseService (cron, session, memory)

Same `BaseModel + ABC` pattern with auto-instrumented public methods.

### 3.7 Config

```json
{
  "observability": {
    "otel": { "enabled": true, "endpoint": "http://localhost:4318", "serviceName": "nanobot" },
    "langfuse": { "enabled": true }
  }
}
```

Also configurable via env vars: `OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_ENDPOINT`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.

---

## 4. Implementation Phases

### Phase 0: Foundation Refactor

**Goal**: Replace core with PydanticAI, stdlib logging, Pydantic models, simplified providers.

| Task | Details |
|------|---------|
| **0.1 Pydantic migration** | Convert all `@dataclass` to `BaseModel` across codebase |
| **0.2 Replace loguru** | stdlib `logging` + `LoggingInstrumentor` for OTEL trace correlation |
| **0.3 PydanticAI agent** | `agent/core.py`: port tools, hooks, context snipping, memory, subagents |
| **0.4 Provider simplification** | Delete 31-entry registry, config-driven factory with `resolve_model()` |
| **0.5 OTEL + Langfuse** | `Agent.instrument_all()`, `TracerProvider`, `MeterProvider`, `LoggerProvider` |
| **0.6 Modularize channels** | Optional extras, `BaseModel + ABC` base with auto-observability |
| **0.7 Modularize tools** | Entry-point discovery, `InstrumentedToolset` base |

**Dependencies:**
```toml
[project.dependencies]
pydantic = ">=2.12.0"
pydantic-ai-slim = ">=1.0.0"
opentelemetry-api = ">=1.20.0"
opentelemetry-sdk = ">=1.20.0"
opentelemetry-exporter-otlp-proto-http = ">=1.20.0"
opentelemetry-instrumentation-logging = ">=0.50b0"
httpx = ">=0.28.0"

[project.optional-dependencies]
openai = ["pydantic-ai-slim[openai]"]
anthropic = ["pydantic-ai-slim[anthropic]"]
langfuse = ["langfuse>=3.22.0"]
telegram = ["python-telegram-bot>=22.6"]
slack = ["slack-sdk>=3.39.0"]
discord = ["discord.py>=2.5.2"]
matrix = ["matrix-nio>=0.25.2", "mistune", "nh3"]
feishu = ["lark-oapi>=1.5.0"]
dingtalk = ["dingtalk-stream>=0.24.0"]
qq = ["qq-botpy>=1.2.0"]
wecom = ["wecom-aibot-sdk-python"]
weixin = ["qrcode", "pycryptodome"]
security = ["llm-guard>=0.3.16", "detect-secrets>=1.5.0", "cryptography>=44.0.0"]
docker = ["docker>=7.0.0"]
api = ["aiohttp>=3.9.0"]
all-providers = ["nanobot[openai,anthropic]"]
all-channels = ["nanobot[telegram,slack,discord,matrix,feishu,dingtalk,qq,wecom,weixin]"]
all = ["nanobot[all-providers,langfuse,all-channels,security,docker,api]"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "ruff>=0.8"]
```

**Removed from core:** `loguru`, `anthropic`, `openai` (now optional).

---

### Phase 1: Security Hardening

**Goal**: Prompt injection detection, credential leak prevention, secrets encryption, audit.

| Task | Library/Approach |
|------|-----------------|
| **1.1 Prompt injection** | LLM Guard input scanners. Wire into message ingress. |
| **1.2 Credential leak detection** | detect-secrets. Wire into outbound messages + tool results. |
| **1.3 Secrets encryption** | `cryptography` — ChaCha20-Poly1305. `enc2:<hex>` in config. Auto-decrypt on load. |
| **1.4 Audit logging** | NDJSON at `~/.nanobot/audit.log`. Events: tool exec, LLM calls, security alerts. |
| **1.5 Security policy** | `auto_approve_tools` whitelist + `max_actions_per_hour` + PydanticAI `ApprovalRequiredToolset` |

Config:
```json
{
  "security": {
    "promptGuard": { "enabled": true, "sensitivity": 0.7 },
    "leakDetector": { "enabled": true },
    "secrets": { "encrypt": true },
    "audit": { "enabled": true, "maxSizeMb": 100 },
    "policy": { "autoApproveTools": ["read_file", "list_dir", "web_search"], "maxActionsPerHour": 200 }
  }
}
```

---

### Phase 2: Core Agent Enhancements

| Task | Details |
|------|---------|
| **2.1 Model fallback** | PydanticAI native: `Agent([model_primary, model_fallback])`. Config-driven via `build_model_chain()`. |
| **2.2 Loop detection** | Track tool call hashes between `iter()` nodes. Detect 3+ identical or A-B-A-B patterns. |
| **2.3 Cancellation** | `asyncio.Event` as token. Check between nodes. `/stop` triggers it. |
| **2.4 Named agent profiles** | Config: `agents.profiles.code_reviewer = {model, systemPrompt, allowedTools}` |
| **2.5 Cron enhancements** | Shell job type, per-job tool allowlists, startup catch-up. |

---

### Phase 3: Operational Maturity

Follow ZeroClaw patterns:

| Task | New Files |
|------|-----------|
| Emergency stop | `security/estop.py` |
| Workspace isolation | `config/workspace.py` |
| Namespaced memory | Modify `agent/memory.py` |
| SOP engine | `sop/types.py`, `sop/engine.py` |
| Docker runtime for tools | `runtime/docker.py` |
| Declarative cron sync | Modify `cron/service.py` |
| Resource limits | `security/resource_limits.py` |
| Data retention policies | `security/retention.py` |

---

### Phase 4: Tools & Integrations (LOW Priority)

Prefer MCP servers over native implementations.

### Phase 5: Advanced Features (LATER)

Swarm, knowledge graph, memory decay, agent evaluation, etc.

---

## 5. Modularization

### 5.1 Package Structure

```
nanobot/
├── agent/
│   ├── core.py               # NEW: PydanticAI agent + iter() loop
│   ├── loop.py               # REFACTORED: message processing, session management
│   ├── hook.py               # KEPT: hook interface (adapted for iter() nodes)
│   ├── memory.py             # KEPT: consolidation (uses PydanticAI agent)
│   ├── subagent.py           # REFACTORED: uses PydanticAI Agent
│   ├── loop_detector.py      # NEW (Phase 2)
│   ├── cancellation.py       # NEW (Phase 2)
│   └── tools/
│       ├── base.py           # REFACTORED: InstrumentedToolset (BaseModel + ABC)
│       ├── filesystem.py     # KEPT (as PydanticAI tools)
│       ├── shell.py          # KEPT
│       ├── web.py            # KEPT
│       ├── mcp.py            # KEPT
│       ├── message.py        # KEPT
│       ├── spawn.py          # KEPT
│       └── cron.py           # KEPT
├── security/
│   ├── network.py            # KEPT: SSRF protection
│   ├── guard.py              # NEW: LLM Guard wrapper (Phase 1)
│   ├── secrets.py            # NEW: ChaCha20-Poly1305 (Phase 1)
│   ├── audit.py              # NEW: NDJSON audit (Phase 1)
│   ├── policy.py             # NEW: policy + rate limiting (Phase 1)
│   ├── estop.py              # NEW (Phase 3)
│   ├── resource_limits.py    # NEW (Phase 3)
│   └── retention.py          # NEW (Phase 3)
├── observability/
│   ├── __init__.py           # setup_observability() factory
│   ├── tracing.py            # OTEL TracerProvider + Agent.instrument_all()
│   ├── metrics.py            # OTEL MeterProvider + instruments
│   └── logging.py            # stdlib logging + OTEL LoggingHandler
├── providers/
│   ├── factory.py            # resolve_model() + build_model() + build_model_chain()
│   └── config.py             # ProviderConfig Pydantic model
├── config/
│   ├── schema.py             # REFACTORED: simplified Pydantic models
│   ├── loader.py             # KEPT
│   └── workspace.py          # NEW (Phase 3)
├── session/
│   └── manager.py            # REFACTORED: PydanticAI ModelMessage storage
├── channels/
│   ├── base.py               # REFACTORED: BaseChannel (BaseModel + ABC + auto-observability)
│   ├── registry.py           # KEPT: pkgutil + entry-point discovery
│   ├── manager.py            # KEPT: channel lifecycle
│   └── ...                   # Impls: only _handle_message, _send, _start
├── sop/                      # NEW (Phase 3)
├── runtime/                  # NEW (Phase 3)
├── cron/                     # KEPT
├── bus/                      # KEPT
├── cli/                      # KEPT
└── api/                      # KEPT (optional)
```

### 5.2 What Gets Deleted

| File/Code | Reason |
|-----------|--------|
| `providers/registry.py` (31-entry PROVIDERS tuple) | Replaced by user config + factory |
| `providers/openai_compat_provider.py` | PydanticAI's `OpenAIProvider` |
| `providers/anthropic_provider.py` | PydanticAI's `AnthropicProvider` |
| `providers/azure_openai_provider.py` | PydanticAI's `OpenAIProvider` with Azure config |
| `providers/github_copilot_provider.py` | Specialized — add back if needed |
| `providers/openai_codex_provider.py` | Specialized — add back if needed |
| `providers/openai_responses/` | PydanticAI handles internally |
| `providers/base.py` (retry, sanitization) | PydanticAI handles internally |
| `providers/fallback.py` | PydanticAI native fallback |
| `agent/runner.py` | Replaced by `agent/core.py` with PydanticAI `iter()` |
| `loguru` dependency | Replaced by stdlib logging |

---

## 6. Testing Strategy

```
tests/
├── agent/
│   ├── test_core.py            # PydanticAI agent (uses TestModel)
│   ├── test_loop.py            # Message processing pipeline
│   ├── test_memory.py          # Consolidation flow
│   ├── test_loop_detector.py   # Loop detection (Phase 2)
│   └── test_cancellation.py    # Cancellation (Phase 2)
├── observability/
│   ├── test_tracing.py         # OTEL span emission
│   ├── test_metrics.py         # Metric recording
│   └── test_logging.py         # stdlib + OTEL setup
├── security/
│   ├── test_guard.py           # LLM Guard (Phase 1)
│   ├── test_secrets.py         # Encryption round-trip (Phase 1)
│   ├── test_audit.py           # NDJSON logging (Phase 1)
│   └── test_policy.py          # Auto-approve, rate limiting (Phase 1)
├── providers/
│   └── test_factory.py         # resolve_model, build_model_chain
├── channels/
│   └── test_base.py            # BaseChannel instrumentation
├── tools/
│   └── test_*.py               # Per-tool tests
└── sop/
    └── test_engine.py          # SOP execution (Phase 3)
```

PydanticAI `TestModel` for deterministic testing:
```python
from pydantic_ai.models.test import TestModel

def test_agent_tool_call():
    model = TestModel()
    agent = Agent(model, ...)
    result = agent.run_sync("test prompt")
    assert model.last_model_request_parameters.function_tools[0].name == "read_file"
```

---

## 7. Appendix

### 7.1 Full Config Target

```json
{
  "agents": {
    "defaults": {
      "model": "my-anthropic/claude-sonnet-4-5",
      "fallbackModels": ["my-deepseek/deepseek-chat"],
      "maxToolIterations": 40,
      "temperature": 0.1
    },
    "profiles": {
      "code_reviewer": {
        "model": "my-anthropic/claude-sonnet-4-5",
        "systemPrompt": "You are a code reviewer...",
        "allowedTools": ["read_file", "list_dir"]
      }
    }
  },
  "providers": {
    "my-anthropic": { "backend": "anthropic", "apiKey": "$ANTHROPIC_API_KEY" },
    "my-deepseek": { "backend": "openai", "baseUrl": "https://api.deepseek.com/v1", "apiKey": "$DEEPSEEK_API_KEY" },
    "my-local": { "backend": "openai", "baseUrl": "http://localhost:8080/v1", "apiKey": "none" }
  },
  "observability": {
    "otel": { "enabled": true, "endpoint": "http://localhost:4318", "serviceName": "nanobot" },
    "langfuse": { "enabled": true }
  },
  "security": {
    "promptGuard": { "enabled": true, "sensitivity": 0.7 },
    "leakDetector": { "enabled": true },
    "secrets": { "encrypt": true },
    "audit": { "enabled": true, "maxSizeMb": 100 },
    "policy": { "autoApproveTools": ["read_file", "list_dir", "web_search"], "maxActionsPerHour": 200 }
  },
  "channels": {},
  "tools": { "exec": { "enable": true, "timeout": 60 } },
  "cron": { "jobs": [] }
}
```

### 7.2 ZeroClaw Source Reference

| Area | File | Relevance |
|------|------|-----------|
| Prompt injection patterns | `security/prompt_guard.rs` | Reference (we use LLM Guard) |
| Leak detection patterns | `security/leak_detector.rs` | Reference (we use detect-secrets) |
| Audit event structure | `security/audit.rs` | Phase 1 reference |
| Emergency stop | `security/estop.rs` | Phase 3 reference |
| SOP engine | `sop/types.rs`, `sop/engine.rs` | Phase 3 reference |
| Loop detection | `agent/loop_detector.rs` | Phase 2 reference |
