---
name: Observability Architecture
description: PydanticAI native OTEL spans + Langfuse as OTEL backend + stdlib logging with OTEL handler. No Logfire, no loguru, no Prometheus.
type: project
---

Three-layer observability stack:

1. **PydanticAI native OTEL** — `Agent.instrument_all()` emits spans for agent runs, model requests, tool executions. Standard OTEL GenAI semantic conventions.
2. **Langfuse as OTEL backend** — Langfuse v3+ receives OTEL spans via `langfuse.get_client()`. Shows prompts, completions, tokens, cost. Linked by trace_id automatically.
3. **stdlib logging + OTEL** — Python `logging` module with two handlers: `StreamHandler` (stderr) + `LoggingHandler` (OTEL export). `LoggingInstrumentor` auto-injects trace_id/span_id.

**Why:** PydanticAI natively emits OTEL. Langfuse is OTEL-compatible. stdlib logging is the only first-class OTEL logging path. No custom cost tracker — Langfuse handles LLM costs.

**How to apply:** Use `Agent.instrument_all()`, never Logfire. Use stdlib `logging.getLogger()`, never loguru. Configure via env vars (OTEL_*, LANGFUSE_*).
