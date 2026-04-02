---
name: Modular Architecture
description: Providers/channels/tools are optional extras. Base classes auto-instrument tracing/metrics/logging. Extensions only implement business logic.
type: project
---

**Package structure:**
- `nanobot` core: pydantic-ai-slim, OTEL, stdlib logging, CLI. No provider SDKs, no channels.
- `nanobot[openai]` / `nanobot[anthropic]`: provider SDKs as optional extras. Skip gemini for now.
- `nanobot[telegram]` / `nanobot[slack]` etc: channel SDKs as optional extras.
- `nanobot[security]`: LLM Guard + detect-secrets + cryptography.

**Instrumented base classes:**
- `BaseChannel`: `start()`, `handle_message()`, `send()` auto-create OTEL spans, increment metrics counters, log with trace correlation. Subclasses implement `_start()`, `_handle_message()`, `_send()` — no observability code.
- `InstrumentedToolset`: wraps PydanticAI FunctionToolset with metrics/logging. Tool authors write plain functions.
- Same pattern for cron, session, memory services.

**Why:** User wants a clean core that's easily extendable. No observability boilerplate in plugins.

**How to apply:** When creating new channels/tools/services, always extend the instrumented base class. Never add OTEL/metrics/logging code in subclasses.
