---
name: No Prometheus, No Logfire, No loguru
description: Use only OTEL for infra observability, stdlib logging for logs, Langfuse for LLM. No Prometheus, no Logfire, no loguru.
type: feedback
---

- **No Prometheus** — OTEL covers all metrics. If Prometheus scraping needed, use OTEL Collector's exporter.
- **No Logfire** — Closed-source SaaS. PydanticAI's `Agent.instrument_all()` works with pure OTEL, no Logfire needed.
- **No loguru** — No OTEL support, confirmed by OTEL maintainers. Use stdlib `logging` + `LoggingInstrumentor`.

**Why:** User wants standard, industry-grade tooling. One telemetry pipeline (OTEL). loguru is the worst choice for OTEL integration.

**How to apply:** Never add loguru, prometheus-client, or logfire as dependencies. Use `logging.getLogger(__name__)` for all logging. Use OTEL meter for all metrics.
