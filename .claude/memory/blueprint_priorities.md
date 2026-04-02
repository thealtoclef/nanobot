---
name: Blueprint Priorities
description: Phase 0 is full PydanticAI refactor + OTEL + logging migration. Then security, core, operations. Tools LOW.
type: project
---

Phase order:
1. **Phase 0**: Foundation refactor — PydanticAI, stdlib logging, Pydantic models, simplified providers, OTEL + Langfuse, modular channels/tools
2. **Phase 1**: Security — LLM Guard (prompt injection), detect-secrets (leaks), secrets encryption, audit, policy
3. **Phase 2**: Core agent — loop detection, fallback chains, cancellation, named profiles, cron enhancements
4. **Phase 3**: Operations — estop, workspace isolation, SOP engine, Docker runtime, declarative cron, resource limits
5. **Phase 4**: Tools & integrations (LOW — prefer MCP servers)
6. **Phase 5**: Advanced features (LATER)

Full refactor in Phase 0, no backward compatibility. No feature loss.

**Why:** User wants to stand on giants (PydanticAI, OTEL, LLM Guard, detect-secrets) rather than reimplement from scratch.

**How to apply:** Phase 0 is the big refactor. Everything else builds on it.
