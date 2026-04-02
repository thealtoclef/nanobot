---
name: PydanticAI Full Refactor
description: Full refactor to PydanticAI as core agent framework. No backward compatibility. No feature loss. Replaces AgentRunner with agent.iter().
type: project
---

**Decision (revised):** Adopt PydanticAI as the core agent framework. Full refactor, no incremental/backward-compatible approach.

PydanticAI wraps official OpenAI/Anthropic SDKs internally. `agent.iter()` provides full control for hooks, security, observability. Native OTEL spans via `Agent.instrument_all()`.

Key mappings:
- `AgentRunner.run()` → `agent.iter()` loop with node interception
- Hook system → before/after each `agent_run.next(node)` call
- Context snipping → pre-`ModelRequestNode` message trimming
- Memory consolidation → separate PydanticAI Agent with save_memory tool
- Subagent spawning → PydanticAI Agent with `FilteredToolset`

Also use Pydantic `BaseModel` everywhere — no dataclasses.

**Why:** Stand on shoulders of giants. PydanticAI handles tool schemas, retry, streaming, structured output, OTEL instrumentation, approval workflows.

**How to apply:** When writing new agent code, use PydanticAI patterns (Agent, RunContext, FunctionToolset, iter()). When porting existing code, preserve all features listed in BLUEPRINT.md Section 2.1.
