# Roadmap

---

## 🔴 Critical Fixes

- [ ] **C1. Tool call history lost** — PydanticAI `result.new_messages()` not persisted. Only `result.output` (final text) is saved. Multi-turn tool usage is broken. *(agent/agent.py, agent/runner.py)*
- [ ] **C2. Duplicate user messages** — Two paths write user messages to session: `NanobotAgent.run()` and `_save_turn()`. Merged into C1 fix.
- [ ] **C3. tool_calls stored as Python repr, not JSON** — SQLAlchemy `Text` column stores `str(list)`. Read-back iterates characters, not dicts. *(db.py, session/manager.py)*
- [ ] **C4. Multiple Database instances for same SQLite file** — `SessionManager`, `AgentRunner`, and `Nanobot` each create their own engine. Risk of `SQLITE_BUSY`. *(runner.py, nanobot.py)*
- [ ] **C5. Agent run timeout missing** — Use PydanticAI's native `ModelSettings.timeout` and `tool_timeout`. No `asyncio.wait_for()` needed.
- [ ] **C6. Double system prompt** — `ContextBuilder.build_messages()` prepends system message AND `Agent(instructions=...)` adds another. Use dynamic `instructions` callable, remove from `build_messages()`. *(agent/agent.py, agent/context.py)*
- [ ] **C7. `_save_turn` operates on pre-run messages** — Iterates `initial_messages[skip:]` which never includes tool calls/results from the agent run. Merged into C1 fix.
- [ ] **C16. `Nanobot.run()` never saves session** — SDK facade silently loses all messages. *(nanobot.py)*
- [ ] **C17. `MemoryStore` consecutive failures never accumulate** — Fresh instantiation resets `_consecutive_failures` to 0. Raw-archive fallback can never trigger. *(agent/memory.py)*

## 🟠 Important Fixes

- [ ] **C8. No graceful shutdown** — `stop()` only sets a flag. Need ordered teardown: cancel tasks → close MCP → dispose engine.
- [ ] **C9. Subagent reuses main agent's identity and tools** — Create separate `NanobotAgent` with restricted instructions and filtered tool set (no `spawn`, `message`, `cron`).
- [ ] **C10. `_active_tasks` memory leak** — Empty session-key lists remain in dict forever.
- [ ] **C11. `archive_messages` always returns True** — Silent data loss when consolidation fails.
- [ ] **C12. Observability warning when disabled** — `logger.warning` → `logger.info`.
- [ ] **C13. Config default API key placeholder** — Remove `api_key="sk-example"` default.
- [ ] **C14. No MCP reconnection** — Dead MCP subprocess = permanently broken tools.
- [ ] **C15. In-memory MessageBus loses messages on crash** — Document limitation, flush on shutdown.

## 🔵 PydanticAI Native Refactoring Decisions

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Tool base class + registry + adapter | ❌ Eliminate | `FunctionToolset[ToolDeps]` + `@toolset.tool` |
| MCP tool wrapper | ❌ Eliminate | `MCPServerStdio` / `MCPServerStreamableHTTP` (not FastMCPToolset — lacks sampling/elicitation) |
| Web search/fetch tools | ✅ Keep custom | SSRF protection, proxy, fallback chains > native capabilities |
| Subagent pattern | ✅ Keep fire-and-forget | PydanticAI's sync delegation doesn't fit async channel communication |
| Guardrails | 🔜 Use `Hooks` | `before_tool_execute`, `after_model_request`, etc. |
| Cost tracking | ❌ Don't build | Langfuse gives this for free |
| Observability | 🔄 Langfuse | LLM-native, self-hostable, replaces logfire for LLM traces |
| Skills | ✅ Already native | `SkillsCapability` from pydantic-ai-skills |

## Feature Roadmap

### Phase 0 — Foundation Hardening
- [x] PydanticAI agent loop
- [x] SQLAlchemy + Alembic SQLite persistence
- [x] Config-driven model/provider system
- [x] Streaming support
- [x] Subagent spawning
- [x] Token-based memory consolidation
- [x] Skills system
- [ ] Fix all critical + important bugs (C1–C17)
- [ ] Integration tests using `TestModel`/`FunctionModel` + `capture_run_messages()`

### Phase 0.1 — Production Hardening
- [ ] Fix `_active_tasks` cleanup lambda → proper `_cleanup_task()` method (runner.py)
- [ ] Fix user message extraction — pass separately instead of popping from history (runner.py)
- [ ] Make `save_all_messages()` transactional — wrap delete+insert in explicit transaction (db.py)
- [ ] Extract shared `persist_new_messages()` — eliminate duplication between `_save_turn()` and `Nanobot.run()`
- [ ] Add error handling to `Nanobot.run()` — save partial state on exceptions
- [ ] Fix `MemoryConsolidator._stores` unbounded growth — use `WeakValueDictionary`
- [ ] Complete `model_messages_to_session_messages()` — handle `RetryPromptPart`, `tool_results`, multimodal content

### Phase 0.5 — PydanticAI Native Refactoring
- [ ] Migrate tools to `FunctionToolset[ToolDeps]`
- [ ] Migrate MCP to native toolsets
- [ ] Fix subagent isolation
- [ ] Prompt template migration (jinja2)

### Phase 1 — Guardrails via Hooks
### Phase 2 — Memory with ChromaDB
### Phase 2.5 — Persistent Message Queue
- [ ] Replace `asyncio.Queue` in `MessageBus` with SQLite-backed persistent queue ([persist-queue](https://github.com/peter-wangxu/persist-queue)) — ensures message durability across crashes/restarts

### Phase 3 — Observability via Langfuse
### Phase 4 — Approval System for Dangerous Operations
### Phase 5 — Multi-Agent Orchestration

## Architecture Comparison

| | nanobot | ShibaClaw | Hermes | ZeroClaw |
|---|---|---|---|---|
| Framework | PydanticAI | DIY loop | DIY (OpenAI-compat) | DIY (Rust) |
| Providers | 2 | 15+ | 10+ | 30+ |
| Channels | 3 | 11 | 16 | 20+ |
| Memory | SQLite → ChromaDB | SQLite + proactive | FTS5 + Honcho | Vector RAG |
| Observability | Langfuse | None | Trajectory saving | Built-in |
| DB | SQLAlchemy + Alembic | JSON files | SQLite FTS5 | Built-in |

## Top Risks

1. **C1+C7 fix complexity** — Message round-trip conversion between PydanticAI and session dicts. Mitigate with integration test first.
2. **Phase 0.5a cascading failures** — Tool migration touches everything. Mitigate with complete self-contained PR.
3. **No tests = blind refactoring** — Mitigate with Phase 0 test milestone using PydanticAI's `TestModel`/`FunctionModel`.
