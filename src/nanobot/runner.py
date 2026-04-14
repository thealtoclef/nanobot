"""AgentRunner: thin orchestrator around NanobotAgent.

Owns infrastructure (bus, sessions, tools, MCP) and delegates LLM
execution to NanobotAgent (pydanticAI-based).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger
from pydantic_ai.messages import ModelMessage

from nanobot.agents.talker import TalkerAgent, _to_user_content
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.builtin import register_builtin_commands
from nanobot.command.router import CommandRouter
from nanobot.context import ContextBuilder
from nanobot.db import Database
from nanobot.memory.compressor import HistoryCompressor
from nanobot.session import SessionManager
from nanobot.skill_loader import BUILTIN_SKILLS_DIR
from nanobot.tools.cron import CronTool
from nanobot.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.tools.message import MessageTool
from nanobot.tools.registry import ToolRegistry
from nanobot.tools.shell import ExecTool
from nanobot.tools.spawn import SpawnTool
from nanobot.tools.web import WebFetchTool, WebSearchTool

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        CubeConfig,
        ExecToolConfig,
        MCPServerConfig,
        MemoryConfig,
        WebToolsConfig,
    )
    from nanobot.cron.service import CronService


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

_RUNTIME_CHECKPOINT_KEY = "runtime_checkpoint"


class AgentRunner:
    """Thin orchestrator that owns infrastructure and delegates to NanobotAgent.

    This class owns everything that is NOT pydanticAI-specific:
    - MessageBus (pub/sub for channels)
    - SessionManager (conversation history)
    - ToolRegistry (nanobot's tool instances)
    - MCP server connections (stdio/SSE/HTTP)
    - Cron service integration
    - Concurrency control (semaphore + per-session locks)
    - Background task tracking

    It creates and owns a single NanobotAgent instance.
    """

    def __init__(
        self,
        workspace: Path,
        models: list[Any],
        bus: MessageBus,
        *,
        session_manager: SessionManager | None = None,
        cron_service: CronService | None = None,
        mcp_servers: dict[str, MCPServerConfig] | None = None,
        channels_config: ChannelsConfig | None = None,
        max_iterations: int = 200,
        max_tool_result_chars: int = 16000,
        context_window_tokens: int = 65536,
        web_config: WebToolsConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        restrict_to_workspace: bool = False,
        timezone: str = "UTC",
        skills_directories: list[Path] | None = None,
        memory_config: MemoryConfig | None = None,
        cube_config: CubeConfig | None = None,
        **kwargs: Any,
    ) -> None:
        from nanobot.config.schema import ExecToolConfig as ExecCfg
        from nanobot.config.schema import WebToolsConfig as WebCfg

        self.workspace = workspace
        self.models = models
        self.bus = bus
        self.channels_config = channels_config
        self.cron_service = cron_service
        self.mcp_servers = mcp_servers or {}
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.context_window_tokens = context_window_tokens
        self.web_config = web_config or WebCfg()
        self.exec_config = exec_config or ExecCfg()
        self.restrict_to_workspace = restrict_to_workspace
        self.timezone = timezone

        # Tool registry
        self.tools = ToolRegistry()

        # Database — single instance shared across all components
        self.db = Database(workspace)

        # Session manager — reuse the same Database instance
        self.sessions = session_manager or SessionManager(workspace, db=self.db)

        # Context builder (for building messages from history + current input)
        self.context = ContextBuilder(workspace, db=self.db, timezone=timezone)

        # Mem0 memory client (lazy-initialized on first use when enabled)
        self._mem0: Any = None
        if memory_config and memory_config.enabled:
            from nanobot.memory.mem0_client import Mem0Client

            self._mem0 = Mem0Client(memory_config, workspace)

        # Cube semantic layer
        self._cube_service: Any = None
        self._cube_memory: Any = None
        self._cube_schema_index: Any = None
        if cube_config and cube_config.enabled:
            from nanobot.cube.service import CubeService

            self._cube_service = CubeService(cube_config)
            # initialize() is async — deferred to first tool use (matches Mem0Client pattern)

            if cube_config.memory.enabled:
                from nanobot.cube.query_memory import QueryMemory

                persist_path = workspace / "chroma"

                self._cube_memory = QueryMemory(
                    persist_dir=persist_path,
                    max_results=cube_config.memory.max_results,
                    embedder=cube_config.memory.embedder,
                    reranker=cube_config.memory.reranker,
                )
                self._cube_memory.initialize()

                # Schema index uses the same ChromaDB directory ({workspace}/chroma/)
                # with its own embedder/reranker (can differ from memory's)
                # Set _schema_index directly before tools use it (health check still deferred to first tool use)
                if cube_config.schema_index.enabled:
                    from nanobot.cube.schema_index import CubeSchemaIndex

                    self._cube_schema_index = CubeSchemaIndex(
                        persist_dir=persist_path,
                        embedder=cube_config.schema_index.embedder,
                        reranker=cube_config.schema_index.reranker,
                    )
                    self._cube_schema_index.initialize()
                    self._cube_service._schema_index = self._cube_schema_index

        # NanobotAgent (pydanticAI-based)
        self.agent = TalkerAgent(
            workspace=workspace,
            models=models,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            context_window_tokens=context_window_tokens,
            timezone=timezone,
            skills_directories=skills_directories,
            mem0_client=self._mem0,
        )

        # Subagent manager (import here to avoid circular import)
        from nanobot.subagent import SubagentManager

        self.subagents = SubagentManager(
            db=self.db,
            agent=self.agent,
            workspace=workspace,
            bus=bus,
            max_tool_result_chars=max_tool_result_chars,
            web_config=self.web_config,
            exec_config=self.exec_config,
            restrict_to_workspace=self.restrict_to_workspace,
        )
        self.subagents.set_sessions(self.sessions)

        # History compressor
        self.history_compressor = HistoryCompressor(
            db=self.db,
            agent=self.agent,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            max_completion_tokens=4096,
            mem0_client=self._mem0,
        )

        # Background tasks tracking
        self._background_tasks: list[asyncio.Task] = []

        # MCP state (initialized lazily in _connect_mcp)
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False

        # Runtime state
        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}

        # Command router
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of nanobot tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None

        self.tools.register(
            ReadFileTool(
                workspace=self.workspace,
                allowed_dir=allowed_dir,
                extra_allowed_dirs=extra_read,
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))

        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    path_append=self.exec_config.path_append,
                )
            )

        if self.web_config.enable:
            self.tools.register(
                WebSearchTool(
                    config=self.web_config.search,
                    proxy=self.web_config.proxy,
                )
            )
            self.tools.register(WebFetchTool(proxy=self.web_config.proxy))

        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))

        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

        # Register Cube tools (always register if config enabled — lazy init in tools)
        if self._cube_service:
            from nanobot.tools.cube import CubeQueryTool, CubeSchemaTool, CubeSearchTool

            self.tools.register(CubeSchemaTool(self._cube_service))
            self.tools.register(CubeQueryTool(self._cube_service, self._cube_memory))
            if self._cube_memory and self._cube_memory.is_available:
                self.tools.register(CubeSearchTool(self._cube_memory))

        # Register tools on the NanobotAgent
        for name, tool in self.tools._tools.items():
            self.agent.tool_adapter.register(tool)

    # -------------------------------------------------------------------------
    # MCP
    # -------------------------------------------------------------------------

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (lazy, with reconnect support).

        On first call, connects to all configured servers.
        If previously connected but the stack was closed (e.g., after close_mcp),
        re-connects automatically.
        """
        if self._mcp_connecting or not self.mcp_servers:
            return
        if self._mcp_connected:
            return
        self._mcp_connecting = True
        from nanobot.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self.mcp_servers, self.tools, self._mcp_stack)

            # Register MCP tools on NanobotAgent
            for name, tool in self.tools._tools.items():
                if not self.agent.tool_adapter._tools or tool.name not in [
                    t.name for t in self.agent.tool_adapter._tools
                ]:
                    self.agent.tool_adapter.register(tool)

            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers: {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
            self._mcp_connected = False
        finally:
            self._mcp_connecting = False

    async def close_mcp(self) -> None:
        """Close MCP connections and drain background tasks."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._mcp_stack = None
        self._mcp_connected = False

    # -------------------------------------------------------------------------
    # Tool context
    # -------------------------------------------------------------------------

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
    ) -> None:
        """Update routing context on tools that need it."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """Run the agent loop: consume from bus, dispatch messages."""
        self._running = True
        await self._connect_mcp()
        logger.info("AgentRunner started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = _CommandContext(
                    msg=msg, session=None, key=msg.session_key, raw=raw, runner=self
                )
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue

            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(self._cleanup_task)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        meta = dict(msg.metadata or {})
                        meta["_stream_delta"] = True
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content=delta,
                                metadata=meta,
                            )
                        )

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        meta = dict(msg.metadata or {})
                        meta["_stream_end"] = True
                        meta["_resuming"] = resuming
                        meta["_stream_id"] = _current_stream_id()
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="",
                                metadata=meta,
                            )
                        )
                        stream_segment += 1

                response = await self._process_message(
                    msg,
                    on_stream=on_stream,
                    on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="",
                            metadata=msg.metadata or {},
                        )
                    )
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="Sorry, I encountered an error.",
                    )
                )

    async def _process_message(
        self,
        msg: InboundMessage,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        from nanobot.utils.runtime import EMPTY_FINAL_RESPONSE_MESSAGE

        # System messages: inject from subagent result
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            self.sessions.ensure_session(key)
            session = self.sessions.get_session(key)
            if self._restore_runtime_checkpoint(key):
                pass  # checkpoint restored inline
            # History summarization
            await self.history_compressor.maybe_summarize_by_tokens(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))

            unconsolidated = self.sessions.get_unconsolidated_messages(key)
            model_history, prompt_content = self.context.build_messages(
                history=unconsolidated,
                current_message=msg.content,
                channel=channel,
                chat_id=chat_id,
                session_key=key,
            )
            final_content, new_messages = await self._run_agent_loop(
                model_history,
                user_content=prompt_content,
                session=session,
                channel=channel,
                chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            # Append new messages
            if new_messages:
                self.sessions.add_messages(key, new_messages)
            self._clear_runtime_checkpoint(key)
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        self.sessions.ensure_session(msg.session_key)
        session = self.sessions.get_session(msg.session_key)
        if self._restore_runtime_checkpoint(session.key):
            pass  # checkpoint restored inline

        # Slash commands
        raw = msg.content.strip()
        ctx = _CommandContext(msg=msg, session=session, key=msg.session_key, raw=raw, runner=self)
        if result := await self.commands.dispatch(ctx):
            return result

        # History summarization
        await self.history_compressor.maybe_summarize_by_tokens(msg.session_key)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        unconsolidated = self.sessions.get_unconsolidated_messages(msg.session_key)
        model_history, prompt_content = self.context.build_messages(
            history=unconsolidated,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            session_key=session.key,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        final_content, new_messages = await self._run_agent_loop(
            model_history,
            user_content=prompt_content,
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )

        if not final_content or not final_content.strip():
            final_content = EMPTY_FINAL_RESPONSE_MESSAGE

        # Append new messages
        if new_messages:
            self.sessions.add_messages(session.key, new_messages)

        self._clear_runtime_checkpoint(session.key)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None:
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[ModelMessage],
        user_content: str | list[dict[str, Any]] = "",
        session: Any | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        deps: Any = None,
    ) -> tuple[str | None, list]:
        """Run the agent loop with optional streaming.

        Returns:
            Tuple of (final_text, new_model_messages).
        """
        from nanobot.agents.deps import AgentDeps

        await self._connect_mcp()

        model_messages = list(initial_messages)
        prompt = _to_user_content(user_content) if isinstance(user_content, list) else user_content

        # Build deps if not provided
        if deps is None:
            deps = AgentDeps(
                session_key=f"{channel}:{chat_id}",
                channel=channel,
                chat_id=chat_id,
                message_id=message_id,
                mem0_client=self._mem0,
            )

        if on_stream is not None:
            async with self.agent.run_stream(
                prompt,
                message_history=model_messages,
                deps=deps,
            ) as result:
                async for delta in result.stream_text(delta=True):
                    if delta and on_stream:
                        await on_stream(delta)
            if on_stream_end:
                await on_stream_end(resuming=False)
            output = result.response.text if result.response.text is not None else ""
            return output, result.new_messages()
        else:
            output, new_messages = await self.agent.run(
                prompt,
                message_history=model_messages,
                deps=deps,
            )
            return output, new_messages

    def _cleanup_task(self, task: asyncio.Task) -> None:
        for key, tasks in self._active_tasks.items():
            if task in tasks:
                tasks.remove(task)
                if not tasks:
                    del self._active_tasks[key]
                return

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("AgentRunner stopping")

    async def shutdown(self) -> None:
        """Ordered teardown: cancel tasks → close MCP → dispose engine."""
        self._running = False

        # 1. Cancel all active dispatch tasks
        all_tasks: list[asyncio.Task] = []
        for tasks in self._active_tasks.values():
            all_tasks.extend(tasks)
        for t in all_tasks:
            t.cancel()

        # 2. Wait for cancellation with timeout
        if all_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True),
                    timeout=10,
                )
            except TimeoutError:
                stuck = [t for t in all_tasks if not t.done()]
                if stuck:
                    logger.warning(
                        "Shutdown: {} task(s) did not cancel within 10s: {}",
                        len(stuck),
                        [id(t) for t in stuck],
                    )

        # 3. Cancel background subagent tasks
        await self.subagents.cancel_all()

        # 4. Close Cube service
        if self._cube_service:
            await self._cube_service.close()

        # 5. Drain mem0 ingest tasks
        if self.history_compressor:
            await self.history_compressor.close()

        # 6. Close MCP connections
        await self.close_mcp()

        # 7. Dispose SQLAlchemy engine
        self.db.engine.dispose()

        # 8. Drain inbound queue so messages are not silently dropped
        drained = 0
        while not self.bus.inbound.empty():
            try:
                self.bus.inbound.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.warning("Shutdown: discarded {} pending inbound message(s)", drained)

        self._active_tasks.clear()
        logger.info("AgentRunner shutdown complete")

    # -------------------------------------------------------------------------
    # process_direct: single-message entry point (for CLI, cron, heartbeat)
    # -------------------------------------------------------------------------

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single message directly (no bus)."""
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content,
            session_key=session_key,
        )
        return await self._process_message(
            msg,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )

    # -------------------------------------------------------------------------
    # Session helpers
    # -------------------------------------------------------------------------

    def _set_runtime_checkpoint(self, session_key: str, payload: dict[str, Any]) -> None:
        self._checkpoints[session_key] = payload

    def _clear_runtime_checkpoint(self, session_key: str) -> None:
        self._checkpoints.pop(session_key, None)

    @staticmethod
    def _checkpoint_message_key(msg: ModelMessage) -> tuple[Any, ...]:
        """Create a comparison key for overlap detection."""
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )

        if isinstance(msg, ModelRequest):
            parts_key = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    parts_key.append(("user", part.content))
                elif isinstance(part, SystemPromptPart):
                    parts_key.append(("system", part.content))
                elif isinstance(part, ToolReturnPart):
                    parts_key.append(
                        ("tool_return", part.tool_call_id, part.tool_name, part.content)
                    )
                else:
                    parts_key.append(("other", str(part)))
            return ("request", tuple(parts_key))
        elif isinstance(msg, ModelResponse):
            parts_key = []
            for part in msg.parts:
                if isinstance(part, TextPart):
                    parts_key.append(("text", part.content))
                elif isinstance(part, ToolCallPart):
                    parts_key.append(
                        ("tool_call", part.tool_call_id, part.tool_name, str(part.args))
                    )
                else:
                    parts_key.append(("other", str(part)))
            return ("response", tuple(parts_key))
        return ("unknown",)

    def _restore_runtime_checkpoint(self, session_key: str) -> bool:
        """Materialize an unfinished turn into session history."""
        checkpoint = self._checkpoints.get(session_key)
        if not isinstance(checkpoint, dict):
            return False

        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            ToolReturnPart,
        )

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored: list[ModelMessage] = []

        # Reconstruct assistant message as ModelResponse
        if isinstance(assistant_message, dict):
            parts = []
            if assistant_message.get("content"):
                parts.append(TextPart(content=assistant_message["content"]))
            for tc in assistant_message.get("tool_calls") or []:
                parts.append(
                    ToolCallPart(
                        tool_name=tc.get("name", tc.get("function", {}).get("name", "")),
                        tool_call_id=tc.get("id", ""),
                        args=tc.get("arguments", {}),
                    )
                )
            if parts:
                restored.append(ModelResponse(parts=parts))

        # Reconstruct completed tool results as ModelRequest with ToolReturnPart
        for result in completed_tool_results:
            if isinstance(result, dict):
                restored.append(
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name=result.get("name", ""),
                                tool_call_id=result.get("tool_call_id", ""),
                                content=result.get("content", ""),
                            )
                        ]
                    )
                )

        # Reconstruct pending tool calls as error tool returns
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=name,
                            tool_call_id=tool_id,
                            content="Error: Task interrupted before this tool finished.",
                        )
                    ]
                )
            )

        # Get current unconsolidated for overlap detection (already list[ModelMessage])
        unconsolidated = self.sessions.get_unconsolidated_messages(session_key)

        overlap = 0
        max_overlap = min(len(unconsolidated), len(restored))
        for size in range(max_overlap, 0, -1):
            existing = unconsolidated[-size:]
            restored_slice = restored[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored_slice)
            ):
                overlap = size
                break

        if restored[overlap:]:
            self.sessions.add_messages(session_key, restored[overlap:])

        self._clear_runtime_checkpoint(session_key)
        return True


# ---------------------------------------------------------------------------
# Command context (mirrors old CommandContext from loop.py)
# ---------------------------------------------------------------------------


@dataclass
class _CommandContext:
    """Lightweight command context for slash commands."""

    msg: InboundMessage
    session: Any
    key: str
    raw: str
    runner: AgentRunner

    @property
    def loop(self) -> AgentRunner:
        """Alias for runner (backward-compat with cmd handlers that use ctx.loop)."""
        return self.runner
