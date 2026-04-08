"""AgentRunner: thin orchestrator around NanobotAgent.

Owns infrastructure (bus, sessions, tools, MCP) and delegates LLM
execution to NanobotAgent (pydanticAI-based).
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import AsyncExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.agent import NanobotAgent
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command.router import CommandRouter
from nanobot.command.builtin import register_builtin_commands
from nanobot.db import Database
from nanobot.session.manager import SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        ExecToolConfig,
        MCPServerConfig,
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
        **kwargs: Any,
    ) -> None:
        from nanobot.config.schema import ExecToolConfig as ExecCfg, WebToolsConfig as WebCfg

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

        # NanobotAgent (pydanticAI-based)
        self.agent = NanobotAgent(
            workspace=workspace,
            models=models,
            max_iterations=max_iterations,
            max_tool_result_chars=max_tool_result_chars,
            context_window_tokens=context_window_tokens,
            timezone=timezone,
            skills_directories=skills_directories,
        )

        # Subagent manager (import here to avoid circular import)
        from nanobot.agent.subagent import SubagentManager

        self.subagents = SubagentManager(
            agent=self.agent,
            workspace=workspace,
            bus=bus,
            max_tool_result_chars=max_tool_result_chars,
            web_config=self.web_config,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        # Memory consolidator
        from nanobot.agent.memory import MemoryConsolidator

        self.memory_consolidator = MemoryConsolidator(
            db=self.db,
            agent=self.agent,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            max_completion_tokens=4096,
        )

        # Lifecycle state
        self._running = False
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )

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
        from nanobot.agent.tools.mcp import connect_mcp_servers

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
        from nanobot.utils.helpers import image_placeholder_text, truncate_text

        # System messages: inject from subagent result
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            if self._restore_runtime_checkpoint(session):
                self.sessions.save(session)
            # Memory consolidation
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages, raw_user_content = self.context.build_messages(
                history=history,
                current_message=msg.content,
                channel=channel,
                chat_id=chat_id,
                current_role=current_role,
                session_key=session.key,
            )
            final_content, new_messages = await self._run_agent_loop(
                messages,
                user_content=raw_user_content,
                session=session,
                channel=channel,
                chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, new_messages)
            self._clear_runtime_checkpoint(session)
            self.sessions.save(session)
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        session = self.sessions.get_or_create(msg.session_key)
        if self._restore_runtime_checkpoint(session):
            self.sessions.save(session)

        # Slash commands
        raw = msg.content.strip()
        ctx = _CommandContext(msg=msg, session=session, key=msg.session_key, raw=raw, runner=self)
        if result := await self.commands.dispatch(ctx):
            return result

        # Memory consolidation
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages, raw_user_content = self.context.build_messages(
            history=history,
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
            initial_messages,
            user_content=raw_user_content,
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

        self._save_turn(session, new_messages)
        self._clear_runtime_checkpoint(session)
        self.sessions.save(session)

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
        initial_messages: list[dict[str, Any]],
        user_content: str | list[dict[str, Any]] = "",
        session: Any | None = None,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list]:
        """Run the agent loop with optional streaming.

        Returns:
            Tuple of (final_text, new_model_messages).
        """
        from nanobot.agent.agent import session_messages_to_model_messages

        await self._connect_mcp()

        model_messages = session_messages_to_model_messages(initial_messages)

        if on_stream is not None:
            async with self.agent.run_stream(
                user_content,
                message_history=model_messages,
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
                user_content,
                message_history=model_messages,
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

        # 4. Close MCP connections
        await self.close_mcp()

        # 5. Dispose SQLAlchemy engine
        self.db.engine.dispose()

        # 6. Drain inbound queue so messages are not silently dropped
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

    def _save_turn(self, session: Any, new_model_messages: list) -> None:
        from nanobot.agent.agent import persist_new_messages

        persist_new_messages(session, new_model_messages, self.max_tool_result_chars)

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        from nanobot.agent.agent import _sanitize_persisted_blocks as _shared

        return _shared(
            content,
            self.max_tool_result_chars,
            truncate_text=truncate_text,
            drop_runtime=drop_runtime,
        )

    def _set_runtime_checkpoint(self, session: Any, payload: dict[str, Any]) -> None:
        session.metadata[_RUNTIME_CHECKPOINT_KEY] = payload
        self.sessions.save(session)

    def _clear_runtime_checkpoint(self, session: Any) -> None:
        if _RUNTIME_CHECKPOINT_KEY in session.metadata:
            session.metadata.pop(_RUNTIME_CHECKPOINT_KEY, None)

    @staticmethod
    def _checkpoint_message_key(message: dict[str, Any]) -> tuple[Any, ...]:
        return (
            message.get("role"),
            message.get("content"),
            message.get("tool_call_id"),
            message.get("name"),
            message.get("tool_calls"),
            message.get("reasoning_content"),
            message.get("thinking_blocks"),
        )

    def _restore_runtime_checkpoint(self, session: Any) -> bool:
        """Materialize an unfinished turn into session history."""
        from datetime import datetime

        checkpoint = session.metadata.get(_RUNTIME_CHECKPOINT_KEY)
        if not isinstance(checkpoint, dict):
            return False

        assistant_message = checkpoint.get("assistant_message")
        completed_tool_results = checkpoint.get("completed_tool_results") or []
        pending_tool_calls = checkpoint.get("pending_tool_calls") or []

        restored_messages: list[dict[str, Any]] = []
        if isinstance(assistant_message, dict):
            restored = dict(assistant_message)
            restored.setdefault("timestamp", datetime.now().isoformat())
            restored_messages.append(restored)
        for message in completed_tool_results:
            if isinstance(message, dict):
                restored = dict(message)
                restored.setdefault("timestamp", datetime.now().isoformat())
                restored_messages.append(restored)
        for tool_call in pending_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_id = tool_call.get("id")
            name = ((tool_call.get("function") or {}).get("name")) or "tool"
            restored_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": name,
                    "content": "Error: Task interrupted before this tool finished.",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        overlap = 0
        max_overlap = min(len(session.messages), len(restored_messages))
        for size in range(max_overlap, 0, -1):
            existing = session.messages[-size:]
            restored = restored_messages[:size]
            if all(
                self._checkpoint_message_key(left) == self._checkpoint_message_key(right)
                for left, right in zip(existing, restored)
            ):
                overlap = size
                break
        session.messages.extend(restored_messages[overlap:])
        self._clear_runtime_checkpoint(session)
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
