"""Memory system for persistent agent memory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, cast

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from nanobot.db import Database
from nanobot.utils.helpers import estimate_message_tokens, estimate_prompt_tokens

if TYPE_CHECKING:
    from nanobot.agent.agent import NanobotAgent
    from nanobot.session.manager import SessionManager


def _model_message_to_dict(msg: ModelMessage) -> dict[str, Any]:
    """Convert a PydanticAI ModelMessage to a nanobot session dict."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        UserPromptPart,
        ToolReturnPart,
        RetryPromptPart,
        TextPart,
        ToolCallPart,
        ThinkingPart,
    )

    if isinstance(msg, ModelRequest):
        parts = []
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                parts.append({"role": "system", "content": part.content})
            elif isinstance(part, UserPromptPart):
                parts.append({"role": "user", "content": part.content})
            elif isinstance(part, ToolReturnPart):
                parts.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "name": part.tool_name,
                        "content": part.content,
                    }
                )
            elif isinstance(part, RetryPromptPart):
                parts.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "name": part.tool_name or "",
                        "content": part.content,
                    }
                )
        return parts[0] if len(parts) == 1 else {"role": "request", "parts": parts}
    elif isinstance(msg, ModelResponse):
        text_content = ""
        tool_calls: list[dict[str, Any]] = []
        thinking_content = ""
        for part in msg.parts:
            if isinstance(part, TextPart):
                text_content = part.content
            elif isinstance(part, ToolCallPart):
                tool_calls.append(
                    {
                        "tool_name": part.tool_name,
                        "tool_call_id": part.tool_call_id,
                        "args": part.args,
                    }
                )
            elif isinstance(part, ThinkingPart):
                thinking_content = part.content
        return {
            "role": "assistant",
            "content": text_content,
            "tool_calls": tool_calls or None,
            "thinking": thinking_content or None,
        }
    return {"role": "unknown", "content": str(msg)}


_INITIAL_MEMORY_TEMPLATE = """# Long-term Memory

This file stores important information that should persist across sessions.

## User Information

(Important facts about the user)

## Preferences

(User preferences learned over time)

## Project Context

(Information about ongoing projects)

## Important Notes

(Things to remember)
"""


# ---------------------------------------------------------------------------
# Consolidation structured output & deps
# ---------------------------------------------------------------------------


class ConsolidationResult(BaseModel):
    history_entry: str = Field(
        description=(
            "A paragraph summarizing key events/decisions/topics. "
            "Start with a timestamp [YYYY-MM-DD HH:MM]. "
            "Include detail useful for grep search."
        ),
    )
    memory_update: str = Field(
        description=(
            "Full updated long-term memory as markdown. "
            "Include all existing facts plus new ones. "
            "Return unchanged if nothing new."
        ),
    )


@dataclass
class ConsolidationDeps:
    """Dependencies injected into the consolidation agent."""

    current_memory: str
    session_messages: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Consolidation agent (PydanticAI Agent with structured output)
# ---------------------------------------------------------------------------

_consolidation_agent = Agent(
    output_type=ConsolidationResult,
    deps_type=ConsolidationDeps,
    instructions=(
        "You are a memory consolidation agent. Analyze the conversation and produce:\n"
        "1. history_entry: A paragraph summarizing key events/decisions/topics. "
        "Start with a timestamp [YYYY-MM-DD HH:MM]. Include detail useful for grep search.\n"
        "2. memory_update: Full updated long-term memory as markdown. "
        "Include all existing facts plus new ones. Return unchanged if nothing new."
    ),
    retries=2,
)


@_consolidation_agent.instructions
def _consolidation_context(ctx: RunContext[ConsolidationDeps]) -> str:
    formatted = MemoryStore._format_messages(ctx.deps.session_messages)
    return (
        f"## Current Long-term Memory\n{ctx.deps.current_memory or '(empty)'}\n\n"
        f"## Conversation to Process\n{formatted}"
    )


class MemoryStore:
    _MAX_FAILURES_BEFORE_RAW_ARCHIVE = 3

    def __init__(self, db: Database, session_key: str):
        self._db = db
        self._session_key = session_key
        self._consecutive_failures = 0

    def read_long_term(self) -> str:
        content = self._db.get_curated_memory(self._session_key)
        return content or ""

    def write_long_term(self, content: str) -> None:
        self._db.upsert_memory(self._session_key, "curated", "curated_memory", content)

    def append_history(self, entry: str) -> None:
        self._db.append_history(self._session_key, entry)

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            tools = (
                f" [tools: {', '.join(message['tools_used'])}]" if message.get("tools_used") else ""
            )
            lines.append(
                f"[{message.get('timestamp', '?')[:16]}] {message['role'].upper()}{tools}: {message['content']}"
            )
        return "\n".join(lines)

    async def consolidate(
        self,
        messages: list[dict],
        agent: NanobotAgent,
    ) -> bool:
        """Consolidate the provided message chunk into Database-backed memory.

        Uses a PydanticAI agent with structured output (``output_type``) to
        summarize the conversation into a history entry and updated long-term
        memory, and ``deps_type`` for typed context injection.
        """
        if not messages:
            return True

        current_memory = self.read_long_term()
        # Seed template on first consolidation so the LLM has structure to fill in
        if not current_memory:
            self.write_long_term(_INITIAL_MEMORY_TEMPLATE)
            current_memory = _INITIAL_MEMORY_TEMPLATE

        deps = ConsolidationDeps(
            current_memory=current_memory,
            session_messages=messages,
        )
        model = agent.pydantic_agent.model

        try:
            result = await _consolidation_agent.run(
                user_prompt="Consolidate the conversation into long-term memory.",
                deps=deps,
                model=model,
            )

            output = cast(ConsolidationResult, result.output)

            entry = output.history_entry.strip()
            if not entry:
                logger.warning("Memory consolidation: history_entry is empty")
                return self._fail_or_raw_archive(messages)

            update = output.memory_update.strip()

            self.append_history(entry)
            if update and update != current_memory:
                self.write_long_term(update)

            self._consecutive_failures = 0
            logger.info("Memory consolidation done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return self._fail_or_raw_archive(messages)

    def _fail_or_raw_archive(self, messages: list[dict]) -> bool:
        """Increment failure count; after threshold, raw-archive messages and return True."""
        self._consecutive_failures += 1
        if self._consecutive_failures < self._MAX_FAILURES_BEFORE_RAW_ARCHIVE:
            return False
        self._raw_archive(messages)
        self._consecutive_failures = 0
        return True

    def _raw_archive(self, messages: list[dict]) -> None:
        """Fallback: dump raw messages to HISTORY.md without LLM summarization."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.append_history(
            f"[{ts}] [RAW] {len(messages)} messages\n{self._format_messages(messages)}"
        )
        logger.warning("Memory consolidation degraded: raw-archived {} messages", len(messages))


class MemoryConsolidator:
    """Owns consolidation policy, locking, and session offset updates."""

    _MAX_CONSOLIDATION_ROUNDS = 5

    _SAFETY_BUFFER = 1024  # extra headroom for tokenizer estimation drift

    def __init__(
        self,
        db: Database,
        agent: NanobotAgent,
        sessions: SessionManager,
        context_window_tokens: int,
        build_messages: Callable[..., tuple[list[dict[str, Any]], str | list[dict[str, Any]]]],
        max_completion_tokens: int = 4096,
    ):
        self._db = db
        self.agent = agent
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._build_messages = build_messages
        self._locks: dict[str, asyncio.Lock] = {}
        self._stores: dict[str, MemoryStore] = {}

    def get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the shared consolidation lock for one session."""
        return self._locks.setdefault(session_key, asyncio.Lock())

    async def consolidate_messages(
        self, session_key: str, messages: list[dict[str, object]]
    ) -> bool:
        """Archive a selected message chunk into persistent memory."""
        store = self._stores.get(session_key)
        if store is None:
            store = MemoryStore(self._db, session_key)
            self._stores[session_key] = store
        return await store.consolidate(messages, self.agent)

    def pick_consolidation_boundary(
        self,
        messages: list[ModelMessage],
        tokens_to_remove: int,
    ) -> tuple[int, int] | None:
        """Pick a user-turn boundary that removes enough old prompt tokens."""
        if not messages or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary: tuple[int, int] | None = None
        for idx, message in enumerate(messages):
            msg_dict = _model_message_to_dict(message)
            if idx > 0 and msg_dict.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += estimate_message_tokens(msg_dict)

        return last_boundary

    def estimate_session_prompt_tokens(
        self, session_key: str, all_messages: list[ModelMessage]
    ) -> int:
        """Estimate current prompt size for the normal session history view.

        Uses tiktoken directly (no provider needed).
        """
        history = [_model_message_to_dict(m) for m in all_messages]
        channel, chat_id = session_key.split(":", 1) if ":" in session_key else (None, None)
        probe_messages, _ = self._build_messages(
            history=history,
            current_message="[token-probe]",
            channel=channel,
            chat_id=chat_id,
        )
        return estimate_prompt_tokens(probe_messages, None)

    async def archive_messages(self, session_key: str, messages: list[dict[str, object]]) -> bool:
        """Archive messages with guaranteed persistence (retries until raw-dump fallback)."""
        if not messages:
            return True
        for _ in range(MemoryStore._MAX_FAILURES_BEFORE_RAW_ARCHIVE):
            if await self.consolidate_messages(session_key, messages):
                return True
        return False

    async def maybe_consolidate_by_tokens(self, session_key: str) -> None:
        """Loop: archive old messages until prompt fits within safe budget.

        The budget reserves space for completion tokens and a safety buffer
        so the LLM request never exceeds the context window.
        """
        all_messages = self.sessions.get_all_messages(session_key)
        if not all_messages or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session_key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated = self.estimate_session_prompt_tokens(session_key, all_messages)
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "Token consolidation idle {}: {}/{}",
                    session_key,
                    estimated,
                    self.context_window_tokens,
                )
                return

            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return

                boundary = self.pick_consolidation_boundary(
                    all_messages, max(1, estimated - target)
                )
                if boundary is None:
                    logger.debug(
                        "Token consolidation: no safe boundary for {} (round {})",
                        session_key,
                        round_num,
                    )
                    return

                end_idx = boundary[0]
                chunk = all_messages[:end_idx]
                if not chunk:
                    return

                # Convert ModelMessage chunk to session dicts for archiving
                from nanobot.agent.agent import model_messages_to_session_messages

                chunk_dicts = model_messages_to_session_messages(chunk)

                logger.info(
                    "Token consolidation round {} for {}: {}/{}, chunk={} msgs",
                    round_num,
                    session_key,
                    estimated,
                    self.context_window_tokens,
                    len(chunk),
                )
                if not await self.consolidate_messages(session_key, chunk_dicts):
                    return

                # Update consolidation boundary using message row id
                boundary_row_id = self.sessions.get_boundary_message_id(session_key, end_idx)
                if boundary_row_id is not None:
                    self.sessions.update_last_consolidated_message_id(session_key, boundary_row_id)

                # Re-fetch messages after boundary update
                all_messages = self.sessions.get_all_messages(session_key)
                estimated = self.estimate_session_prompt_tokens(session_key, all_messages)
                if estimated <= 0:
                    return
