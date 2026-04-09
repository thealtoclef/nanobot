"""Memory system: Summarizer agent + Extractor agent for history and facts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from nanobot.db import Database
from nanobot.utils.helpers import estimate_message_tokens

if TYPE_CHECKING:
    from nanobot.agent.agent import Talker
    from nanobot.session.manager import SessionManager


# ---------------------------------------------------------------------------
# Format helpers — NO dict conversion, direct ModelMessage → text
# ---------------------------------------------------------------------------


def format_messages_for_summarizer(messages: list[ModelMessage]) -> str:
    """Format ModelMessages into readable text for the summarizer/extractor prompt."""
    from pydantic_ai.messages import (
        UserPromptPart,
        SystemPromptPart,
        ToolReturnPart,
        RetryPromptPart,
        TextPart,
        ToolCallPart,
    )

    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content if isinstance(c, str))
                    lines.append(f"USER: {content}")
                elif isinstance(part, SystemPromptPart):
                    lines.append(f"SYSTEM: {part.content}")
                elif isinstance(part, ToolReturnPart):
                    lines.append(f"TOOL [{part.tool_name}]: {part.content}")
                elif isinstance(part, RetryPromptPart):
                    lines.append(f"TOOL [{part.tool_name or 'unknown'}]: {part.content}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f"ASSISTANT: {part.content}")
                elif isinstance(part, ToolCallPart):
                    lines.append(f"ASSISTANT [call {part.tool_name}]: {part.args}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarizer agent
# ---------------------------------------------------------------------------


class SummarizerResult(BaseModel):
    summary: str = Field(
        description=(
            "A comprehensive recursive summary. "
            "If there is an existing summary, incorporate it with the new conversation. "
            "Start with a timestamp [YYYY-MM-DD HH:MM]. "
            "Include enough detail for context recovery. "
            "If the existing summary contains errors or omissions, correct them."
        ),
    )


@dataclass
class SummarizerDeps:
    existing_summary: str
    formatted_messages: str


_summarizer_agent = Agent(
    output_type=SummarizerResult,
    deps_type=SummarizerDeps,
    instructions=(
        "You are a conversation summarizer. Analyze the conversation and produce a comprehensive summary.\n"
        "If there is an existing summary, incorporate it with the new conversation.\n"
        "Start with a timestamp [YYYY-MM-DD HH:MM].\n"
        "Include enough detail for context recovery.\n"
        "If the existing summary contains errors or omissions, correct them in your new summary.\n"
        "Be concise but thorough. Focus on decisions, topics, and key information."
    ),
    retries=2,
)


@_summarizer_agent.instructions
def _summarizer_context(ctx: RunContext[SummarizerDeps]) -> str:
    parts = []
    if ctx.deps.existing_summary:
        parts.append(f"## Existing Summary\n{ctx.deps.existing_summary}")
    parts.append(f"## Conversation to Summarize\n{ctx.deps.formatted_messages}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Extractor agent
# ---------------------------------------------------------------------------


class FactItem(BaseModel):
    content: str = Field(description="A single factual statement.")
    category: str = Field(description="Either 'fact' or 'preference'.")


class ExtractorResult(BaseModel):
    facts: list[FactItem] = Field(
        description="List of facts extracted from the conversation.",
        default_factory=list,
    )


@dataclass
class ExtractorDeps:
    formatted_messages: str
    existing_facts: str


_extractor_agent = Agent(
    output_type=ExtractorResult,
    deps_type=ExtractorDeps,
    instructions=(
        "You are a fact extractor. Analyze the conversation and extract individual facts.\n"
        "For each fact, provide:\n"
        "- content: A clear, self-contained factual statement\n"
        "- category: Either 'fact' (objective information) or 'preference' (user preferences)\n\n"
        "Rules:\n"
        "- Extract ONLY genuinely new facts not already in the existing facts list\n"
        "- Skip trivial observations (greetings, confirmations)\n"
        "- Each fact should be atomic and independently useful\n"
        "- If no new facts are found, return an empty list"
    ),
    retries=1,
)


@_extractor_agent.instructions
def _extractor_context(ctx: RunContext[ExtractorDeps]) -> str:
    parts = [f"## Conversation\n{ctx.deps.formatted_messages}"]
    if ctx.deps.existing_facts:
        parts.append(f"## Existing Facts (do NOT duplicate these)\n{ctx.deps.existing_facts}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# HistoryStore — writes to the histories table
# ---------------------------------------------------------------------------


class HistoryStore:
    def __init__(self, db: Database, session_key: str):
        self._db = db
        self._session_key = session_key

    def add(self, summary: str, summarized_through_message_id: int | None) -> int:
        return self._db.add_history(self._session_key, summary, summarized_through_message_id)

    def get_current_summary(self) -> str | None:
        return self._db.get_latest_history_summary(self._session_key)


# ---------------------------------------------------------------------------
# FactStore — writes to the facts table
# ---------------------------------------------------------------------------


class FactStore:
    def __init__(self, db: Database, session_key: str):
        self._db = db
        self._session_key = session_key

    def add(self, content: str, category: str) -> int:
        return self._db.add_fact(self._session_key, content, category)

    def add_many(self, facts: list[tuple[str, str]]) -> None:
        self._db.add_facts(self._session_key, facts)

    def get_digest(self, max_tokens: int = 500) -> str:
        return self._db.get_fact_digest(self._session_key, max_tokens)

    def get_existing_facts_text(self) -> str:
        facts = self._db.get_facts(self._session_key)
        if not facts:
            return ""
        return "\n".join(f"- [{f.category}] {f.content}" for f in facts)


# ---------------------------------------------------------------------------
# HistorySummarizer — the main orchestrator (replaces MemoryConsolidator)
# ---------------------------------------------------------------------------


class HistorySummarizer:
    _MAX_SUMMARIZATION_ROUNDS = 5
    _SAFETY_BUFFER = 2048  # Increased per council review
    _MAX_FAILURES_BEFORE_RAW_SUMMARY = 3

    def __init__(
        self,
        db: Database,
        agent: Talker,
        sessions: SessionManager,
        context_window_tokens: int,
        max_completion_tokens: int = 4096,
    ):
        self._db = db
        self.agent = agent
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._locks: dict[str, asyncio.Lock] = {}
        self._history_stores: dict[str, HistoryStore] = {}
        self._fact_stores: dict[str, FactStore] = {}
        self._consecutive_failures: dict[str, int] = {}

    def get_lock(self, session_key: str) -> asyncio.Lock:
        return self._locks.setdefault(session_key, asyncio.Lock())

    def _get_history_store(self, session_key: str) -> HistoryStore:
        if session_key not in self._history_stores:
            self._history_stores[session_key] = HistoryStore(self._db, session_key)
        return self._history_stores[session_key]

    def _get_fact_store(self, session_key: str) -> FactStore:
        if session_key not in self._fact_stores:
            self._fact_stores[session_key] = FactStore(self._db, session_key)
        return self._fact_stores[session_key]

    async def summarize_messages(self, session_key: str, messages: list[ModelMessage]) -> bool:
        """Summarize messages. Returns True on success (or empty input)."""
        if not messages:
            return True

        store = self._get_history_store(session_key)
        existing_summary = store.get_current_summary() or ""
        formatted = format_messages_for_summarizer(messages)
        model = self.agent.pydantic_agent.model

        try:
            deps = SummarizerDeps(
                existing_summary=existing_summary,
                formatted_messages=formatted,
            )
            result = await _summarizer_agent.run(
                user_prompt="Summarize the conversation into a comprehensive summary.",
                deps=deps,
                model=model,
            )
            output = cast(SummarizerResult, result.output)
            summary = output.summary.strip()

            if not summary:
                logger.warning("Summarizer: summary is empty")
                return self._fail_or_raw_summary(session_key, messages)

            blobs = self.sessions.get_unconsolidated_blobs_with_ids(session_key)
            boundary_row_id = blobs[-1][0] if blobs else None

            history_id = store.add(summary, boundary_row_id)
            self.sessions.update_current_history_id(session_key, history_id)

            self._consecutive_failures[session_key] = 0
            logger.info("Summarization done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Summarization failed")
            return self._fail_or_raw_summary(session_key, messages)

    async def extract_facts(self, session_key: str, messages: list[ModelMessage]) -> None:
        """Extract facts. Best-effort, non-blocking. Never raises."""
        if not messages:
            return
        try:
            fact_store = self._get_fact_store(session_key)
            formatted = format_messages_for_summarizer(messages)
            existing_facts = fact_store.get_existing_facts_text()
            deps = ExtractorDeps(formatted_messages=formatted, existing_facts=existing_facts)
            model = self.agent.pydantic_agent.model

            result = await _extractor_agent.run(
                user_prompt="Extract facts from the conversation.",
                deps=deps,
                model=model,
            )
            output = cast(ExtractorResult, result.output)
            if output.facts:
                fact_tuples = [(f.content, f.category) for f in output.facts]
                fact_store.add_many(fact_tuples)
                logger.info("Extracted {} facts", len(fact_tuples))
        except Exception:
            logger.warning("Fact extraction failed (best-effort, continuing)", exc_info=True)

    def _fail_or_raw_summary(self, session_key: str, messages: list[ModelMessage]) -> bool:
        failures = self._consecutive_failures.get(session_key, 0) + 1
        self._consecutive_failures[session_key] = failures
        if failures < self._MAX_FAILURES_BEFORE_RAW_SUMMARY:
            return False
        self._raw_summary(session_key, messages)
        self._consecutive_failures[session_key] = 0
        return True

    def _raw_summary(self, session_key: str, messages: list[ModelMessage]) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        formatted = format_messages_for_summarizer(messages)
        summary = f"[{ts}] [RAW] {len(messages)} messages\n{formatted}"

        blobs = self.sessions.get_unconsolidated_blobs_with_ids(session_key)
        boundary_row_id = blobs[-1][0] if blobs else None

        store = self._get_history_store(session_key)
        history_id = store.add(summary, boundary_row_id)
        self.sessions.update_current_history_id(session_key, history_id)
        logger.warning("Summarization degraded: raw-summarized {} messages", len(messages))

    def pick_summarization_boundary(
        self,
        blobs: list[tuple[int, list[ModelMessage]]],
        tokens_to_remove: int,
    ) -> int | None:
        """Pick blob-level boundary. Returns row_id or None."""
        if not blobs or tokens_to_remove <= 0:
            return None
        removed_tokens = 0
        for row_id, messages in blobs:
            for msg in messages:
                removed_tokens += estimate_message_tokens(msg)
            if removed_tokens >= tokens_to_remove:
                return row_id
        return None

    def estimate_session_prompt_tokens(self, messages: list[ModelMessage]) -> int:
        """Simple sum of per-message token estimates."""
        return sum(estimate_message_tokens(msg) for msg in messages)

    async def summarize_and_extract(self, session_key: str, messages: list[ModelMessage]) -> bool:
        """Combined: summarize + extract facts."""
        if not messages:
            return True
        for _ in range(self._MAX_FAILURES_BEFORE_RAW_SUMMARY):
            if await self.summarize_messages(session_key, messages):
                await self.extract_facts(session_key, messages)
                return True
        await self.extract_facts(session_key, messages)
        return False

    async def maybe_summarize_by_tokens(self, session_key: str) -> None:
        """Loop: summarize old blobs until prompt fits within budget."""
        unconsolidated = self.sessions.get_unconsolidated_messages(session_key)
        if not unconsolidated or self.context_window_tokens <= 0:
            return

        lock = self.get_lock(session_key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER
            target = budget // 2
            estimated = self.estimate_session_prompt_tokens(unconsolidated)
            if estimated <= 0:
                return
            if estimated < budget:
                logger.debug(
                    "Token summarization idle {}: {}/{}",
                    session_key,
                    estimated,
                    self.context_window_tokens,
                )
                return

            for round_num in range(self._MAX_SUMMARIZATION_ROUNDS):
                if estimated <= target:
                    return

                blobs = self.sessions.get_unconsolidated_blobs_with_ids(session_key)
                boundary_row_id = self.pick_summarization_boundary(
                    blobs, max(1, estimated - target)
                )
                if boundary_row_id is None:
                    logger.debug(
                        "Token summarization: no boundary for {} (round {})", session_key, round_num
                    )
                    return

                chunk: list[ModelMessage] = []
                for row_id, msgs in blobs:
                    chunk.extend(msgs)
                    if row_id == boundary_row_id:
                        break

                if not chunk:
                    return

                logger.info(
                    "Token summarization round {} for {}: {}/{}, chunk={} msgs",
                    round_num,
                    session_key,
                    estimated,
                    self.context_window_tokens,
                    len(chunk),
                )
                if not await self.summarize_messages(session_key, chunk):
                    return

                unconsolidated = self.sessions.get_unconsolidated_messages(session_key)
                if not unconsolidated:
                    return
                estimated = self.estimate_session_prompt_tokens(unconsolidated)
                if estimated <= 0:
                    return


# ---------------------------------------------------------------------------
# Backward-compatibility aliases — these are stubs to keep context.py / runner.py
# importing while the rest of the system transitions to the new API.
# ---------------------------------------------------------------------------


# Kept for import compatibility only; context.py uses get_memory_context()
class MemoryStore:
    """Compatibility stub — use HistoryStore or FactStore directly."""

    def __init__(self, db: Database, session_key: str):
        self._db = db
        self._session_key = session_key

    def get_memory_context(self) -> str:
        """Return formatted facts as context (new API) or empty str."""
        store = FactStore(self._db, self._session_key)
        digest = store.get_digest()
        return f"## Long-term Memory\n{digest}" if digest else ""


# Kept for import compatibility only; runner.py uses build_messages parameter.
# HistorySummarizer does NOT accept build_messages — runner.py must be updated.
MemoryConsolidator = HistorySummarizer  # type: ignore[misc]
