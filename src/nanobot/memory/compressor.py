"""Memory compression: HistoryCompressor orchestrates summarization and history compression."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from loguru import logger
from pydantic_ai.messages import ModelMessage

from nanobot.agents.helpers import format_messages_for_text
from nanobot.agents.summarizer import SummarizerAgent, SummarizerDeps, SummarizerResult
from nanobot.memory.history_store import HistoryStore
from nanobot.utils.helpers import estimate_message_tokens

if TYPE_CHECKING:
    from nanobot.agents.talker import Talker
    from nanobot.session import SessionManager


# ---------------------------------------------------------------------------
# HistoryCompressor — the main orchestrator (renamed from HistorySummarizer)
# ---------------------------------------------------------------------------


class HistoryCompressor:
    _MAX_SUMMARIZATION_ROUNDS = 5
    _SAFETY_BUFFER = 2048  # Increased per council review
    _MAX_FAILURES_BEFORE_RAW_SUMMARY = 3

    def __init__(
        self,
        db: Any,
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
        self._summarizer = SummarizerAgent()  # type: ignore[operator]

    @property
    def _model(self):
        """The agent's pydantic model, resolved at access time for testability."""
        return self.agent.pydantic_agent.model

    def get_lock(self, session_key: str) -> asyncio.Lock:
        return self._locks.setdefault(session_key, asyncio.Lock())

    def _get_history_store(self, session_key: str) -> HistoryStore:
        if session_key not in self._history_stores:
            self._history_stores[session_key] = HistoryStore(self._db, session_key)
        return self._history_stores[session_key]

    async def summarize_messages(self, session_key: str, messages: list[ModelMessage]) -> bool:
        """Summarize messages. Returns True on success (or empty input)."""
        if not messages:
            return True

        store = self._get_history_store(session_key)
        existing_summary = store.get_current_summary() or ""
        formatted = format_messages_for_text(messages)
        model = self._model

        try:
            deps = SummarizerDeps(
                existing_summary=existing_summary,
                formatted_messages=formatted,
            )
            result = await self._summarizer.run(
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

            logger.info("Summarization done for {} messages", len(messages))
            return True
        except Exception:
            logger.exception("Summarization failed")
            return self._fail_or_raw_summary(session_key, messages)

    def _fail_or_raw_summary(self, session_key: str, messages: list[ModelMessage]) -> bool:
        self._raw_summary(session_key, messages)
        return True

    def _raw_summary(self, session_key: str, messages: list[ModelMessage]) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        formatted = format_messages_for_text(messages)
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

    async def summarize(self, session_key: str, messages: list[ModelMessage]) -> bool:
        """Summarize messages into history. Returns True on success."""
        if not messages:
            return True
        return await self.summarize_messages(session_key, messages)

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
