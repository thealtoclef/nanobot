"""Summarizer agent tests using PydanticAI's TestModel.

Exercises the real ``_summarizer_agent`` with
``agent.override(model=TestModel(...))`` — no patching of ``agent.run``.
Validates that ``output_type=SummarizerResult`` produces correct
structured output, ``deps_type`` is injected into instructions,
and the summarization flow works end-to-end with a deterministic
model substitute.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)
from pydantic_ai.models.test import TestModel

from nanobot.memory import (
    SummarizerDeps,
    SummarizerResult,
    HistoryStore,
)
from nanobot.memory.compressor import HistoryCompressor as HistorySummarizer
from nanobot.agents.summarizer import _summarizer_agent
from nanobot.db import Database, upgrade_db


def _make_model_messages(count: int = 5) -> list[ModelMessage]:
    """Create ModelMessage list for testing."""
    messages = []
    for i in range(count):
        if i % 2 == 0:
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"message {i}")]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=f"message {i}")]))
    return messages


def _summarizer_test_model(summary: str = "[2026-04-09 10:00] Test summary.") -> TestModel:
    return TestModel(
        custom_output_args={"summary": summary},
    )


_EMPTY_SUMMARIZER_DEPS = SummarizerDeps(existing_summary="", formatted_messages="")


@pytest.fixture
def db(tmp_path: Path) -> Database:
    from nanobot.db import Base

    upgrade_db(tmp_path)
    db = Database(tmp_path)
    Base.metadata.create_all(db.engine)
    db.ensure_session("session:test")
    return db


@pytest.fixture
def mock_agent() -> MagicMock:
    mock = MagicMock()
    mock.pydantic_agent.model = None
    return mock


# ---------------------------------------------------------------------------
# TestSummarizerResultOutput
# ---------------------------------------------------------------------------


class TestSummarizerResultOutput:
    """Verify TestModel produces SummarizerResult through the agent."""

    @pytest.mark.asyncio
    async def test_returns_summarizer_result_type(self) -> None:
        tm = _summarizer_test_model()
        with _summarizer_agent.override(model=tm):
            result = await _summarizer_agent.run(
                user_prompt="Summarize the conversation.", model=tm, deps=_EMPTY_SUMMARIZER_DEPS
            )
        assert isinstance(result.output, SummarizerResult)

    @pytest.mark.asyncio
    async def test_summary_field_parsed(self) -> None:
        expected = "[2026-04-09 12:00] User discussed new features."
        tm = _summarizer_test_model(summary=expected)
        with _summarizer_agent.override(model=tm):
            result = await _summarizer_agent.run(
                user_prompt="Summarize.", model=tm, deps=_EMPTY_SUMMARIZER_DEPS
            )
        assert result.output.summary == expected

    @pytest.mark.asyncio
    async def test_empty_summary(self) -> None:
        tm = _summarizer_test_model(summary="   ")
        with _summarizer_agent.override(model=tm):
            result = await _summarizer_agent.run(
                user_prompt="Summarize.", model=tm, deps=_EMPTY_SUMMARIZER_DEPS
            )
        assert result.output.summary.strip() == ""

    @pytest.mark.asyncio
    async def test_timestamp_prefix(self) -> None:
        """Summary should start with a timestamp."""
        tm = _summarizer_test_model(summary="[2026-04-09 14:00] Discussion of tests.")
        with _summarizer_agent.override(model=tm):
            result = await _summarizer_agent.run(
                user_prompt="Summarize.", model=tm, deps=_EMPTY_SUMMARIZER_DEPS
            )
        assert result.output.summary.startswith("[2026-04-09")


# ---------------------------------------------------------------------------
# TestSummarizationFlowWithTestModel
# ---------------------------------------------------------------------------


class TestSummarizationFlowWithTestModel:
    """End-to-end summarization via HistorySummarizer using TestModel override."""

    @pytest.mark.asyncio
    async def test_writes_history_row(self, db: Database, mock_agent: MagicMock) -> None:
        sessions_mock = MagicMock()
        sessions_mock.get_unconsolidated_blobs_with_ids.return_value = []
        sessions_mock.update_current_history_id = MagicMock()

        summarizer = HistorySummarizer(
            db=db,
            agent=mock_agent,
            sessions=sessions_mock,
            context_window_tokens=100,
        )
        tm = _summarizer_test_model(summary="[2026-04-09 10:00] User discussed testing patterns.")
        mock_agent.pydantic_agent.model = tm

        messages = _make_model_messages(3)
        with _summarizer_agent.override(model=tm):
            result = await summarizer.summarize_messages("session:test", messages)

        assert result is True
        histories = db.get_all_histories("session:test")
        assert len(histories) == 1
        assert "discussed testing" in histories[0].summary.lower()

    @pytest.mark.asyncio
    async def test_updates_session_current_history_id(
        self, db: Database, mock_agent: MagicMock
    ) -> None:
        sessions_mock = MagicMock()
        sessions_mock.get_unconsolidated_blobs_with_ids.return_value = []
        sessions_mock.update_current_history_id = MagicMock()

        summarizer = HistorySummarizer(
            db=db,
            agent=mock_agent,
            sessions=sessions_mock,
            context_window_tokens=100,
        )
        tm = _summarizer_test_model(summary="[2026-04-09 10:00] Summary.")
        mock_agent.pydantic_agent.model = tm

        messages = _make_model_messages(3)
        with _summarizer_agent.override(model=tm):
            await summarizer.summarize_messages("session:test", messages)

        sessions_mock.update_current_history_id.assert_called_once()
        call_args = sessions_mock.update_current_history_id.call_args
        assert call_args[0][0] == "session:test"
        assert isinstance(call_args[0][1], int)

    @pytest.mark.asyncio
    async def test_incorporates_existing_summary(self, db: Database, mock_agent: MagicMock) -> None:
        # Pre-populate with a summary
        store = HistoryStore(db, "session:test")
        store.add("Previous summary content.", None)

        sessions_mock = MagicMock()
        sessions_mock.get_unconsolidated_blobs_with_ids.return_value = []
        sessions_mock.update_current_history_id = MagicMock()

        summarizer = HistorySummarizer(
            db=db,
            agent=mock_agent,
            sessions=sessions_mock,
            context_window_tokens=100,
        )
        tm = _summarizer_test_model(summary="[2026-04-09 11:00] Updated with new info.")
        mock_agent.pydantic_agent.model = tm

        messages = _make_model_messages(2)
        with _summarizer_agent.override(model=tm):
            await summarizer.summarize_messages("session:test", messages)

        histories = db.get_all_histories("session:test")
        assert len(histories) == 2
        # New summary should be there
        assert "Updated with new info" in histories[1].summary

    @pytest.mark.asyncio
    async def test_usage_tracked(self) -> None:
        """Verify usage is tracked after agent run."""
        tm = _summarizer_test_model()
        with _summarizer_agent.override(model=tm):
            result = await _summarizer_agent.run(
                user_prompt="Summarize.", model=tm, deps=_EMPTY_SUMMARIZER_DEPS
            )
        usage = result.usage()
        assert usage.requests >= 1
