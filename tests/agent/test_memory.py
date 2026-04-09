"""Tests for Database-backed MemoryStore and HistoryCompressor."""

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
    ToolCallPart,
    ToolReturnPart,
    SystemPromptPart,
    RetryPromptPart,
)
from pydantic_ai.models.test import TestModel

from nanobot.memory import (
    HistoryStore,
    FactStore,
    HistoryCompressor,
    format_messages_for_summarizer,
    _summarizer_agent,
    _extractor_agent,
)
from nanobot.db import Database, upgrade_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_messages(count: int = 10) -> list[ModelMessage]:
    """Create ModelMessage list for testing."""
    messages = []
    for i in range(count):
        if i % 2 == 0:
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"msg{i}")]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=f"msg{i}")]))
    return messages


class _FailingModel(TestModel):
    def request(self, messages, model_settings, model_request_parameters):
        raise Exception("summarization failed")


# ---------------------------------------------------------------------------
# TestFormatMessagesForSummarizer
# ---------------------------------------------------------------------------


class TestFormatMessagesForSummarizer:
    def test_user_prompt(self) -> None:
        messages = [ModelRequest(parts=[UserPromptPart(content="Hello world")])]
        result = format_messages_for_summarizer(messages)
        assert "USER: Hello world" in result

    def test_assistant_text(self) -> None:
        messages = [ModelResponse(parts=[TextPart(content="Hi there")])]
        result = format_messages_for_summarizer(messages)
        assert "ASSISTANT: Hi there" in result

    def test_tool_call(self) -> None:
        messages = [
            ModelResponse(
                parts=[ToolCallPart(tool_name="test_tool", args={"x": 1}, tool_call_id="abc")]
            )
        ]
        result = format_messages_for_summarizer(messages)
        assert "ASSISTANT [call test_tool]" in result
        assert "x" in result

    def test_tool_result(self) -> None:
        messages = [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="test_tool", content="result ok", tool_call_id="abc")
                ]
            )
        ]
        result = format_messages_for_summarizer(messages)
        assert "TOOL [test_tool]: result ok" in result

    def test_system_prompt(self) -> None:
        messages = [ModelRequest(parts=[SystemPromptPart(content="You are helpful")])]
        result = format_messages_for_summarizer(messages)
        assert "SYSTEM: You are helpful" in result

    def test_retry_prompt(self) -> None:
        messages = [
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name="test_tool", content="retry content", tool_call_id="abc"
                    )
                ]
            )
        ]
        result = format_messages_for_summarizer(messages)
        assert "TOOL [test_tool]: retry content" in result

    def test_thinking_part_skipped(self) -> None:
        from pydantic_ai.messages import ThinkingPart

        messages = [
            ModelResponse(parts=[ThinkingPart(content="thinking..."), TextPart(content="answer")])
        ]
        result = format_messages_for_summarizer(messages)
        assert "thinking" not in result
        assert "ASSISTANT: answer" in result

    def test_empty_list(self) -> None:
        result = format_messages_for_summarizer([])
        assert result == ""

    def test_mixed_messages(self) -> None:
        messages = [
            ModelRequest(parts=[UserPromptPart(content="What is X?")]),
            ModelResponse(parts=[TextPart(content="X is Y.")]),
            ModelRequest(parts=[UserPromptPart(content="Thanks!")]),
        ]
        result = format_messages_for_summarizer(messages)
        assert "USER: What is X?" in result
        assert "ASSISTANT: X is Y." in result
        assert "USER: Thanks!" in result


# ---------------------------------------------------------------------------
# TestHistoryStoreOperations
# ---------------------------------------------------------------------------


class TestHistoryStoreOperations:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        db.ensure_session("session:test")
        return db

    def test_add_returns_id(self, db: Database) -> None:
        store = HistoryStore(db, "session:test")
        history_id = store.add("Test summary", None)
        assert isinstance(history_id, int)
        assert history_id > 0

    def test_get_current_summary_empty(self, db: Database) -> None:
        store = HistoryStore(db, "session:test")
        assert store.get_current_summary() is None

    def test_get_current_summary_after_add(self, db: Database) -> None:
        store = HistoryStore(db, "session:test")
        id1 = store.add("First summary", None)
        db.update_current_history_id("session:test", id1)
        assert store.get_current_summary() == "First summary"

    def test_multiple_histories(self, db: Database) -> None:
        store = HistoryStore(db, "session:test")
        id1 = store.add("Summary 1", None)
        db.update_current_history_id("session:test", id1)
        # Second history has no message reference (None), just to accumulate
        id2 = store.add("Summary 2", None)
        db.update_current_history_id("session:test", id2)
        assert store.get_current_summary() == "Summary 2"
        # Verify get_all_histories
        all_histories = db.get_all_histories("session:test")
        assert len(all_histories) == 2
        assert all_histories[0].summary == "Summary 1"
        assert all_histories[1].summary == "Summary 2"


# ---------------------------------------------------------------------------
# TestFactStoreOperations
# ---------------------------------------------------------------------------


class TestFactStoreOperations:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        db.ensure_session("session:test")
        return db

    def test_add(self, db: Database) -> None:
        store = FactStore(db, "session:test")
        fact_id = store.add("User likes coffee", "preference")
        assert isinstance(fact_id, int)
        assert fact_id > 0

    def test_add_many(self, db: Database) -> None:
        store = FactStore(db, "session:test")
        store.add_many([("Fact 1", "fact"), ("Fact 2", "fact")])
        facts = db.get_facts("session:test")
        assert len(facts) == 2

    def test_get_digest_empty(self, db: Database) -> None:
        store = FactStore(db, "session:test")
        assert store.get_digest() == ""

    def test_get_digest_with_facts(self, db: Database) -> None:
        store = FactStore(db, "session:test")
        store.add("Coffee preference", "preference")
        store.add("User is a developer", "fact")
        digest = store.get_digest()
        assert "## Known Facts" in digest
        assert "preference" in digest
        assert "Coffee preference" in digest

    def test_get_existing_facts_text_empty(self, db: Database) -> None:
        store = FactStore(db, "session:test")
        assert store.get_existing_facts_text() == ""

    def test_get_existing_facts_text(self, db: Database) -> None:
        store = FactStore(db, "session:test")
        store.add("Likes pizza", "preference")
        store.add("Is from NYC", "fact")
        text = store.get_existing_facts_text()
        assert "[preference] Likes pizza" in text
        assert "[fact] Is from NYC" in text


# ---------------------------------------------------------------------------
# TestHistoryCompressorBoundary
# ---------------------------------------------------------------------------


class TestHistoryCompressorBoundary:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        return db

    def _make_blobs(self, count: int) -> list[tuple[int, list]]:
        """Create blobs (row_id, messages) pairs for testing."""
        messages = _make_model_messages(count)
        blobs = []
        for i, msg in enumerate(messages):
            blobs.append((i + 1, [msg]))
        return blobs

    def test_boundary_returns_row_id(self, db: Database) -> None:
        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
        )
        blobs = self._make_blobs(6)
        boundary = summarizer.pick_summarization_boundary(blobs, tokens_to_remove=1)
        assert boundary is not None
        assert isinstance(boundary, int)

    def test_no_boundary_empty_blobs(self, db: Database) -> None:
        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
        )
        boundary = summarizer.pick_summarization_boundary([], tokens_to_remove=50)
        assert boundary is None

    def test_no_boundary_zero_tokens(self, db: Database) -> None:
        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
        )
        blobs = self._make_blobs(3)
        boundary = summarizer.pick_summarization_boundary(blobs, tokens_to_remove=0)
        assert boundary is None

    def test_boundary_finds_valid_row_id(self, db: Database) -> None:
        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
        )
        blobs = self._make_blobs(6)
        boundary = summarizer.pick_summarization_boundary(blobs, tokens_to_remove=1)
        assert boundary is not None
        assert 1 <= boundary <= 6


# ---------------------------------------------------------------------------
# TestHistoryCompressorSummarize
# ---------------------------------------------------------------------------


class TestHistoryCompressorSummarize:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        db.ensure_session("session:test")
        return db

    def _make_mock_agent(self, tm: TestModel) -> MagicMock:
        mock = MagicMock()
        mock.pydantic_agent.model = tm
        return mock

    @pytest.mark.asyncio
    async def test_empty_returns_true(self, db: Database, tmp_path: Path) -> None:
        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
        )
        result = await summarizer.summarize_messages("session:test", [])
        assert result is True

    @pytest.mark.asyncio
    async def test_failure_returns_false(self, db: Database, tmp_path: Path) -> None:
        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
        )
        fm = _FailingModel()
        summarizer.agent.pydantic_agent.model = fm

        messages = _make_model_messages(5)
        result = await summarizer.summarize_messages("session:test", messages)
        assert result is False

    @pytest.mark.asyncio
    async def test_raw_summary_after_max_failures(self, db: Database, tmp_path: Path) -> None:
        sessions_mock = MagicMock()
        # Return empty blobs so boundary_row_id is None (avoids FK constraint on non-existent message)
        sessions_mock.get_unconsolidated_blobs_with_ids.return_value = []
        sessions_mock.update_current_history_id = MagicMock()

        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=sessions_mock,
            context_window_tokens=100,
        )
        summarizer.agent.pydantic_agent.model = _FailingModel()

        messages = _make_model_messages(5)
        for _ in range(HistoryCompressor._MAX_FAILURES_BEFORE_RAW_SUMMARY - 1):
            await summarizer.summarize_messages("session:test", messages)

        with _summarizer_agent.override(model=_FailingModel()):
            result = await summarizer.summarize_messages("session:test", messages)

        assert result is True
        histories = db.get_all_histories("session:test")
        assert len(histories) == 1
        assert "[RAW]" in histories[0].summary


# ---------------------------------------------------------------------------
# TestFactExtraction
# ---------------------------------------------------------------------------


class TestFactExtraction:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        db.ensure_session("session:test")
        return db

    @pytest.mark.asyncio
    async def test_extracts_facts(self, db: Database, tmp_path: Path) -> None:
        sessions_mock = MagicMock()
        sessions_mock.get_unconsolidated_blobs_with_ids.return_value = []

        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=sessions_mock,
            context_window_tokens=100,
        )
        tm = TestModel(
            custom_output_args={
                "facts": [
                    {"content": "User likes Python", "category": "preference"},
                    {"content": "User works at Acme", "category": "fact"},
                ]
            }
        )
        summarizer.agent.pydantic_agent.model = tm

        messages = _make_model_messages(3)
        with _extractor_agent.override(model=tm):
            await summarizer.extract_facts("session:test", messages)

        facts = db.get_facts("session:test")
        assert len(facts) == 2
        categories = {f.category for f in facts}
        assert "preference" in categories
        assert "fact" in categories

    @pytest.mark.asyncio
    async def test_no_facts_from_empty(self, db: Database, tmp_path: Path) -> None:
        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
        )
        await summarizer.extract_facts("session:test", [])
        facts = db.get_facts("session:test")
        assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_failure_is_swallowed(self, db: Database, tmp_path: Path) -> None:
        sessions_mock = MagicMock()
        sessions_mock.get_unconsolidated_blobs_with_ids.return_value = []

        summarizer = HistoryCompressor(
            db=db,
            agent=MagicMock(),
            sessions=sessions_mock,
            context_window_tokens=100,
        )
        summarizer.agent.pydantic_agent.model = _FailingModel()

        messages = _make_model_messages(3)
        # Should not raise
        await summarizer.extract_facts("session:test", messages)
        # No facts should be added
        facts = db.get_facts("session:test")
        assert len(facts) == 0
