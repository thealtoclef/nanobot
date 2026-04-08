"""Tests for Database-backed MemoryStore and MemoryConsolidator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.models.test import TestModel

from nanobot.agent.memory import (
    MemoryStore,
    _consolidation_agent,
)
from nanobot.db import Database, upgrade_db
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


def _make_model_messages(count: int = 10) -> list:
    """Create ModelMessage list for testing MemoryConsolidator."""
    messages = []
    for i in range(count):
        if i % 2 == 0:
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"msg{i}")]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=f"msg{i}")]))
    return messages


def _model_messages_to_dicts(messages: list) -> list[dict]:
    """Convert ModelMessage list to dicts for MemoryStore.consolidate."""
    result = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    result.append(
                        {
                            "role": "user",
                            "content": part.content,
                            "timestamp": f"2026-01-01T00:{len(result):02d}",
                            "tools_used": [],
                        }
                    )
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    result.append(
                        {
                            "role": "assistant",
                            "content": part.content,
                            "timestamp": f"2026-01-01T00:{len(result):02d}",
                            "tools_used": [],
                        }
                    )
    return result


class _FailingModel(TestModel):
    def request(self, messages, model_settings, model_request_parameters):
        raise Exception("consolidation failed")


class TestMemoryStoreConsolidate:
    """Test MemoryStore.consolidate() with Database backend.

    Uses PydanticAI's TestModel / FunctionModel instead of patching
    ``_consolidation_agent.run``.  Tests that duplicate
    ``test_memory_testmodel.py`` (e.g. basic write / read, failure-counter
    reset) have been removed — see ``TestConsolidationFlowWithTestModel``.
    """

    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        db.ensure_session("session:test")
        return db

    @pytest.mark.asyncio
    async def test_consolidate_empty_chunk_returns_true(self, db: Database, tmp_path: Path) -> None:
        store = MemoryStore(db, "session:test")
        result = await store.consolidate([], MagicMock())
        assert result is True

    @pytest.mark.asyncio
    async def test_consolidate_failure_returns_false(self, db: Database, tmp_path: Path) -> None:
        store = MemoryStore(db, "session:test")
        fm = _FailingModel()
        mock_agent = MagicMock()
        mock_agent.pydantic_agent.model = fm

        with _consolidation_agent.override(model=fm):
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(5)), mock_agent
            )
            assert result is False
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(5)), mock_agent
            )
            assert result is False
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(5)), mock_agent
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_consolidate_empty_history_entry_returns_false(
        self, db: Database, tmp_path: Path
    ) -> None:
        store = MemoryStore(db, "session:test")
        tm = TestModel(
            custom_output_args={
                "history_entry": "   ",
                "memory_update": "# Memory\nNew fact.",
            }
        )
        mock_agent = MagicMock()
        mock_agent.pydantic_agent.model = tm

        with _consolidation_agent.override(model=tm):
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(5)), mock_agent
            )
        assert result is False

    @pytest.mark.asyncio
    async def test_consolidate_multiple_messages_archived(
        self, db: Database, tmp_path: Path
    ) -> None:
        store = MemoryStore(db, "session:test")
        tm = TestModel(
            custom_output_args={
                "history_entry": "[2026-01-01] User discussed A, B, and C.",
                "memory_update": "# Memory\nTopics A, B, C discussed.",
            }
        )
        mock_agent = MagicMock()
        mock_agent.pydantic_agent.model = tm

        with _consolidation_agent.override(model=tm):
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(20)), mock_agent
            )

        assert result is True
        history = db.get_recent_history("session:test")
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_raw_archive_after_max_failures(self, db: Database, tmp_path: Path) -> None:
        store = MemoryStore(db, "session:test")
        fm = _FailingModel()
        mock_agent = MagicMock()
        mock_agent.pydantic_agent.model = fm

        with _consolidation_agent.override(model=fm):
            for _ in range(MemoryStore._MAX_FAILURES_BEFORE_RAW_ARCHIVE - 1):
                await store.consolidate(
                    _model_messages_to_dicts(_make_model_messages(5)), mock_agent
                )

            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(5)), mock_agent
            )
        assert result is True

        history = db.get_recent_history("session:test")
        assert len(history) == 1
        assert "[RAW]" in history[0]
        assert "5 messages" in history[0]


class TestMemoryStoreDatabaseOperations:
    """Test MemoryStore database read/write operations."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        # Ensure all tables exist (migration may be incomplete)
        Base.metadata.create_all(db.engine)
        db.ensure_session("session:test")
        return db

    def test_read_long_term_empty(self, db: Database) -> None:
        store = MemoryStore(db, "session:test")
        assert store.read_long_term() == ""

    def test_write_and_read_long_term(self, db: Database) -> None:
        store = MemoryStore(db, "session:test")
        store.write_long_term("# Memory\nA fact.")
        assert store.read_long_term() == "# Memory\nA fact."

    def test_append_and_read_history(self, db: Database) -> None:
        store = MemoryStore(db, "session:test")
        store.append_history("[2026-01-01] Event 1.")
        store.append_history("[2026-01-01] Event 2.")
        history = db.get_recent_history("session:test", limit=10)
        assert len(history) == 2
        assert "Event 1" in history[0]
        assert "Event 2" in history[1]

    def test_get_memory_context_empty(self, db: Database) -> None:
        store = MemoryStore(db, "session:test")
        assert store.get_memory_context() == ""

    def test_get_memory_context_with_content(self, db: Database) -> None:
        store = MemoryStore(db, "session:test")
        store.write_long_term("# Memory\nA fact.")
        ctx = store.get_memory_context()
        assert "## Long-term Memory" in ctx
        assert "A fact" in ctx

    def test_per_session_isolation(self, db: Database) -> None:
        """Each session has its own memory and history."""
        db.ensure_session("session:A")
        db.ensure_session("session:B")
        store_a = MemoryStore(db, "session:A")
        store_b = MemoryStore(db, "session:B")

        store_a.write_long_term("# Memory\nFact for A.")
        store_b.write_long_term("# Memory\nFact for B.")

        assert store_a.read_long_term() == "# Memory\nFact for A."
        assert store_b.read_long_term() == "# Memory\nFact for B."

        store_a.append_history("[2026-01-01] Event A.")
        store_b.append_history("[2026-01-01] Event B.")

        history_a = db.get_recent_history("session:A")
        history_b = db.get_recent_history("session:B")
        assert len(history_a) == 1
        assert len(history_b) == 1
        assert "Event A" in history_a[0]
        assert "Event B" in history_b[0]


class TestMemoryConsolidatorBoundary:
    """Test MemoryConsolidator.pick_consolidation_boundary()."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        return db

    def _make_session_manager(self, db: Database) -> MagicMock:
        """Create a mock SessionManager with messages."""
        from nanobot.session.manager import SessionManager

        mgr = MagicMock(spec=SessionManager)
        return mgr

    def test_boundary_returns_tuple_with_index_and_tokens(self, db: Database) -> None:
        """Boundary returns (index, removed_tokens) tuple."""
        from nanobot.agent.memory import MemoryConsolidator

        consolidator = MemoryConsolidator(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
            build_messages=MagicMock(),
        )

        messages = _make_model_messages(6)
        # Pick boundary should work with ModelMessage list
        boundary = consolidator.pick_consolidation_boundary(messages, tokens_to_remove=100)
        assert boundary is not None
        end_idx, removed_tokens = boundary
        assert isinstance(end_idx, int)
        assert isinstance(removed_tokens, int)
        assert 0 < end_idx <= len(messages)
        assert removed_tokens >= 0

    def test_no_boundary_when_already_at_start(self, db: Database) -> None:
        """Returns None if start >= len(messages)."""
        from nanobot.agent.memory import MemoryConsolidator

        consolidator = MemoryConsolidator(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
            build_messages=MagicMock(),
        )

        # No messages to consolidate
        boundary = consolidator.pick_consolidation_boundary([], tokens_to_remove=50)
        assert boundary is None

    def test_no_boundary_when_tokens_to_remove_zero(self, db: Database) -> None:
        """Returns None if tokens_to_remove <= 0."""
        from nanobot.agent.memory import MemoryConsolidator

        consolidator = MemoryConsolidator(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
            build_messages=MagicMock(),
        )

        messages = _make_model_messages(3)
        boundary = consolidator.pick_consolidation_boundary(messages, tokens_to_remove=0)
        assert boundary is None

    def test_boundary_finds_user_turn_boundary(self, db: Database) -> None:
        """Boundary should land on a user-turn boundary."""
        from nanobot.agent.memory import MemoryConsolidator

        consolidator = MemoryConsolidator(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
            build_messages=MagicMock(),
        )

        messages = _make_model_messages(6)
        boundary = consolidator.pick_consolidation_boundary(messages, tokens_to_remove=100)
        assert boundary is not None
        end_idx, _ = boundary
        # Should be at a user message boundary (even indices are user in our fixture)
        assert end_idx % 2 == 0 or end_idx == 1


class TestMemoryConsolidatorArchiveMessages:
    """Test MemoryConsolidator.archive_messages()."""

    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        return db

    @pytest.mark.asyncio
    async def test_archive_empty_messages_returns_true(self, db: Database) -> None:
        """Archiving empty list is a no-op returning True."""
        from nanobot.agent.memory import MemoryConsolidator

        consolidator = MemoryConsolidator(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
            build_messages=MagicMock(),
        )

        result = await consolidator.archive_messages("session:test", [])
        assert result is True

    @pytest.mark.asyncio
    async def test_archive_messages_retries_until_success(self, db: Database) -> None:
        """archive_messages retries consolidation until it succeeds."""
        from nanobot.agent.memory import MemoryConsolidator
        from unittest.mock import AsyncMock

        call_count = 0

        async def fake_consolidate(session_key: str, messages: list[dict[str, object]]) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count >= 2  # fails first time, succeeds second

        consolidator = MemoryConsolidator(
            db=db,
            agent=MagicMock(),
            sessions=MagicMock(),
            context_window_tokens=100,
            build_messages=MagicMock(),
        )
        # Replace the bound method with a standalone async function
        # noinspection PyTypeChecker
        consolidator.consolidate_messages = fake_consolidate  # type: ignore[assignment]

        result = await consolidator.archive_messages(
            "session:test", _model_messages_to_dicts(_make_model_messages(1))
        )
        assert result is True
        assert call_count == 2
