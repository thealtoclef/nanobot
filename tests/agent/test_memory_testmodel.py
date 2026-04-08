"""Consolidation tests using PydanticAI's TestModel.

Exercises the real ``_consolidation_agent`` with
``agent.override(model=TestModel(...))`` — no patching of ``agent.run``.
Validates that ``output_type=ConsolidationResult`` produces correct structured
output, ``deps_type=ConsolidationDeps`` is injected into instructions, and the
consolidation flow works end-to-end with a deterministic model substitute.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.models.test import TestModel

from nanobot.agent.memory import (
    ConsolidationDeps,
    ConsolidationResult,
    MemoryStore,
    _consolidation_agent,
    _INITIAL_MEMORY_TEMPLATE,
)
from nanobot.db import Database, upgrade_db
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)


def _make_model_messages(count: int = 5) -> list:
    """Create ModelMessage list for testing."""
    messages = []
    for i in range(count):
        if i % 2 == 0:
            messages.append(ModelRequest(parts=[UserPromptPart(content=f"message {i}")]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=f"message {i}")]))
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
                            "timestamp": f"2026-04-08T10:{len(result):02d}:00",
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
                            "timestamp": f"2026-04-08T10:{len(result):02d}:00",
                            "tools_used": [],
                        }
                    )
    return result


def _consolidation_test_model(
    history_entry: str = "[2026-04-08 10:00] Test consolidation.",
    memory_update: str = "# Memory\nConsolidated.",
) -> TestModel:
    return TestModel(
        custom_output_args={
            "history_entry": history_entry,
            "memory_update": memory_update,
        },
    )


_EMPTY_DEPS = ConsolidationDeps(current_memory="", session_messages=[])


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


class TestConsolidationResultOutput:
    """Verify TestModel produces ConsolidationResult through the agent."""

    @pytest.mark.asyncio
    async def test_returns_consolidation_result_type(self) -> None:
        tm = _consolidation_test_model()
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=_EMPTY_DEPS)
        assert isinstance(result.output, ConsolidationResult)

    @pytest.mark.asyncio
    async def test_history_entry_field_parsed(self) -> None:
        expected = "[2026-04-08 12:00] User discussed testing patterns."
        tm = _consolidation_test_model(history_entry=expected)
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=_EMPTY_DEPS)
        assert result.output.history_entry == expected

    @pytest.mark.asyncio
    async def test_memory_update_field_parsed(self) -> None:
        expected = "# Memory\n- Fact A\n- Fact B"
        tm = _consolidation_test_model(memory_update=expected)
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=_EMPTY_DEPS)
        assert result.output.memory_update == expected

    @pytest.mark.asyncio
    async def test_both_fields_populated(self) -> None:
        tm = _consolidation_test_model(
            history_entry="[2026-04-08 10:00] Multi-field test.",
            memory_update="# Memory\nMulti.",
        )
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=_EMPTY_DEPS)
        output = result.output
        assert output.history_entry == "[2026-04-08 10:00] Multi-field test."
        assert output.memory_update == "# Memory\nMulti."


class TestConsolidationDepsInjection:
    """Verify deps are injected into the agent's dynamic instructions."""

    @pytest.mark.asyncio
    async def test_current_memory_passed(self) -> None:
        tm = _consolidation_test_model()
        deps = ConsolidationDeps(
            current_memory="# Memory\nExisting fact.",
            session_messages=[],
        )
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=deps)
        assert isinstance(result.output, ConsolidationResult)

    @pytest.mark.asyncio
    async def test_session_messages_passed(self) -> None:
        messages = _model_messages_to_dicts(_make_model_messages(3))
        tm = _consolidation_test_model()
        deps = ConsolidationDeps(current_memory="", session_messages=messages)
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=deps)
        assert isinstance(result.output, ConsolidationResult)

    @pytest.mark.asyncio
    async def test_both_deps_fields(self) -> None:
        tm = _consolidation_test_model()
        deps = ConsolidationDeps(
            current_memory="# Memory\nUser prefers dark mode.",
            session_messages=_model_messages_to_dicts(_make_model_messages(5)),
        )
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=deps)
        assert isinstance(result.output, ConsolidationResult)

    @pytest.mark.asyncio
    async def test_empty_deps(self) -> None:
        tm = _consolidation_test_model()
        deps = ConsolidationDeps(current_memory="", session_messages=[])
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=deps)
        assert isinstance(result.output, ConsolidationResult)


class TestConsolidationFlowWithTestModel:
    """End-to-end consolidation via MemoryStore using TestModel override."""

    @pytest.mark.asyncio
    async def test_writes_history(self, db: Database, mock_agent: MagicMock) -> None:
        store = MemoryStore(db, "session:test")
        tm = _consolidation_test_model(
            history_entry="[2026-04-08 10:00] Discussed testing.",
            memory_update="# Memory\nTesting topic.",
        )

        with _consolidation_agent.override(model=tm):
            mock_agent.pydantic_agent.model = tm
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(3)), mock_agent
            )

        assert result is True
        history = db.get_recent_history("session:test", limit=5)
        assert len(history) == 1
        assert "Discussed testing" in history[0]

    @pytest.mark.asyncio
    async def test_updates_long_term_memory(self, db: Database, mock_agent: MagicMock) -> None:
        store = MemoryStore(db, "session:test")
        tm = _consolidation_test_model(
            history_entry="[2026-04-08 10:00] Summary.",
            memory_update="# Memory\nUpdated fact.",
        )

        with _consolidation_agent.override(model=tm):
            mock_agent.pydantic_agent.model = tm
            await store.consolidate(_model_messages_to_dicts(_make_model_messages(2)), mock_agent)

        curated = db.get_curated_memory("session:test")
        assert curated is not None
        assert "Updated fact" in curated

    @pytest.mark.asyncio
    async def test_seeds_template_on_empty_memory(
        self, db: Database, mock_agent: MagicMock
    ) -> None:
        store = MemoryStore(db, "session:test")
        assert store.read_long_term() == ""

        tm = _consolidation_test_model(
            history_entry="[2026-04-08 10:00] First consolidation.",
            memory_update=_INITIAL_MEMORY_TEMPLATE + "\n## Notes\nFirst run.",
        )

        with _consolidation_agent.override(model=tm):
            mock_agent.pydantic_agent.model = tm
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(2)), mock_agent
            )

        assert result is True
        assert db.get_curated_memory("session:test") is not None

    @pytest.mark.asyncio
    async def test_preserves_memory_when_update_equals_current(
        self, db: Database, mock_agent: MagicMock
    ) -> None:
        store = MemoryStore(db, "session:test")
        original = "# Memory\nOriginal fact."
        store.write_long_term(original)

        tm = _consolidation_test_model(
            history_entry="[2026-04-08 10:00] No changes.",
            memory_update=original,
        )

        with _consolidation_agent.override(model=tm):
            mock_agent.pydantic_agent.model = tm
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(2)), mock_agent
            )

        assert result is True
        assert store.read_long_term() == original

    @pytest.mark.asyncio
    async def test_empty_memory_update_keeps_existing(
        self, db: Database, mock_agent: MagicMock
    ) -> None:
        store = MemoryStore(db, "session:test")
        store.write_long_term("# Memory\nExisting content.")

        tm = _consolidation_test_model(
            history_entry="[2026-04-08 10:00] History only.",
            memory_update="",
        )

        with _consolidation_agent.override(model=tm):
            mock_agent.pydantic_agent.model = tm
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(2)), mock_agent
            )

        assert result is True
        assert "Existing content" in store.read_long_term()

    @pytest.mark.asyncio
    async def test_resets_failure_counter(self, db: Database, mock_agent: MagicMock) -> None:
        store = MemoryStore(db, "session:test")
        store._consecutive_failures = 2

        tm = _consolidation_test_model()
        with _consolidation_agent.override(model=tm):
            mock_agent.pydantic_agent.model = tm
            result = await store.consolidate(
                _model_messages_to_dicts(_make_model_messages(2)), mock_agent
            )

        assert result is True
        assert store._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_multiple_rounds_accumulate_history(
        self, db: Database, mock_agent: MagicMock
    ) -> None:
        store = MemoryStore(db, "session:test")
        tm1 = _consolidation_test_model(
            history_entry="[2026-04-08 10:00] Round 1.",
            memory_update="# Memory\nRound 1 memory.",
        )
        tm2 = _consolidation_test_model(
            history_entry="[2026-04-08 11:00] Round 2.",
            memory_update="# Memory\nRound 1 + 2 memory.",
        )

        with _consolidation_agent.override(model=tm1):
            mock_agent.pydantic_agent.model = tm1
            await store.consolidate(_model_messages_to_dicts(_make_model_messages(3)), mock_agent)

        with _consolidation_agent.override(model=tm2):
            mock_agent.pydantic_agent.model = tm2
            await store.consolidate(_model_messages_to_dicts(_make_model_messages(3)), mock_agent)

        history = db.get_recent_history("session:test", limit=10)
        assert len(history) == 2
        assert "Round 1" in history[0]
        assert "Round 2" in history[1]

    @pytest.mark.asyncio
    async def test_usage_tracked(self) -> None:
        tm = _consolidation_test_model()
        with _consolidation_agent.override(model=tm):
            result = await _consolidation_agent.run(user_prompt="test", model=tm, deps=_EMPTY_DEPS)
        usage = result.usage()
        assert usage.requests >= 1
