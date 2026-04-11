"""Tests for SubagentManager.cancel_by_session and message duplication fix."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai import ModelResponse, TextPart
from pydantic_ai.messages import ModelRequest, ModelMessage, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from nanobot.db import Database, SubagentSessionRow, upgrade_db
from nanobot.subagent import SubagentManager


def _make_manager(tmp_path: Path) -> tuple[SubagentManager, Database]:
    """Create a SubagentManager with a real Database."""
    from nanobot.bus.queue import MessageBus

    upgrade_db(tmp_path)
    db = Database(tmp_path)
    bus = MessageBus()
    agent = MagicMock()
    agent.models = []

    with patch.object(SubagentManager, "_build_subagent_agent", return_value=_make_subagent_mock()):
        mgr = SubagentManager(
            db=db,
            agent=agent,
            workspace=tmp_path,
            bus=bus,
            max_tool_result_chars=16000,
        )
    return mgr, db


def _make_subagent_mock() -> MagicMock:
    subagent = MagicMock()
    subagent.run = AsyncMock(return_value=('{"action": "skip"}', []))
    return subagent


# =============================================================================
# Tests for cancel_by_session
# =============================================================================


@pytest.mark.asyncio
async def test_cancel_by_session_returns_zero_when_no_tasks(tmp_path: Path) -> None:
    """cancel_by_session returns 0 when no tasks exist for that session."""
    mgr, _ = _make_manager(tmp_path)

    result = await mgr.cancel_by_session("nonexistent-session")

    assert result == 0


@pytest.mark.asyncio
async def test_cancel_by_session_returns_zero_when_session_has_no_running_tasks(
    tmp_path: Path,
) -> None:
    """Returns 0 when session exists but has no running tasks in DB."""
    mgr, db = _make_manager(tmp_path)

    # Ensure parent session exists (FK constraint)
    db.ensure_session("test:c1")

    # Create a "completed" subagent entry in DB (no running task)
    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key="test:c1",
        label="completed",
        task="already done",
        origin_channel="cli",
        origin_chat_id="direct",
    )
    db.complete_subagent_session(id=id, status="completed", result="done")

    result = await mgr.cancel_by_session("test:c1")

    assert result == 0


@pytest.mark.asyncio
async def test_cancel_by_session_cancels_active_subagent(tmp_path: Path) -> None:
    """cancel_by_session cancels running subagent tasks for the session."""
    mgr, db = _make_manager(tmp_path)

    # Ensure parent session exists (FK constraint)
    db.ensure_session("test:c1")

    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def slow_coro():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    # Create DB entries for subagent
    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key="test:c1",
        label="slow",
        task="slow task",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    # Create a task that mimics a running subagent
    task = asyncio.create_task(slow_coro())
    mgr._running_tasks[id] = task
    await asyncio.wait_for(started.wait(), timeout=1.0)

    result = await mgr.cancel_by_session("test:c1")

    assert result == 1
    assert cancelled.is_set()

    # Cleanup
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_cancel_by_session_ignores_already_done_tasks(tmp_path: Path) -> None:
    """Already-completed tasks are not counted as cancelled."""
    mgr, db = _make_manager(tmp_path)

    # Ensure parent session exists (FK constraint)
    db.ensure_session("test:c1")

    async def already_done():
        return

    # Create DB entry
    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key="test:c1",
        label="done",
        task="done task",
        origin_channel="cli",
        origin_chat_id="direct",
    )
    db.complete_subagent_session(id=id, status="completed", result="done")

    done_task = asyncio.create_task(already_done())
    await done_task  # ensure task is truly done

    mgr._running_tasks[id] = done_task

    result = await mgr.cancel_by_session("test:c1")

    assert result == 0


# =============================================================================
# Test for no history duplication in subagent
# =============================================================================


@pytest.mark.asyncio
async def test_run_subagent_no_duplication(tmp_path: Path) -> None:
    """Verify _run_subagent passes correct user_message and empty message_history.

    This ensures the subagent's system prompt doesn't leak through history —
    the subagent gets its instructions via Agent.instructions, not via messages.
    """
    captured: dict = {}

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured["messages"] = messages
        return ModelResponse(parts=[TextPart(content='{"action": "done"}')])

    # Build a real SubagentManager with a mock agent providing a FunctionModel
    from nanobot.bus.queue import MessageBus

    upgrade_db(tmp_path)
    db = Database(tmp_path)
    bus = MessageBus()
    agent = MagicMock()
    agent.models = [FunctionModel(capture)]

    mgr = SubagentManager(
        db=db,
        agent=agent,
        workspace=tmp_path,
        bus=bus,
        max_tool_result_chars=16000,
    )

    # Ensure parent session exists (FK constraint)
    db.ensure_session("cli:direct")

    # Create a DB entry for this subagent
    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key="cli:direct",
        label="dedup test",
        task="do something important",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    # Override the subagent's pydantic_agent model to capture call arguments
    with mgr._subagent_agent.pydantic_agent.override(model=FunctionModel(capture)):
        await mgr._run_subagent(
            id=id,
            task="do something important",
            label="dedup test",
            origin={
                "channel": "cli",
                "chat_id": "direct",
                "session_key": "cli:direct",
                "parent_key": "cli:direct",
            },
        )

    # Extract all UserPromptPart contents from captured messages
    user_prompts = []
    for msg in captured.get("messages", []):
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str):
                        user_prompts.append(content)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                user_prompts.append(item)

    # Verify: exactly one user prompt with the task content, no history duplication
    assert len(user_prompts) == 1, (
        f"Expected 1 user prompt, got {len(user_prompts)}: {user_prompts}"
    )
    assert user_prompts[0] == "do something important"


# =============================================================================
# Tests for spawn
# =============================================================================


@pytest.mark.asyncio
async def test_spawn_returns_error_on_duplicate_label(tmp_path: Path) -> None:
    """Spawning with a duplicate label returns an error."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    # Ensure parent session exists (FK constraint)
    db.ensure_session(parent_key)

    # Create a subagent entry in DB
    db.create_subagent_session(
        id=str(uuid.uuid7()),
        parent_key=parent_key,
        label="existing",
        task="existing task",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    # Try to spawn with the same label
    result = await mgr.spawn(
        task="new task with same label",
        parent_key=parent_key,
        label="existing",
    )

    assert "already in use" in result


# =============================================================================
# Tests for list_subagents and get_by_id
# =============================================================================


def test_list_subagents_empty(tmp_path: Path) -> None:
    """list_subagents returns empty list when no subagents."""
    mgr, _ = _make_manager(tmp_path)

    result = mgr.list_subagents()

    assert result == []


def test_list_subagents_filters_by_parent_key(tmp_path: Path) -> None:
    """list_subagents filters by parent_key."""
    mgr, db = _make_manager(tmp_path)

    # Ensure parent sessions exist (FK constraint)
    db.ensure_session("cli:direct")
    db.ensure_session("telegram:123:456")

    # Create subagents with different parent_keys
    id1 = str(uuid.uuid7())
    id2 = str(uuid.uuid7())
    db.create_subagent_session(
        id=id1,
        parent_key="cli:direct",
        label="a",
        task="task1",
        origin_channel="cli",
        origin_chat_id="direct",
    )
    db.create_subagent_session(
        id=id2,
        parent_key="telegram:123:456",
        label="b",
        task="task2",
        origin_channel="telegram",
        origin_chat_id="456",
    )

    result = mgr.list_subagents(parent_key="cli:direct")

    assert len(result) == 1
    assert result[0].id == id1


def test_get_by_id_found(tmp_path: Path) -> None:
    """get_by_id returns row when subagent exists."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    # Ensure parent session exists (FK constraint)
    db.ensure_session(parent_key)

    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key=parent_key,
        label="research",
        task="do research",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    result = mgr.get_by_id(id)

    assert result is not None
    assert result.id == id
    assert result.label == "research"


def test_get_by_id_not_found(tmp_path: Path) -> None:
    """get_by_id returns None when subagent doesn't exist."""
    mgr, _ = _make_manager(tmp_path)

    result = mgr.get_by_id(str(uuid.uuid7()))

    assert result is None


# =============================================================================
# Tests for kill_by_id
# =============================================================================


@pytest.mark.asyncio
async def test_kill_by_id_cancels_task(tmp_path: Path) -> None:
    """kill_by_id cancels a running subagent task."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"
    cancelled = asyncio.Event()

    async def slow_task():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    # Ensure parent session exists (FK constraint)
    db.ensure_session(parent_key)

    # Create DB entry
    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key=parent_key,
        label="killable",
        task="slow task",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    # Create and register the running task
    task = asyncio.create_task(slow_task())
    await asyncio.sleep(0.01)  # let task start
    mgr._running_tasks[id] = task

    result = await mgr.kill_by_id(id)

    assert result is True
    assert cancelled.is_set()

    # Verify DB status was updated
    row = db.get_subagent_session(id)
    assert row is not None
    assert row.status == "cancelled"

    # Cleanup
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_kill_by_id_not_found(tmp_path: Path) -> None:
    """kill_by_id returns False when subagent doesn't exist."""
    mgr, _ = _make_manager(tmp_path)

    result = await mgr.kill_by_id(str(uuid.uuid7()))

    assert result is False


# =============================================================================
# Tests for CancelledError handling
# =============================================================================


@pytest.mark.asyncio
async def test_run_subagent_handles_cancelled_error(tmp_path: Path) -> None:
    """_run_subagent properly handles CancelledError without calling _announce_result."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    # Mock the subagent agent to raise CancelledError
    mgr._subagent = MagicMock()
    mgr._subagent.run = AsyncMock(side_effect=asyncio.CancelledError)

    # Ensure parent session exists (FK constraint)
    db.ensure_session(parent_key)

    # Create DB entry
    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key=parent_key,
        label="cancellable",
        task="cancellable task",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    # Should not raise, should not call _announce_result
    await mgr._run_subagent(
        id=id,
        task="cancellable task",
        label="cancellable",
        origin={"channel": "cli", "chat_id": "direct", "session_key": "", "parent_key": parent_key},
    )

    # Verify DB status was updated to cancelled
    row = db.get_subagent_session(id)
    assert row is not None
    assert row.status == "cancelled"


# =============================================================================
# Additional Tests for SubagentManager DB Integration
# =============================================================================


@pytest.mark.asyncio
async def test_spawn_creates_db_row(tmp_path: Path) -> None:
    """spawn() creates a subagent_sessions DB row."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    db.ensure_session(parent_key)

    with patch.object(SubagentManager, "_build_subagent_agent", return_value=_make_subagent_mock()):
        result = await mgr.spawn(
            task="Do something",
            parent_key=parent_key,
            label="DB Row Test",
        )

    assert "started" in result
    rows = db.list_subagent_sessions(parent_key=parent_key)
    assert len(rows) == 1
    assert rows[0].label == "db-row-test"  # label is slugified
    assert rows[0].status == "running"


@pytest.mark.asyncio
async def test_spawn_returns_error_for_duplicate_label_on_same_parent(tmp_path: Path) -> None:
    """Spawning twice with same label on same parent returns error for second."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    db.ensure_session(parent_key)

    with patch.object(SubagentManager, "_build_subagent_agent", return_value=_make_subagent_mock()):
        result1 = await mgr.spawn(task="Research task", parent_key=parent_key, label="research")
        result2 = await mgr.spawn(
            task="Another research task", parent_key=parent_key, label="research"
        )

    assert "started" in result1
    assert "already in use" in result2

    rows = db.list_subagent_sessions(parent_key=parent_key)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_spawn_returns_error_for_duplicate_label(tmp_path: Path) -> None:
    """spawn() returns error message when label already exists in DB."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    db.ensure_session(parent_key)

    # Pre-create a subagent entry
    db.create_subagent_session(
        id=str(uuid.uuid7()),
        parent_key=parent_key,
        label="duplicate-test",
        task="Already exists",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    with patch.object(SubagentManager, "_build_subagent_agent", return_value=_make_subagent_mock()):
        result = await mgr.spawn(
            task="New task with same label",
            parent_key=parent_key,
            label="duplicate-test",
        )

    assert "already in use" in result


@pytest.mark.asyncio
async def test_spawn_accepts_parent_key_parameter(tmp_path: Path) -> None:
    """spawn() correctly uses parent_key for label uniqueness scope."""
    mgr, db = _make_manager(tmp_path)

    db.ensure_session("parent-a")
    db.ensure_session("parent-b")

    with patch.object(SubagentManager, "_build_subagent_agent", return_value=_make_subagent_mock()):
        # Same label "shared" for different parents should work
        result1 = await mgr.spawn(task="Task A", parent_key="parent-a", label="shared")
        result2 = await mgr.spawn(task="Task B", parent_key="parent-b", label="shared")

    assert "started" in result1
    assert "started" in result2

    rows_a = db.list_subagent_sessions(parent_key="parent-a")
    rows_b = db.list_subagent_sessions(parent_key="parent-b")
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0].label == "shared"
    assert rows_b[0].label == "shared"


@pytest.mark.asyncio
async def test_cancel_all_cancels_all_running_tasks(tmp_path: Path) -> None:
    """cancel_all() cancels all running subagent tasks."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"
    cancelled_events: list[asyncio.Event] = []

    async def slow_task(idx: int):
        evt = asyncio.Event()
        cancelled_events.append(evt)
        evt.set()  # Signal task started
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    db.ensure_session(parent_key)

    # Create multiple running tasks
    for i in range(3):
        id = str(uuid.uuid7())
        db.create_subagent_session(
            id=id,
            parent_key=parent_key,
            label=f"CancelAll{i}",
            task=f"Task {i}",
            origin_channel="cli",
            origin_chat_id="direct",
        )
        task = asyncio.create_task(slow_task(i))
        mgr._running_tasks[id] = task

    # Wait for tasks to start
    for evt in cancelled_events:
        await asyncio.wait_for(evt.wait(), timeout=1.0)

    result = await mgr.cancel_all()

    assert result == 3
    assert len(mgr._running_tasks) == 0

    # Cleanup - tasks should be cancelled
    for tid, task in {}.items():  # _running_tasks is already cleared
        pass  # No cleanup needed since cancel_all already gathered them


@pytest.mark.asyncio
async def test_list_subagents_returns_db_rows(tmp_path: Path) -> None:
    """list_subagents() returns SubagentSessionRow objects from DB."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    db.ensure_session(parent_key)

    id1 = str(uuid.uuid7())
    id2 = str(uuid.uuid7())
    db.create_subagent_session(
        id=id1,
        parent_key=parent_key,
        label="LR1",
        task="Task 1",
        origin_channel="cli",
        origin_chat_id="direct",
    )
    db.create_subagent_session(
        id=id2,
        parent_key=parent_key,
        label="LR2",
        task="Task 2",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    result = mgr.list_subagents(parent_key=parent_key)

    assert len(result) == 2
    assert all(isinstance(row, SubagentSessionRow) for row in result)
    assert {row.id for row in result} == {id1, id2}


@pytest.mark.asyncio
async def test_list_subagents_filter_by_parent_key(tmp_path: Path) -> None:
    """list_subagents(parent_key) filters correctly."""
    mgr, db = _make_manager(tmp_path)

    db.ensure_session("filter-p1")
    db.ensure_session("filter-p2")

    db.create_subagent_session(
        id=str(uuid.uuid7()),
        parent_key="filter-p1",
        label="F1",
        task="T1",
        origin_channel="cli",
        origin_chat_id="direct",
    )
    db.create_subagent_session(
        id=str(uuid.uuid7()),
        parent_key="filter-p2",
        label="F2",
        task="T2",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    result = mgr.list_subagents(parent_key="filter-p1")

    assert len(result) == 1
    assert result[0].label == "F1"


@pytest.mark.asyncio
async def test_get_by_id_queries_db(tmp_path: Path) -> None:
    """get_by_id(id) queries DB correctly."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    db.ensure_session(parent_key)

    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key=parent_key,
        label="Query Test",
        task="Task for query",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    result = mgr.get_by_id(id)

    assert result is not None
    assert isinstance(result, SubagentSessionRow)
    assert result.id == id
    assert result.label == "Query Test"


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_unknown(tmp_path: Path) -> None:
    """get_by_id returns None for unknown id."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    db.ensure_session(parent_key)

    result = mgr.get_by_id(str(uuid.uuid7()))

    assert result is None


@pytest.mark.asyncio
async def test_kill_by_id_updates_db_status(tmp_path: Path) -> None:
    """kill_by_id updates the DB row status to 'cancelled'."""
    mgr, db = _make_manager(tmp_path)
    parent_key = "cli:direct"

    async def slow_task():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    db.ensure_session(parent_key)

    id = str(uuid.uuid7())
    db.create_subagent_session(
        id=id,
        parent_key=parent_key,
        label="KillDB",
        task="Slow task",
        origin_channel="cli",
        origin_chat_id="direct",
    )

    task = asyncio.create_task(slow_task())
    await asyncio.sleep(0.01)
    mgr._running_tasks[id] = task

    result = await mgr.kill_by_id(id)

    assert result is True

    row = db.get_subagent_session(id)
    assert row is not None
    assert row.status == "cancelled"

    # Cleanup
    try:
        await task
    except asyncio.CancelledError:
        pass
