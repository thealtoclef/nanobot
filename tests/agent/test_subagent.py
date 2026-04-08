"""Tests for SubagentManager.cancel_by_session and message duplication fix."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai.messages import ModelRequest, UserPromptPart

from nanobot.agent.subagent import SubagentManager


def _make_subagent_mock() -> MagicMock:
    subagent = MagicMock()
    subagent.run = AsyncMock(return_value=('{"action": "skip"}', []))
    return subagent


def _make_manager(tmp_path: Path) -> SubagentManager:
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    agent = MagicMock()
    agent.models = []
    subagent_mock = _make_subagent_mock()

    with patch.object(SubagentManager, "_build_subagent_agent", return_value=subagent_mock):
        mgr = SubagentManager(
            agent=agent,
            workspace=tmp_path,
            bus=bus,
            max_tool_result_chars=16000,
        )
    return mgr


@pytest.mark.asyncio
async def test_cancel_by_session_returns_zero_when_no_tasks(tmp_path: Path) -> None:
    """cancel_by_session returns 0 when no tasks exist for that session."""
    mgr = _make_manager(tmp_path)

    result = await mgr.cancel_by_session("nonexistent-session")

    assert result == 0


@pytest.mark.asyncio
async def test_cancel_by_session_returns_zero_when_session_has_no_tasks(tmp_path: Path) -> None:
    """Returns 0 when session exists but has no running tasks."""
    mgr = _make_manager(tmp_path)
    mgr._session_tasks["test:c1"] = set()

    result = await mgr.cancel_by_session("test:c1")

    assert result == 0


@pytest.mark.asyncio
async def test_cancel_by_session_cancels_active_subagent(tmp_path: Path) -> None:
    """cancel_by_session cancels running subagent tasks for the session."""
    mgr = _make_manager(tmp_path)

    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def slow_task():
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(slow_task())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    mgr._running_tasks["sub-1"] = task
    mgr._session_tasks["test:c1"] = {"sub-1"}

    result = await mgr.cancel_by_session("test:c1")

    assert result == 1
    assert cancelled.is_set()
    assert task.cancelled()
    # Cleanup
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_cancel_by_session_ignores_already_done_tasks(tmp_path: Path) -> None:
    """Already-completed tasks are not counted as cancelled."""
    mgr = _make_manager(tmp_path)

    async def already_done():
        return

    done_task = asyncio.create_task(already_done())
    await done_task  # ensure task is truly done

    mgr._running_tasks["sub-1"] = done_task
    mgr._session_tasks["test:c1"] = {"sub-1"}

    result = await mgr.cancel_by_session("test:c1")

    assert result == 0


@pytest.mark.asyncio
async def test_run_subagent_no_duplication(tmp_path: Path) -> None:
    captured: dict = {}

    async def fake_run(user_message, message_history=None):
        captured["user_message"] = user_message
        captured["message_history"] = message_history or []
        return '{"action": "skip"}', []

    subagent_mock = MagicMock()
    subagent_mock.run = fake_run

    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.publish_inbound = AsyncMock()

    agent = MagicMock()
    agent.models = []

    with patch.object(SubagentManager, "_build_subagent_agent", return_value=subagent_mock):
        mgr = SubagentManager(
            agent=agent,
            workspace=tmp_path,
            bus=bus,
            max_tool_result_chars=16000,
        )

    await mgr._run_subagent(
        task_id="sub-dedup-test",
        task="do something important",
        label="dedup test",
        origin={"channel": "cli", "chat_id": "direct"},
    )

    assert captured["user_message"] == "do something important"
    assert captured["message_history"] == []
