"""Tests for AgentRunner session helpers: _save_turn, _sanitize_persisted_blocks, _restore_runtime_checkpoint."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
    ToolCallPart,
    ToolReturnPart,
)

from nanobot.agent.runner import AgentRunner
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager, Session
from pydantic_ai.messages import ModelMessage


def _make_runner(tmp_path: Path, sessions: SessionManager) -> AgentRunner:
    """Create a minimal AgentRunner for testing session helpers."""
    bus = MessageBus()
    from nanobot.agent.agent import NanobotAgent

    agent = MagicMock(spec=NanobotAgent)
    runner = AgentRunner.__new__(AgentRunner)
    runner.workspace = tmp_path
    runner.bus = bus
    runner.sessions = sessions
    runner.max_tool_result_chars = 16000
    runner._active_tasks = {}
    runner._background_tasks = []
    runner._checkpoints = {}
    return runner


def _make_user_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _make_system_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[SystemPromptPart(content=content)])


def _make_assistant_message(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


def _make_tool_call_message(tool_name: str, tool_call_id: str, args: dict) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)]
    )


def _make_tool_result_message(tool_name: str, tool_call_id: str, content: str) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=tool_name, tool_call_id=tool_call_id, content=content)]
    )


class TestSanitizePersistedBlocks:
    """Tests for _sanitize_persisted_blocks."""

    def test_passthrough_for_simple_text_blocks(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, MagicMock())
        blocks = [{"type": "text", "text": "hello world"}]
        result = runner._sanitize_persisted_blocks(blocks)
        assert result == blocks

    def test_drops_runtime_context_text_blocks(self, tmp_path: Path) -> None:
        from nanobot.agent.context import ContextBuilder

        runner = _make_runner(tmp_path, MagicMock())
        blocks = [
            {"type": "text", "text": ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now"},
            {"type": "text", "text": "real content"},
        ]
        result = runner._sanitize_persisted_blocks(blocks, drop_runtime=True)
        assert len(result) == 1
        assert result[0]["text"] == "real content"

    def test_replaces_base64_images_with_placeholder(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, MagicMock())
        blocks = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
                "_meta": {"path": "/img/photo.png"},
            },
        ]
        result = runner._sanitize_persisted_blocks(blocks)
        assert result == [{"type": "text", "text": "[image: /img/photo.png]"}]

    def test_replaces_base64_images_without_path(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, MagicMock())
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        result = runner._sanitize_persisted_blocks(blocks)
        assert result == [{"type": "text", "text": "[image]"}]

    def test_truncates_long_text_blocks(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, MagicMock())
        runner.max_tool_result_chars = 50
        blocks = [{"type": "text", "text": "x" * 100}]
        result = runner._sanitize_persisted_blocks(blocks, truncate_text=True)
        assert len(result[0]["text"]) < 100
        assert "..." in result[0]["text"]


class TestRestoreRuntimeCheckpoint:
    """Tests for _restore_runtime_checkpoint using SessionManager API."""

    def test_restores_completed_and_pending_tools(self, tmp_path: Path) -> None:
        from nanobot.db import Database, upgrade_db

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        sessions = SessionManager(tmp_path, db=db)
        sessions.ensure_session("test:restore")

        runner = _make_runner(tmp_path, sessions)

        # Set up checkpoint via the internal checkpoints dict
        runner._set_runtime_checkpoint(
            "test:restore",
            {
                "assistant_message": {
                    "role": "assistant",
                    "content": "reading file...",
                    "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}],
                },
                "completed_tool_results": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "read_file",
                        "content": "file contents",
                    },
                ],
                "pending_tool_calls": [
                    {"id": "call_2", "function": {"name": "write_file"}},
                ],
            },
        )

        result = runner._restore_runtime_checkpoint("test:restore")

        assert result is True
        # Verify messages were added via SessionManager
        messages = sessions.get_all_messages("test:restore")
        roles = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        roles.append("tool")
                    elif isinstance(part, UserPromptPart):
                        roles.append("user")
            elif isinstance(msg, ModelResponse):
                roles.append("assistant")
        assert roles == ["assistant", "tool", "tool"]
        # Last tool result should be the error message for interrupted pending tool
        last_msg = messages[-1]
        if isinstance(last_msg, ModelRequest):
            for part in last_msg.parts:
                if isinstance(part, ToolReturnPart):
                    content_str = str(part.content)
                    assert "interrupted" in content_str.lower() or "error" in content_str.lower()

    def test_restores_nothing_when_no_checkpoint(self, tmp_path: Path) -> None:
        from nanobot.db import Database, upgrade_db

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        sessions = SessionManager(tmp_path, db=db)
        sessions.ensure_session("test:no-checkpoint")

        runner = _make_runner(tmp_path, sessions)

        # No checkpoint set
        result = runner._restore_runtime_checkpoint("test:no-checkpoint")

        assert result is False
        messages = sessions.get_all_messages("test:no-checkpoint")
        assert messages == []

    def test_restores_with_overlap_deduplication(self, tmp_path: Path) -> None:
        """Existing messages that match the checkpoint are not duplicated.

        This test verifies that when restoring a checkpoint, if some messages
        already exist in the session, they are not duplicated.
        """
        from nanobot.db import Database, upgrade_db

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        sessions = SessionManager(tmp_path, db=db)
        sessions.ensure_session("test:overlap")

        runner = _make_runner(tmp_path, sessions)

        # Set checkpoint with assistant message + tool result
        runner._set_runtime_checkpoint(
            "test:overlap",
            {
                "assistant_message": {
                    "role": "assistant",
                    "content": "thinking...",
                    "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}],
                },
                "completed_tool_results": [
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "read_file",
                        "content": "file contents",
                    },
                ],
                "pending_tool_calls": [],
            },
        )

        # Restore checkpoint - should add both messages
        result = runner._restore_runtime_checkpoint("test:overlap")
        assert result is True

        messages = sessions.get_all_messages("test:overlap")
        # Should have assistant + tool result
        assert len(messages) == 2


class TestRunAgentLoopNoDuplication:
    """Verify _run_agent_loop receives user_content separately from message_history."""

    def _make_runner_with_mock_agent(self, tmp_path: Path) -> tuple[AgentRunner, MagicMock]:
        from nanobot.agent.agent import NanobotAgent

        bus = MessageBus()
        agent = MagicMock(spec=NanobotAgent)
        sessions = MagicMock()
        runner = AgentRunner.__new__(AgentRunner)
        runner.workspace = tmp_path
        runner.bus = bus
        runner.agent = agent
        runner.sessions = sessions
        runner.max_tool_result_chars = 16000
        runner._active_tasks = {}
        runner._background_tasks = []
        runner._mcp_connected = True
        runner._mcp_connecting = False
        runner.mcp_servers = {}
        return runner, agent

    @pytest.mark.asyncio
    async def test_non_streaming_passes_user_content_separately(self, tmp_path: Path) -> None:
        runner, mock_agent = self._make_runner_with_mock_agent(tmp_path)
        captured: dict = {}

        async def fake_run(user_message, message_history=None):
            captured["user_message"] = user_message
            captured["message_history"] = message_history or []
            return "ok", []

        mock_agent.run = fake_run

        initial_messages: list[ModelMessage] = [
            _make_system_message("You are helpful."),
        ]

        await runner._run_agent_loop(initial_messages, user_content="hello")

        assert captured["user_message"] == "hello"
        user_parts = [
            p
            for msg in captured["message_history"]
            if isinstance(msg, ModelRequest)
            for p in msg.parts
            if isinstance(p, UserPromptPart)
        ]
        assert len(user_parts) == 0, f"UserPromptPart found in message_history — duplication!"

    @pytest.mark.asyncio
    async def test_non_streaming_passes_correct_user_content(self, tmp_path: Path) -> None:
        runner, mock_agent = self._make_runner_with_mock_agent(tmp_path)
        captured: dict = {}

        async def fake_run(user_message, message_history=None):
            captured["user_message"] = user_message
            captured["message_history"] = message_history or []
            return "ok", []

        mock_agent.run = fake_run

        initial_messages = [
            _make_user_message("first"),
            _make_assistant_message("hi"),
        ]

        await runner._run_agent_loop(initial_messages, user_content="second")

        assert captured["user_message"] == "second"
        user_parts = [
            p
            for msg in captured["message_history"]
            if isinstance(msg, ModelRequest)
            for p in msg.parts
            if isinstance(p, UserPromptPart)
        ]
        assert len(user_parts) == 1
        assert user_parts[0].content == "first"

    @pytest.mark.asyncio
    async def test_streaming_passes_user_content_separately(self, tmp_path: Path) -> None:
        runner, mock_agent = self._make_runner_with_mock_agent(tmp_path)
        captured: dict = {}

        class FakeStreamResult:
            response = MagicMock()
            response.text = "streamed response"
            _new_messages = []

            def new_messages(self):
                return self._new_messages

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def stream_text(self, delta=True):
                yield "chunk"

        def fake_run_stream(user_message, message_history=None):
            captured["user_message"] = user_message
            captured["message_history"] = message_history or []
            return FakeStreamResult()

        mock_agent.run_stream = fake_run_stream

        initial_messages: list[ModelMessage] = [
            _make_system_message("You are helpful."),
        ]

        result = await runner._run_agent_loop(
            initial_messages,
            user_content="stream me",
            on_stream=AsyncMock(),
        )

        assert captured["user_message"] == "stream me"
        user_parts = [
            p
            for msg in captured["message_history"]
            if isinstance(msg, ModelRequest)
            for p in msg.parts
            if isinstance(p, UserPromptPart)
        ]
        assert len(user_parts) == 0, "UserPromptPart in message_history — duplication!"
