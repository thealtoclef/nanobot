"""Tests for AgentRunner session helpers: _save_turn, _sanitize_persisted_blocks, _restore_runtime_checkpoint."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from nanobot.agent.runner import AgentRunner
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import Session


def _make_runner(tmp_path: Path) -> AgentRunner:
    """Create a minimal AgentRunner for testing session helpers."""
    bus = MessageBus()
    from nanobot.agent.agent import NanobotAgent

    agent = MagicMock(spec=NanobotAgent)
    runner = AgentRunner.__new__(AgentRunner)
    runner.workspace = tmp_path
    runner.bus = bus
    runner.sessions = runner.sessions = MagicMock()
    runner.max_tool_result_chars = 16000
    runner._active_tasks = {}
    runner._background_tasks = []
    return runner


class TestSaveTurn:
    """Tests for _save_turn and _sanitize_persisted_blocks."""

    def test_save_turn_skips_runtime_context_only_user_messages(self, tmp_path: Path) -> None:
        from nanobot.agent.context import ContextBuilder

        runner = _make_runner(tmp_path)
        session = Session(key="test:runtime-only")
        runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\n\n"

        new_messages = [ModelRequest(parts=[UserPromptPart(content=runtime)])]
        runner._save_turn(session, new_messages)
        assert session.messages == []

    def test_save_turn_keeps_user_message_after_runtime_strip(self, tmp_path: Path) -> None:
        from nanobot.agent.context import ContextBuilder

        runner = _make_runner(tmp_path)
        session = Session(key="test:user-after-runtime")
        runtime = ContextBuilder._RUNTIME_CONTEXT_TAG + "\n\nHello, how are you?"

        new_messages = [ModelRequest(parts=[UserPromptPart(content=runtime)])]
        runner._save_turn(session, new_messages)
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "Hello, how are you?"

    def test_save_turn_keeps_image_placeholder_with_path(self, tmp_path: Path) -> None:
        from nanobot.agent.context import ContextBuilder

        runner = _make_runner(tmp_path)
        session = Session(key="test:image-placeholder")

        new_messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            {
                                "type": "text",
                                "text": ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc"},
                                "_meta": {"path": "/media/feishu/photo.jpg"},
                            },
                        ]
                    )
                ]
            )
        ]
        runner._save_turn(session, new_messages)
        assert len(session.messages) == 1
        content = session.messages[0]["content"]
        assert isinstance(content, list)
        assert {"type": "text", "text": "[image: /media/feishu/photo.jpg]"} in content

    def test_save_turn_keeps_image_placeholder_without_meta(self, tmp_path: Path) -> None:
        from nanobot.agent.context import ContextBuilder

        runner = _make_runner(tmp_path)
        session = Session(key="test:image-no-meta")

        new_messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            {
                                "type": "text",
                                "text": ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc"},
                            },
                        ]
                    )
                ]
            )
        ]
        runner._save_turn(session, new_messages)
        assert len(session.messages) == 1
        content = session.messages[0]["content"]
        assert isinstance(content, list)
        assert {"type": "text", "text": "[image]"} in content

    def test_save_turn_truncates_long_tool_result(self, tmp_path: Path) -> None:
        from pydantic_ai.messages import ToolReturnPart

        runner = _make_runner(tmp_path)
        runner.max_tool_result_chars = 100
        session = Session(key="test:tool-long")

        content = "x" * 200

        new_messages = [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="read_file", tool_call_id="call_1", content=content)
                ]
            )
        ]
        runner._save_turn(session, new_messages)
        assert len(session.messages[0]["content"]) < 200
        assert "..." in session.messages[0]["content"]

    def test_save_turn_skips_empty_assistant_messages(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        session = Session(key="test:empty-assistant")

        new_messages = [ModelResponse(parts=[TextPart(content="")])]
        runner._save_turn(session, new_messages)
        assert session.messages == []

    def test_save_turn_adds_timestamp(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        session = Session(key="test:timestamp")

        new_messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        runner._save_turn(session, new_messages)
        assert "timestamp" in session.messages[0]


class TestSanitizePersistedBlocks:
    """Tests for _sanitize_persisted_blocks."""

    def test_passthrough_for_simple_text_blocks(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        blocks = [{"type": "text", "text": "hello world"}]
        result = runner._sanitize_persisted_blocks(blocks)
        assert result == blocks

    def test_drops_runtime_context_text_blocks(self, tmp_path: Path) -> None:
        from nanobot.agent.context import ContextBuilder

        runner = _make_runner(tmp_path)
        blocks = [
            {"type": "text", "text": ContextBuilder._RUNTIME_CONTEXT_TAG + "\nCurrent Time: now"},
            {"type": "text", "text": "real content"},
        ]
        result = runner._sanitize_persisted_blocks(blocks, drop_runtime=True)
        assert len(result) == 1
        assert result[0]["text"] == "real content"

    def test_replaces_base64_images_with_placeholder(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
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
        runner = _make_runner(tmp_path)
        blocks = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
        result = runner._sanitize_persisted_blocks(blocks)
        assert result == [{"type": "text", "text": "[image]"}]

    def test_truncates_long_text_blocks(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        runner.max_tool_result_chars = 50
        blocks = [{"type": "text", "text": "x" * 100}]
        result = runner._sanitize_persisted_blocks(blocks, truncate_text=True)
        assert len(result[0]["text"]) < 100
        assert "..." in result[0]["text"]


class TestRestoreRuntimeCheckpoint:
    """Tests for _restore_runtime_checkpoint."""

    def test_restores_completed_and_pending_tools(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        session = Session(key="test:restore")
        session.metadata["runtime_checkpoint"] = {
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
        }

        result = runner._restore_runtime_checkpoint(session)

        assert result is True
        roles = [m["role"] for m in session.messages]
        assert roles == ["assistant", "tool", "tool"]
        assert (
            session.messages[2]["content"] == "Error: Task interrupted before this tool finished."
        )

    def test_restores_nothing_when_no_checkpoint(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        session = Session(key="test:no-checkpoint")
        session.metadata["runtime_checkpoint"] = None

        result = runner._restore_runtime_checkpoint(session)

        assert result is False
        assert session.messages == []

    def test_restores_with_overlap_deduplication(self, tmp_path: Path) -> None:
        """Existing messages that match the checkpoint are not duplicated."""
        runner = _make_runner(tmp_path)
        session = Session(key="test:overlap")
        session.messages = [
            {
                "role": "assistant",
                "content": "reading file...",
                "tool_calls": [{"id": "call_1", "function": {"name": "read_file"}}],
            },
        ]
        session.metadata["runtime_checkpoint"] = {
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
            "pending_tool_calls": [],
        }

        result = runner._restore_runtime_checkpoint(session)

        assert result is True
        # Should not duplicate the already-existing assistant message
        assert len(session.messages) == 2  # original assistant + tool result


class TestRunAgentLoopNoDuplication:
    """Verify _run_agent_loop receives user_content separately from message_history."""

    def _make_runner_with_mock_agent(self, tmp_path: Path) -> tuple[AgentRunner, MagicMock]:
        from nanobot.agent.agent import NanobotAgent

        bus = MessageBus()
        agent = MagicMock(spec=NanobotAgent)
        runner = AgentRunner.__new__(AgentRunner)
        runner.workspace = tmp_path
        runner.bus = bus
        runner.agent = agent
        runner.sessions = MagicMock()
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

        initial_messages = [
            {"role": "system", "content": "You are helpful."},
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
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "hi"},
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

        initial_messages = [
            {"role": "system", "content": "You are helpful."},
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
