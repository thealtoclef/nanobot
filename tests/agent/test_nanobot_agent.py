"""Tests for NanobotAgent, ToolAdapter, build_instructions, and message conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from nanobot.agent.agent import (
    BOOTSTRAP_FILES,
    NanobotAgent,
    ToolAdapter,
    build_instructions,
    session_messages_to_model_messages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeTool:
    """Minimal tool-like object for ToolAdapter tests."""

    def __init__(self, name: str = "test_tool", description: str = "A test tool"):
        self.name = name
        self.description = description
        self.execute = AsyncMock(return_value="tool result")


class FakePydanticAgent:
    """Stub PydanticAI Agent that records tool registrations."""

    def __init__(self) -> None:
        self._registered_tools: list[dict[str, Any]] = []

    def tool(self, name: str, description: str):
        """Record tool decorator call."""

        def decorator(func):
            self._registered_tools.append({"name": name, "description": description, "func": func})
            return func

        return decorator


def _make_session(
    messages: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Create a fake session with get_history returning the given messages."""
    session = MagicMock()
    session.get_history = MagicMock(return_value=messages or [])
    session.add_message = MagicMock()
    return session


def _make_model(name: str = "test-model") -> MagicMock:
    m = MagicMock()
    m.model_name = name
    return m


def _make_nanobot_agent_with_testmodel(
    tmp_path: Path,
    *,
    custom_output: str | None = None,
) -> tuple[NanobotAgent, TestModel]:
    """Create a NanobotAgent whose internal pydantic Agent uses TestModel.

    Patches FallbackModel and Agent just enough for construction, then
    replaces the internal agent with a real pydantic Agent backed by TestModel.
    """
    tm = TestModel(custom_output_text=custom_output) if custom_output else TestModel()
    real_agent = PydanticAIAgent(model=tm, instructions="test instructions")

    with (
        patch("pydantic_ai.models.fallback.FallbackModel") as mock_fb,
        patch("nanobot.agent.agent.Agent", return_value=real_agent),
    ):
        mock_fb.return_value = MagicMock()
        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
        )

    return agent, tm


# ---------------------------------------------------------------------------
# build_instructions
# ---------------------------------------------------------------------------


class TestBuildInstructions:
    """Tests for build_instructions()."""

    def test_includes_identity_section(self, tmp_path: Path) -> None:
        """Result always contains the nanobot identity header."""
        instructions = build_instructions(tmp_path)
        assert "You are nanobot" in instructions

    def test_includes_workspace_path(self, tmp_path: Path) -> None:
        """Result contains the resolved workspace path."""
        instructions = build_instructions(tmp_path)
        assert str(tmp_path.resolve()) in instructions

    def test_includes_bootstrap_files(self, tmp_path: Path) -> None:
        """Each existing bootstrap file is included as a section."""
        for filename in BOOTSTRAP_FILES:
            (tmp_path / filename).write_text(f"{filename} content", encoding="utf-8")

        instructions = build_instructions(tmp_path)
        for filename in BOOTSTRAP_FILES:
            assert f"## {filename}" in instructions
            assert f"{filename} content" in instructions

    def test_skips_missing_bootstrap_files(self, tmp_path: Path) -> None:
        """Missing bootstrap files are silently skipped."""
        (tmp_path / "AGENTS.md").write_text("agents content", encoding="utf-8")
        instructions = build_instructions(tmp_path)
        assert "## AGENTS.md" in instructions
        assert "## SOUL.md" not in instructions

    def test_sections_separated_by_divider(self, tmp_path: Path) -> None:
        """Sections are separated by horizontal rule dividers."""
        (tmp_path / "AGENTS.md").write_text("content", encoding="utf-8")
        instructions = build_instructions(tmp_path)
        assert "\n\n---\n\n" in instructions

    def test_empty_workspace_still_returns_identity(self, tmp_path: Path) -> None:
        """Even with no bootstrap files, identity section is returned."""
        instructions = build_instructions(tmp_path)
        assert instructions.startswith("# nanobot")


# ---------------------------------------------------------------------------
# session_messages_to_model_messages
# ---------------------------------------------------------------------------


class TestSessionMessagesToModelMessages:
    """Tests for session_messages_to_model_messages()."""

    def test_empty_list_returns_empty(self) -> None:
        """Empty input produces empty output."""
        result = session_messages_to_model_messages([])
        assert result == []

    def test_system_message(self) -> None:
        """System role produces ModelRequest with SystemPromptPart."""
        result = session_messages_to_model_messages(
            [{"role": "system", "content": "You are helpful."}]
        )
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, ModelRequest)
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], SystemPromptPart)
        assert msg.parts[0].content == "You are helpful."

    def test_user_message(self) -> None:
        """User role produces ModelRequest with UserPromptPart."""
        result = session_messages_to_model_messages([{"role": "user", "content": "Hello"}])
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, ModelRequest)
        assert isinstance(msg.parts[0], UserPromptPart)
        assert msg.parts[0].content == "Hello"

    def test_assistant_text_only(self) -> None:
        """Assistant without tool calls produces ModelResponse with TextPart."""
        result = session_messages_to_model_messages([{"role": "assistant", "content": "Hi there!"}])
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, ModelResponse)
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].content == "Hi there!"

    def test_assistant_with_tool_calls(self) -> None:
        """Assistant with tool_calls produces ModelResponse with TextPart + ToolCallParts."""
        result = session_messages_to_model_messages(
            [
                {
                    "role": "assistant",
                    "content": "Let me read that.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "name": "read_file",
                            "arguments": {"path": "/tmp/f.txt"},
                        }
                    ],
                }
            ]
        )
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, ModelResponse)
        # Should have TextPart + ToolCallPart
        assert len(msg.parts) == 2
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].content == "Let me read that."
        assert isinstance(msg.parts[1], ToolCallPart)
        assert msg.parts[1].tool_name == "read_file"
        assert msg.parts[1].tool_call_id == "call_1"

    def test_assistant_tool_calls_with_nested_function_name(self) -> None:
        """Tool call name extraction works with nested function.name format."""
        result = session_messages_to_model_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {"name": "exec"},
                            "arguments": {"command": "ls"},
                        }
                    ],
                }
            ]
        )
        msg = result[0]
        assert isinstance(msg, ModelResponse)
        # No content text → no TextPart
        tc_part = msg.parts[0]
        assert isinstance(tc_part, ToolCallPart)
        assert tc_part.tool_name == "exec"

    def test_assistant_tool_calls_no_content(self) -> None:
        """Assistant with tool_calls but empty content has only ToolCallParts."""
        result = session_messages_to_model_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_3", "name": "web_search", "arguments": {"q": "test"}},
                    ],
                }
            ]
        )
        msg = result[0]
        assert isinstance(msg, ModelResponse)
        # No TextPart since content is empty/falsy
        text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
        assert len(text_parts) == 0
        tc_parts = [p for p in msg.parts if isinstance(p, ToolCallPart)]
        assert len(tc_parts) == 1

    def test_tool_message(self) -> None:
        """Tool role produces ModelRequest with ToolReturnPart."""
        result = session_messages_to_model_messages(
            [
                {
                    "role": "tool",
                    "name": "read_file",
                    "tool_call_id": "call_1",
                    "content": "file contents here",
                }
            ]
        )
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, ModelRequest)
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], ToolReturnPart)
        assert msg.parts[0].tool_name == "read_file"
        assert msg.parts[0].tool_call_id == "call_1"
        assert msg.parts[0].content == "file contents here"

    def test_unknown_role_skipped(self) -> None:
        """Messages with unrecognized roles are silently skipped."""
        result = session_messages_to_model_messages(
            [
                {"role": "unknown", "content": "???"},
            ]
        )
        assert result == []

    def test_missing_role_skipped(self) -> None:
        """Messages without a role key are skipped."""
        result = session_messages_to_model_messages(
            [
                {"content": "no role"},
            ]
        )
        assert result == []

    def test_full_conversation_round_trip(self) -> None:
        """A typical conversation sequence converts correctly."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Read foo.txt"},
            {
                "role": "assistant",
                "content": "Reading file.",
                "tool_calls": [
                    {"id": "c1", "name": "read_file", "arguments": {"path": "foo.txt"}},
                ],
            },
            {"role": "tool", "name": "read_file", "tool_call_id": "c1", "content": "hello"},
            {"role": "assistant", "content": "The file contains 'hello'."},
        ]
        result = session_messages_to_model_messages(messages)
        assert len(result) == 5
        assert isinstance(result[0], ModelRequest)  # system
        assert isinstance(result[1], ModelRequest)  # user
        assert isinstance(result[2], ModelResponse)  # assistant + tool_calls
        assert isinstance(result[3], ModelRequest)  # tool result
        assert isinstance(result[4], ModelResponse)  # assistant final

    def test_multiple_tool_calls(self) -> None:
        """Multiple tool calls in one assistant message become multiple ToolCallParts."""
        result = session_messages_to_model_messages(
            [
                {
                    "role": "assistant",
                    "content": "Running tools.",
                    "tool_calls": [
                        {"id": "c1", "name": "read_file", "arguments": {"path": "a.txt"}},
                        {"id": "c2", "name": "exec", "arguments": {"command": "ls"}},
                    ],
                }
            ]
        )
        msg = result[0]
        tc_parts = [p for p in msg.parts if isinstance(p, ToolCallPart)]
        assert len(tc_parts) == 2
        assert tc_parts[0].tool_name == "read_file"
        assert tc_parts[1].tool_name == "exec"

    def test_tool_message_defaults(self) -> None:
        """Tool message with missing name/tool_call_id uses empty strings."""
        result = session_messages_to_model_messages(
            [
                {"role": "tool", "content": "some result"},
            ]
        )
        part = result[0].parts[0]
        assert isinstance(part, ToolReturnPart)
        assert part.tool_name == ""
        assert part.tool_call_id == ""

    def test_missing_content_defaults_empty(self) -> None:
        """Messages without content default to empty string."""
        result = session_messages_to_model_messages(
            [
                {"role": "user"},
            ]
        )
        part = result[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == ""


# ---------------------------------------------------------------------------
# ToolAdapter
# ---------------------------------------------------------------------------


class TestToolAdapter:
    """Tests for ToolAdapter."""

    def test_register_creates_tool_on_agent(self) -> None:
        fake_agent = FakePydanticAgent()
        adapter = ToolAdapter(fake_agent)  # type: ignore[arg-type]
        tool = FakeTool(name="my_tool", description="Does things")

        adapter.register(tool)

        assert len(fake_agent._registered_tools) == 1
        assert fake_agent._registered_tools[0]["name"] == "my_tool"
        assert fake_agent._registered_tools[0]["description"] == "Does things"

    def test_register_tracks_tool_instance(self) -> None:
        fake_agent = FakePydanticAgent()
        adapter = ToolAdapter(fake_agent)  # type: ignore[arg-type]
        tool = FakeTool()

        adapter.register(tool)

        assert tool in adapter._tools

    @pytest.mark.asyncio
    async def test_bound_tool_calls_execute(self) -> None:
        fake_agent = FakePydanticAgent()
        adapter = ToolAdapter(fake_agent)  # type: ignore[arg-type]
        tool = FakeTool()
        tool.execute.return_value = "executed!"

        adapter.register(tool)

        bound_fn = fake_agent._registered_tools[0]["func"]
        ctx = MagicMock()
        result = await bound_fn(ctx, path="/tmp/test")
        assert result == "executed!"
        tool.execute.assert_awaited_once_with(path="/tmp/test")

    @pytest.mark.asyncio
    async def test_bound_tool_truncates_long_results(self) -> None:
        fake_agent = FakePydanticAgent()
        adapter = ToolAdapter(fake_agent, max_result_chars=20)  # type: ignore[arg-type]
        tool = FakeTool()
        tool.execute.return_value = "A" * 100

        adapter.register(tool)

        bound_fn = fake_agent._registered_tools[0]["func"]
        ctx = MagicMock()
        result = await bound_fn(ctx)
        assert len(result) < 100
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_bound_tool_short_result_unchanged(self) -> None:
        fake_agent = FakePydanticAgent()
        adapter = ToolAdapter(fake_agent, max_result_chars=1000)  # type: ignore[arg-type]
        tool = FakeTool()
        tool.execute.return_value = "short result"

        adapter.register(tool)

        bound_fn = fake_agent._registered_tools[0]["func"]
        ctx = MagicMock()
        result = await bound_fn(ctx)
        assert result == "short result"

    @pytest.mark.asyncio
    async def test_bound_tool_non_string_result_passes_through(self) -> None:
        fake_agent = FakePydanticAgent()
        adapter = ToolAdapter(fake_agent, max_result_chars=5)  # type: ignore[arg-type]
        tool = FakeTool()
        tool.execute.return_value = ["a", "long", "list", "of", "items"]

        adapter.register(tool)

        bound_fn = fake_agent._registered_tools[0]["func"]
        ctx = MagicMock()
        result = await bound_fn(ctx)
        assert result == ["a", "long", "list", "of", "items"]

    def test_register_multiple_tools(self) -> None:
        fake_agent = FakePydanticAgent()
        adapter = ToolAdapter(fake_agent)  # type: ignore[arg-type]
        tool_a = FakeTool(name="tool_a")
        tool_b = FakeTool(name="tool_b")

        adapter.register(tool_a)
        adapter.register(tool_b)

        assert len(fake_agent._registered_tools) == 2
        assert len(adapter._tools) == 2


# ---------------------------------------------------------------------------
# NanobotAgent
# ---------------------------------------------------------------------------


class TestNanobotAgent:
    """Tests for NanobotAgent (construction, properties, run delegation)."""

    def test_init_with_no_models_raises(self, tmp_path: Path) -> None:
        """Initializing with empty models list raises ValueError."""
        with pytest.raises(ValueError, match="No models configured"):
            NanobotAgent(workspace=tmp_path, models=[])

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    def test_init_creates_pydantic_agent(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        """NanobotAgent.__init__ creates a pydantic Agent with FallbackModel."""
        mock_model = _make_model()
        mock_fallback.return_value = mock_model
        mock_agent_instance = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[mock_model],
        )

        mock_fallback.assert_called_once_with(mock_model)
        mock_agent_cls.assert_called_once()
        assert agent.pydantic_agent is mock_agent_instance

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    def test_init_with_custom_system_prompt(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        """Custom system_prompt overrides build_instructions."""
        mock_fallback.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
            system_prompt="Custom prompt",
        )

        call_kwargs = mock_agent_cls.call_args
        assert (
            call_kwargs.kwargs.get("instructions") == "Custom prompt"
            or call_kwargs[1].get("instructions") == "Custom prompt"
        )

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    def test_init_with_default_instructions(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        """Without system_prompt, build_instructions is used."""
        mock_fallback.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
        )

        call_kwargs = mock_agent_cls.call_args
        instructions = call_kwargs.kwargs.get("instructions") or call_kwargs[1].get("instructions")
        assert "nanobot" in instructions

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    def test_tool_adapter_lazy_init(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        """tool_adapter is created lazily on first access."""
        mock_fallback.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
        )

        assert agent._tool_adapter is None
        adapter = agent.tool_adapter
        assert isinstance(adapter, ToolAdapter)
        assert agent.tool_adapter is adapter

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    def test_pydantic_agent_property(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        """pydantic_agent property returns the underlying agent."""
        mock_fallback.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
        )

        assert agent.pydantic_agent is mock_agent_instance

    @pytest.mark.asyncio
    async def test_run_with_session_history(self, tmp_path: Path) -> None:
        agent, tm = _make_nanobot_agent_with_testmodel(tmp_path)

        session_msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        history = session_messages_to_model_messages(session_msgs)

        with agent.pydantic_agent.override(model=tm):
            output, new_msgs = await agent.run("Hello", message_history=history)

        assert output == "success (no tool calls)"
        assert isinstance(new_msgs, list)

    @pytest.mark.asyncio
    async def test_run_with_explicit_message_history(self, tmp_path: Path) -> None:
        agent, tm = _make_nanobot_agent_with_testmodel(tmp_path)

        history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="prev")])]

        with agent.pydantic_agent.override(model=tm):
            output, new_msgs = await agent.run("Hello", message_history=history)

        assert output == "success (no tool calls)"
        assert isinstance(new_msgs, list)
        assert len(new_msgs) > 0

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    @pytest.mark.asyncio
    async def test_run_returns_empty_string_for_none_output(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        mock_fallback.return_value = MagicMock()
        mock_pydantic_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.output = None
        mock_pydantic_agent.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_pydantic_agent

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
        )

        output, _ = await agent.run("Hello")
        assert output == ""

    @pytest.mark.asyncio
    async def test_run_without_session(self, tmp_path: Path) -> None:
        agent, tm = _make_nanobot_agent_with_testmodel(tmp_path)

        with agent.pydantic_agent.override(model=tm):
            result, new_msgs = await agent.run("Hello")

        assert result == "success (no tool calls)"
        assert isinstance(new_msgs, list)

    @pytest.mark.asyncio
    async def test_run_stream_delegates(self, tmp_path: Path) -> None:
        agent, tm = _make_nanobot_agent_with_testmodel(tmp_path)

        with agent.pydantic_agent.override(model=tm):
            async with agent.run_stream("Hello") as stream_result:
                chunks: list[str] = []
                async for chunk in stream_result.stream_text(delta=True):
                    chunks.append(chunk)

        streamed = "".join(chunks)
        assert "success" in streamed

    def test_get_history(self, tmp_path: Path) -> None:
        agent, _ = _make_nanobot_agent_with_testmodel(tmp_path)

        session = _make_session(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )

        history = agent.get_history(session)
        assert len(history) == 2
        assert isinstance(history[0], ModelRequest)
        assert isinstance(history[1], ModelResponse)

    def test_get_history_with_max_messages(self, tmp_path: Path) -> None:
        agent, _ = _make_nanobot_agent_with_testmodel(tmp_path)

        session = _make_session([])
        agent.get_history(session, max_messages=100)
        session.get_history.assert_called_once_with(max_messages=100)

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    def test_init_with_hooks(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        """Hooks are passed through to capabilities."""
        mock_fallback.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()
        mock_hooks = MagicMock()

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
            hooks=mock_hooks,
        )

        call_kwargs = mock_agent_cls.call_args
        capabilities = call_kwargs.kwargs.get("capabilities") or call_kwargs[1].get("capabilities")
        assert mock_hooks in capabilities

    @patch("pydantic_ai.models.fallback.FallbackModel")
    @patch("nanobot.agent.agent.Agent")
    def test_init_with_retries(
        self, mock_agent_cls: MagicMock, mock_fallback: MagicMock, tmp_path: Path
    ) -> None:
        """retries parameter is passed to pydantic Agent."""
        mock_fallback.return_value = MagicMock()
        mock_agent_cls.return_value = MagicMock()

        agent = NanobotAgent(
            workspace=tmp_path,
            models=[_make_model()],
            retries=3,
        )

        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs.get("retries") == 3 or call_kwargs[1].get("retries") == 3
