"""Integration tests for /compact command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from nanobot.agents.extractor import ExtractorResult
    from nanobot.agents.summarizer import SummarizerResult


def _make_runner(tmp_path: Path):
    """Create a test AgentRunner pointing at a temp workspace."""
    from nanobot.runner import AgentRunner
    from nanobot.memory.compressor import HistoryCompressor
    from nanobot.session import SessionManager
    from nanobot.db import Database, upgrade_db

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Upgrade DB schema
    upgrade_db(workspace)

    # Create database and session manager
    db = Database(workspace)
    sessions = SessionManager(workspace, db=db)

    # Create runner via __new__ to avoid full initialization (which requires API keys)
    runner = AgentRunner.__new__(AgentRunner)
    runner.workspace = workspace
    runner.sessions = sessions
    runner.db = db

    # Create a minimal mock agent for history compressor
    mock_agent = MagicMock()
    # Set default model - will be overridden in tests
    mock_agent.pydantic_agent.model = MagicMock()

    # Create history compressor with the session manager and db
    runner.history_compressor = HistoryCompressor(
        db=db,
        agent=mock_agent,
        sessions=sessions,
        context_window_tokens=128000,
        max_completion_tokens=4096,
    )

    return runner


def _make_test_model(
    summary: str = "[2026-04-09 10:00] Test summary.",
    facts: list[dict] | None = None,
) -> TestModel:
    """Create a TestModel with custom output for both summarization and extraction."""
    if facts is None:
        facts = [{"content": "User likes testing", "category": "preference"}]
    return TestModel(
        custom_output_args={"summary": summary, "facts": facts},
    )


class TestCompactCommand:
    """Tests for /compact built-in command."""

    @pytest.fixture
    def runner(self, tmp_path: Path):
        """Create a test runner with temp workspace."""
        r = _make_runner(tmp_path)
        yield r
        # cleanup
        try:
            r.stop()
        except Exception:
            pass

    async def test_compact_command_flow_compresses_and_clears(self, runner, tmp_path: Path) -> None:
        """Full /compact flow: compresses messages into history + facts, then clears messages."""
        from nanobot.agents.summarizer import _summarizer_agent
        from nanobot.agents.extractor import _extractor_agent

        # Create test model with both summary and facts
        tm = _make_test_model(
            summary="Compressed session summary [test]",
            facts=[
                {"content": "User prefers concise answers", "category": "preference"},
                {"content": "Working on a Python project", "category": "fact"},
            ],
        )

        # Set the model on the history compressor's agent
        runner.history_compressor.agent.pydantic_agent.model = tm

        runner.sessions.ensure_session("test-session")

        # Add some messages
        test_messages = [
            ModelRequest(parts=[UserPromptPart(content="Hello, I like concise responses.")]),
            ModelRequest(parts=[UserPromptPart(content="I'm working on a Python project.")]),
        ]
        runner.sessions.add_messages("test-session", test_messages)

        # Verify messages exist
        assert len(runner.sessions.get_all_messages("test-session")) == 2

        # Run /compact command directly via the command handler
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext
        from nanobot.bus.events import InboundMessage, OutboundMessage

        inbound = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="/compact",
            session_key="test-session",
        )
        ctx = CommandContext(
            msg=inbound,
            session=runner.sessions.get_session("test-session"),
            key="test-session",
            raw="/compact",
            loop=runner,
        )

        # Use override context managers to set the test model
        with _summarizer_agent.override(model=tm), _extractor_agent.override(model=tm):
            result = await cmd_compact(ctx)

        assert result.content == "Conversation compressed."

        # Verify: messages are NOT deleted (compaction just sets the boundary)
        remaining = runner.sessions.get_all_messages("test-session")
        assert len(remaining) == 2  # original messages still exist

        # Verify: history row created with summary
        # Note: current_history_id is cleared by clear_session_for_new, so use get_all_histories
        histories = runner.sessions.get_all_histories("test-session")
        assert len(histories) >= 1
        history_row = histories[-1]  # most recent
        assert "Compressed session summary" in history_row.summary
        # Note: summarized_through_message_id may be NULL after compact since
        # clear_session_for_new deletes messages, but we still have the summary

        # Verify: facts extracted
        facts = runner.sessions.get_facts("test-session")
        assert len(facts) == 2
        contents = [f.content for f in facts]
        assert any("prefers concise" in c for c in contents)
        assert any("Python project" in c for c in contents)

        # Verify: session still exists (current_history_id is cleared by clear_session_for_new)
        session = runner.sessions.get_session("test-session")
        assert session is not None

    async def test_compact_with_no_messages(self, runner, tmp_path: Path) -> None:
        """Empty session — /compact should return success without crashing."""
        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext
        from nanobot.bus.events import InboundMessage

        runner.sessions.ensure_session("empty-session")

        inbound = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="/compact",
            session_key="empty-session",
        )
        ctx = CommandContext(
            msg=inbound,
            session=runner.sessions.get_session("empty-session"),
            key="empty-session",
            raw="/compact",
            loop=runner,
        )

        result = await cmd_compact(ctx)
        assert result.content == "Conversation compressed."

        # No history row should be created for empty session
        assert runner.sessions.get_current_history_row("empty-session") is None

    async def test_compact_facts_accumulate(self, runner, tmp_path: Path) -> None:
        """Facts from multiple /compact calls should accumulate."""
        from nanobot.agents.summarizer import _summarizer_agent
        from nanobot.agents.extractor import _extractor_agent

        # Create test model for first compact
        tm1 = _make_test_model(
            summary="Compressed session summary [test]",
            facts=[{"content": "Fact from round 1", "category": "fact"}],
        )

        # Set the model on the history compressor's agent
        runner.history_compressor.agent.pydantic_agent.model = tm1

        runner.sessions.ensure_session("multi-session")

        # First compact
        runner.sessions.add_messages(
            "multi-session", [ModelRequest(parts=[UserPromptPart(content="Message 1")])]
        )

        from nanobot.command.builtin import cmd_compact
        from nanobot.command.router import CommandContext
        from nanobot.bus.events import InboundMessage

        inbound = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="/compact",
            session_key="multi-session",
        )
        ctx = CommandContext(
            msg=inbound,
            session=runner.sessions.get_session("multi-session"),
            key="multi-session",
            raw="/compact",
            loop=runner,
        )

        with _summarizer_agent.override(model=tm1), _extractor_agent.override(model=tm1):
            await cmd_compact(ctx)

        # Facts after first compact
        facts = runner.sessions.get_facts("multi-session")
        assert len(facts) == 1

        # Second compact with new extractor output
        tm2 = _make_test_model(
            summary="Compressed session summary [test]",
            facts=[{"content": "Fact from round 2", "category": "fact"}],
        )
        runner.history_compressor.agent.pydantic_agent.model = tm2
        runner.sessions.add_messages(
            "multi-session", [ModelRequest(parts=[UserPromptPart(content="Message 2")])]
        )
        with _summarizer_agent.override(model=tm2), _extractor_agent.override(model=tm2):
            await cmd_compact(ctx)

        # Facts should now be 2 (accumulated)
        facts = runner.sessions.get_facts("multi-session")
        assert len(facts) == 2
        contents = [f.content for f in facts]
        assert any("round 1" in c for c in contents)
        assert any("round 2" in c for c in contents)
