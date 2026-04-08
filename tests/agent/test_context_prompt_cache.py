"""Tests for cache-friendly prompt construction."""

from __future__ import annotations

import tempfile
from datetime import datetime as real_datetime
from importlib.resources import files as pkg_files
from pathlib import Path
import datetime as datetime_module
import shutil

from nanobot.agent.context import ContextBuilder
from nanobot.db import Database


class _FakeDatetime(real_datetime):
    current = real_datetime(2026, 2, 24, 13, 59)

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls.current


def _make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return workspace


def _make_db(tmp_path: Path) -> Database:
    db_path = tmp_path / "test.db"
    return Database(tmp_path)


def test_bootstrap_files_are_backed_by_templates() -> None:
    template_dir = pkg_files("nanobot") / "templates"

    for filename in ContextBuilder.BOOTSTRAP_FILES:
        assert (template_dir / filename).is_file(), f"missing bootstrap template: {filename}"


def test_system_prompt_stays_stable_when_clock_changes(tmp_path, monkeypatch) -> None:
    """System prompt should not change just because wall clock minute changes."""
    monkeypatch.setattr(datetime_module, "datetime", _FakeDatetime)

    workspace = _make_workspace(tmp_path)
    db = _make_db(tmp_path)
    builder = ContextBuilder(workspace, db)

    _FakeDatetime.current = real_datetime(2026, 2, 24, 13, 59)
    prompt1 = builder.build_system_prompt()

    _FakeDatetime.current = real_datetime(2026, 2, 24, 14, 0)
    prompt2 = builder.build_system_prompt()

    assert prompt1 == prompt2


def test_runtime_context_is_separate_untrusted_user_message(tmp_path) -> None:
    """Runtime metadata should be merged with the user message (prompt_content), not in history."""
    workspace = _make_workspace(tmp_path)
    db = _make_db(tmp_path)
    builder = ContextBuilder(workspace, db)

    messages, prompt_content = builder.build_messages(
        history=[],
        current_message="Return exactly: OK",
        channel="cli",
        chat_id="direct",
    )

    # History is returned unchanged (no current message added)
    assert messages == []
    # Runtime context is in the prompt_content
    assert ContextBuilder._RUNTIME_CONTEXT_TAG in prompt_content
    assert "Current Time:" in prompt_content
    assert "Channel: cli" in prompt_content
    assert "Chat ID: direct" in prompt_content
    assert "Return exactly: OK" in prompt_content


def test_subagent_result_does_not_create_consecutive_assistant_messages(tmp_path) -> None:
    """History should be returned unchanged; current message goes to prompt_content."""
    workspace = _make_workspace(tmp_path)
    db = _make_db(tmp_path)
    builder = ContextBuilder(workspace, db)

    messages, prompt_content = builder.build_messages(
        history=[{"role": "assistant", "content": "previous result"}],
        current_message="subagent result",
        channel="cli",
        chat_id="direct",
    )

    # History returned unchanged
    assert len(messages) == 1
    assert messages[0]["role"] == "assistant"
    # Current message goes to prompt_content, not into history
    assert "subagent result" in prompt_content
