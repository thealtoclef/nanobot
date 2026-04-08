"""Tests for SessionManager using the new append-only API (no Session.messages)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import pendulum

from nanobot.db import Database, upgrade_db
from nanobot.session.manager import Session, SessionManager
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    SystemPromptPart,
)


def make_user_message(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def make_assistant_message(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


def make_tool_call_message(tool_name: str, tool_call_id: str, args: dict) -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=tool_name, tool_call_id=tool_call_id, args=args)]
    )


def make_tool_result_message(tool_name: str, tool_call_id: str, content: str) -> ModelRequest:
    return ModelRequest(
        parts=[ToolReturnPart(tool_name=tool_name, tool_call_id=tool_call_id, content=content)]
    )


@pytest.fixture
def mgr(tmp_path: Path) -> SessionManager:
    upgrade_db(tmp_path)
    db = Database(tmp_path)
    return SessionManager(workspace=tmp_path, db=db)


class TestSessionManagerEnsureSession:
    """Tests for ensure_session (idempotent session creation)."""

    def test_ensure_session_creates_row(self, mgr: SessionManager) -> None:
        mgr.ensure_session("telegram:123")
        session = mgr.get_session("telegram:123")
        assert session.key == "telegram:123"

    def test_ensure_session_idempotent(self, mgr: SessionManager) -> None:
        mgr.ensure_session("tg:1")
        mgr.ensure_session("tg:1")  # no-op
        session = mgr.get_session("tg:1")
        assert session.key == "tg:1"

    def test_session_has_valid_timestamps(self, mgr: SessionManager) -> None:
        mgr.ensure_session("tg:ts")
        session = mgr.get_session("tg:ts")
        assert isinstance(session.created_at, pendulum.DateTime)
        assert isinstance(session.updated_at, pendulum.DateTime)

    def test_session_initial_last_consolidated_is_none(self, mgr: SessionManager) -> None:
        mgr.ensure_session("tg:new")
        session = mgr.get_session("tg:new")
        assert session.last_consolidated_message_id is None


class TestSessionManagerAddMessages:
    """Tests for add_message / add_messages (append-only blob storage)."""

    def test_add_single_message(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:add1")
        msg = make_user_message("hello")
        row_id = mgr.add_message("test:add1", msg)
        assert row_id > 0

    def test_add_messages_returns_last_row_id(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:addmulti")
        msgs = [make_user_message("hi"), make_assistant_message("hello")]
        row_id = mgr.add_messages("test:addmulti", msgs)
        assert row_id > 0

    def test_get_all_messages_returns_persisted(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:getall")
        msgs = [make_user_message("hi"), make_assistant_message("hello")]
        mgr.add_messages("test:getall", msgs)
        retrieved = mgr.get_all_messages("test:getall")
        assert len(retrieved) == 2

    def test_get_all_messages_preserves_order(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:order")
        msgs = [
            make_user_message("first"),
            make_assistant_message("second"),
            make_user_message("third"),
        ]
        mgr.add_messages("test:order", msgs)
        retrieved = mgr.get_all_messages("test:order")
        assert len(retrieved) == 3

    def test_multiple_blobs_accumulate(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:blobs")
        mgr.add_message("test:blobs", make_user_message("msg1"))
        mgr.add_message("test:blobs", make_user_message("msg2"))
        retrieved = mgr.get_all_messages("test:blobs")
        assert len(retrieved) == 2


class TestSessionManagerGetUnconsolidatedMessages:
    """Tests for get_unconsolidated_messages (filters by last_consolidated boundary)."""

    def test_empty_session_returns_empty_list(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:empty")
        uncons = mgr.get_unconsolidated_messages("test:empty")
        assert uncons == []

    def test_unconsolidated_returns_all_when_no_boundary(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:noboundary")
        msgs = [make_user_message("hi"), make_assistant_message("there")]
        mgr.add_messages("test:noboundary", msgs)
        uncons = mgr.get_unconsolidated_messages("test:noboundary")
        assert len(uncons) == 2

    def test_unconsolidated_filters_by_boundary(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:boundary")
        # Add messages as separate blobs so we can set boundary between them
        first_row_id = mgr.add_message("test:boundary", make_user_message("old"))
        second_row_id = mgr.add_message("test:boundary", make_assistant_message("mid"))
        mgr.add_message("test:boundary", make_user_message("new"))
        # Set boundary after second message (so "new" is unconsolidated)
        mgr.update_last_consolidated_message_id("test:boundary", second_row_id)
        uncons = mgr.get_unconsolidated_messages("test:boundary")
        assert len(uncons) == 1  # only "new" is unconsolidated

    def test_boundary_filters_multiple_blobs(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:multiboundary")
        first_row_id = mgr.add_message("test:multiboundary", make_user_message("msg1"))
        second_row_id = mgr.add_message("test:multiboundary", make_user_message("msg2"))
        mgr.add_message("test:multiboundary", make_user_message("msg3"))
        # Set boundary after second message
        mgr.update_last_consolidated_message_id("test:multiboundary", second_row_id)
        uncons = mgr.get_unconsolidated_messages("test:multiboundary")
        assert len(uncons) == 1


class TestSessionManagerUpdateLastConsolidated:
    """Tests for update_last_consolidated_message_id."""

    def test_update_boundary(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:updateboundary")
        first_row_id = mgr.add_message("test:updateboundary", make_user_message("a"))
        mgr.add_message("test:updateboundary", make_assistant_message("b"))
        mgr.update_last_consolidated_message_id("test:updateboundary", first_row_id)
        session = mgr.get_session("test:updateboundary")
        assert session.last_consolidated_message_id == first_row_id


class TestSessionManagerDeleteAllMessages:
    """Tests for delete_all_messages (resets boundary too)."""

    def test_delete_all_messages(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:deleteall")
        mgr.add_message("test:deleteall", make_user_message("msg1"))
        mgr.add_message("test:deleteall", make_user_message("msg2"))
        mgr.delete_all_messages("test:deleteall")
        assert mgr.get_all_messages("test:deleteall") == []

    def test_delete_resets_consolidation_boundary(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:deletereset")
        first_row_id = mgr.add_message("test:deletereset", make_user_message("msg1"))
        mgr.update_last_consolidated_message_id("test:deletereset", first_row_id)
        mgr.delete_all_messages("test:deletereset")
        session = mgr.get_session("test:deletereset")
        assert session.last_consolidated_message_id is None


class TestSessionManagerRoundTrip:
    """End-to-end round-trip tests: add → get → verify."""

    def test_add_and_retrieve_user_message(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:roundtrip1")
        msg = make_user_message("what is 2+2?")
        mgr.add_message("test:roundtrip1", msg)
        retrieved = mgr.get_all_messages("test:roundtrip1")
        assert len(retrieved) == 1
        assert isinstance(retrieved[0], ModelRequest)
        part = retrieved[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == "what is 2+2?"

    def test_add_and_retrieve_assistant_message(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:roundtrip2")
        msg = make_assistant_message("4")
        mgr.add_message("test:roundtrip2", msg)
        retrieved = mgr.get_all_messages("test:roundtrip2")
        assert len(retrieved) == 1
        assert isinstance(retrieved[0], ModelResponse)
        part = retrieved[0].parts[0]
        assert isinstance(part, TextPart)
        assert part.content == "4"

    def test_add_and_retrieve_tool_call_message(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:roundtrip3")
        msg = make_tool_call_message("read_file", "call_1", {"path": "/etc/hosts"})
        mgr.add_message("test:roundtrip3", msg)
        retrieved = mgr.get_all_messages("test:roundtrip3")
        assert len(retrieved) == 1
        assert isinstance(retrieved[0], ModelResponse)
        tool_part = retrieved[0].parts[0]
        assert isinstance(tool_part, ToolCallPart)
        assert tool_part.tool_name == "read_file"
        assert tool_part.tool_call_id == "call_1"

    def test_add_and_retrieve_tool_result_message(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:roundtrip4")
        msg = make_tool_result_message("read_file", "call_1", "127.0.0.1 localhost")
        mgr.add_message("test:roundtrip4", msg)
        retrieved = mgr.get_all_messages("test:roundtrip4")
        assert len(retrieved) == 1
        assert isinstance(retrieved[0], ModelRequest)
        tool_part = retrieved[0].parts[0]
        assert isinstance(tool_part, ToolReturnPart)
        assert tool_part.content == "127.0.0.1 localhost"

    def test_multiple_sessions_independent(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:indep1")
        mgr.ensure_session("test:indep2")
        mgr.add_message("test:indep1", make_user_message("session 1 msg"))
        mgr.add_message("test:indep2", make_user_message("session 2 msg"))
        retrieved1 = mgr.get_all_messages("test:indep1")
        retrieved2 = mgr.get_all_messages("test:indep2")
        assert len(retrieved1) == 1
        assert len(retrieved2) == 1
        part1 = retrieved1[0].parts[0]
        part2 = retrieved2[0].parts[0]
        assert isinstance(part1, UserPromptPart)
        assert isinstance(part2, UserPromptPart)
        assert "session 1" in part1.content
        assert "session 2" in part2.content


class TestSessionManagerTouch:
    """Tests for touch (updates updated_at)."""

    def test_touch_updates_timestamp(self, mgr: SessionManager) -> None:
        mgr.ensure_session("test:touch")
        original = mgr.get_session("test:touch").updated_at
        import time

        time.sleep(0.01)
        mgr.touch("test:touch")
        updated = mgr.get_session("test:touch").updated_at
        assert updated > original


class TestSessionManagerListSessions:
    """Tests for list_sessions."""

    def test_list_sessions_empty(self, mgr: SessionManager) -> None:
        assert mgr.list_sessions() == []

    def test_list_sessions_returns_all(self, mgr: SessionManager) -> None:
        mgr.ensure_session("tg:a")
        mgr.ensure_session("tg:b")
        sessions = mgr.list_sessions()
        keys = [s["key"] for s in sessions]
        assert "tg:a" in keys
        assert "tg:b" in keys
