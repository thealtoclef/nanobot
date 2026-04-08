from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from nanobot.db import Database, upgrade_db
from nanobot.session.manager import Session, SessionManager


class TestSessionDataclass:
    def test_default_state(self) -> None:
        s = Session(key="test:1")
        assert s.messages == []
        assert s.last_consolidated == 0
        assert s.metadata == {}

    def test_add_message_appends(self) -> None:
        s = Session(key="test:1")
        s.add_message("user", "hello")
        assert len(s.messages) == 1
        assert s.messages[0]["role"] == "user"
        assert s.messages[0]["content"] == "hello"
        assert "timestamp" in s.messages[0]

    def test_add_message_with_kwargs(self) -> None:
        s = Session(key="test:1")
        s.add_message("assistant", "response", tool_calls=[{"id": "tc1"}])
        assert s.messages[0]["tool_calls"] == [{"id": "tc1"}]

    def test_add_message_updates_timestamp(self) -> None:
        s = Session(key="test:1")
        before = s.updated_at
        s.add_message("user", "hello")
        assert s.updated_at >= before

    def test_clear_resets_state(self) -> None:
        s = Session(key="test:1")
        s.add_message("user", "hello")
        s.last_consolidated = 5
        s.clear()
        assert s.messages == []
        assert s.last_consolidated == 0

    def test_get_history_empty(self) -> None:
        s = Session(key="test:1")
        assert s.get_history() == []

    def test_get_history_returns_clean_entries(self) -> None:
        s = Session(key="test:1")
        s.add_message("user", "hello")
        s.add_message("assistant", "hi", tool_calls=[{"id": "tc1"}])
        history = s.get_history()
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["tool_calls"] == [{"id": "tc1"}]
        assert "timestamp" not in history[0]

    def test_get_history_limits_messages(self) -> None:
        s = Session(key="test:1")
        for i in range(20):
            s.add_message("user", f"msg{i}")
        history = s.get_history(max_messages=5)
        assert len(history) <= 5

    def test_get_history_omits_none_content(self) -> None:
        s = Session(key="test:1")
        s.messages.append({"role": "user", "content": "q"})
        s.messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]})
        history = s.get_history()
        assert history[1]["content"] is None


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    upgrade_db(tmp_path)
    db = Database(tmp_path)
    return SessionManager(workspace=tmp_path, db=db)


class TestSessionManagerGetOrCreate:
    def test_returns_empty_session_for_new_key(self, manager: SessionManager) -> None:
        session = manager.get_or_create("telegram:123")
        assert session.key == "telegram:123"
        assert session.messages == []
        assert session.last_consolidated == 0

    def test_returns_same_key_on_repeat_call(self, manager: SessionManager) -> None:
        s1 = manager.get_or_create("tg:1")
        s2 = manager.get_or_create("tg:1")
        assert s1.key == s2.key == "tg:1"

    def test_sessions_are_isolated(self, manager: SessionManager) -> None:
        s1 = manager.get_or_create("tg:1")
        s1_copy = manager.get_or_create("tg:1")
        s2 = manager.get_or_create("tg:2")
        assert s1.key != s2.key
        assert s1_copy.key == s1.key

    def test_session_has_valid_timestamps(self, manager: SessionManager) -> None:
        session = manager.get_or_create("tg:1")
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)


class TestSessionManagerSave:
    def test_save_and_reload_preserves_messages(self, manager: SessionManager) -> None:
        session = manager.get_or_create("tg:save")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi there")
        manager.save(session)

        loaded = manager.get_or_create("tg:save")
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["role"] == "user"
        assert loaded.messages[0]["content"] == "hello"
        assert loaded.messages[1]["role"] == "assistant"
        assert loaded.messages[1]["content"] == "hi there"

    def test_save_replaces_existing_messages(self, manager: SessionManager) -> None:
        session = manager.get_or_create("tg:replace")
        session.add_message("user", "old msg")
        manager.save(session)

        session.clear()
        session.add_message("user", "new msg")
        manager.save(session)

        loaded = manager.get_or_create("tg:replace")
        assert len(loaded.messages) == 1
        assert loaded.messages[0]["content"] == "new msg"

    def test_save_persists_last_consolidated(self, manager: SessionManager) -> None:
        session = manager.get_or_create("tg:cons")
        session.add_message("user", "hello")
        session.last_consolidated = 1
        manager.save(session)

        loaded = manager.get_or_create("tg:cons")
        assert loaded.last_consolidated == 1

    def test_save_preserves_tool_call_data(self, manager: SessionManager) -> None:
        session = manager.get_or_create("tg:tools")
        session.add_message("user", "do thing")
        tool_calls_json = '[{"id": "tc_1", "type": "function", "function": {"name": "my_tool", "arguments": "{}"}}]'
        session.messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls_json})
        session.messages.append(
            {"role": "tool", "content": "result", "tool_call_id": "tc_1", "name": "my_tool"}
        )
        manager.save(session)

        loaded = manager.get_or_create("tg:tools")
        assert len(loaded.messages) == 3
        assert loaded.messages[1]["tool_calls"] == tool_calls_json
        assert loaded.messages[2]["tool_call_id"] == "tc_1"
        assert loaded.messages[2]["name"] == "my_tool"


class TestSessionManagerTouch:
    def test_touch_updates_activity(self, manager: SessionManager) -> None:
        session = manager.get_or_create("tg:touch")
        original_updated = session.updated_at
        manager.touch("tg:touch")

        loaded = manager.get_or_create("tg:touch")
        assert loaded.updated_at >= original_updated


class TestSessionManagerListSessions:
    def test_list_sessions_returns_empty(self, manager: SessionManager) -> None:
        assert manager.list_sessions() == []


class TestSessionManagerEndToEnd:
    def test_full_lifecycle(self, manager: SessionManager) -> None:
        session = manager.get_or_create("tg:e2e")
        session.add_message("user", "what is 2+2?")
        session.add_message("assistant", "4")
        manager.save(session)

        loaded = manager.get_or_create("tg:e2e")
        assert len(loaded.messages) == 2

        loaded.add_message("user", "thanks!")
        manager.save(loaded)

        final = manager.get_or_create("tg:e2e")
        assert len(final.messages) == 3
        assert final.messages[2]["content"] == "thanks!"

    def test_multiple_sessions_independent(self, manager: SessionManager) -> None:
        s1 = manager.get_or_create("tg:1")
        s2 = manager.get_or_create("tg:2")

        s1.add_message("user", "session 1 msg")
        s2.add_message("user", "session 2 msg")

        manager.save(s1)
        manager.save(s2)

        loaded1 = manager.get_or_create("tg:1")
        loaded2 = manager.get_or_create("tg:2")

        assert len(loaded1.messages) == 1
        assert loaded1.messages[0]["content"] == "session 1 msg"
        assert len(loaded2.messages) == 1
        assert loaded2.messages[0]["content"] == "session 2 msg"
