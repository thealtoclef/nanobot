"""Tests for nanobot.db.Database."""

import threading
from collections import Counter
from pathlib import Path

import pytest
import sqlalchemy as sa

from nanobot.db import Database, upgrade_db


class TestDatabase:
    def test_database_initializes_with_migrations(self, tmp_path: Path) -> None:
        """upgrade_db() creates the sessions.db file."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        assert db.engine is not None
        assert (tmp_path / "sessions.db").exists()

    def test_get_or_create_session(self, tmp_path: Path) -> None:
        """Can create and retrieve a session."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        session = db.get_or_create_session("test-session")
        assert session.key == "test-session"
        assert session.message_count == 0

    def test_append_and_retrieve_message(self, tmp_path: Path) -> None:
        """Can append a message and retrieve it."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.append_message("test-session", role="user", content="hello")
        messages = db.get_messages("test-session")
        assert len(messages) == 1
        assert messages[0].content == "hello"
        assert messages[0].role == "user"

    def test_message_positions_increment_correctly(self, tmp_path: Path) -> None:
        """Messages get sequential positions within a session."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)

        db.append_message("s1", role="user", content="msg1")
        db.append_message("s1", role="assistant", content="msg2")
        db.append_message("s1", role="user", content="msg3")

        messages = db.get_all_messages("s1")
        assert len(messages) == 3
        assert messages[0].position == 1
        assert messages[0].content == "msg1"
        assert messages[1].position == 2
        assert messages[1].content == "msg2"
        assert messages[2].position == 3
        assert messages[2].content == "msg3"

    def test_message_count_tracked_in_session_table(self, tmp_path: Path) -> None:
        """The sessions table message_count matches actual message count."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)

        db.append_message("s1", role="user", content="msg1")
        db.append_message("s1", role="assistant", content="msg2")
        db.append_message("s1", role="user", content="msg3")

        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT message_count FROM sessions WHERE key = 's1'")
            ).fetchone()
            assert row[0] == 3

    def test_save_all_messages_replaces_existing(self, tmp_path: Path) -> None:
        """save_all_messages replaces all messages for a session."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)

        db.append_message("s1", role="user", content="old")
        db.save_all_messages(
            "s1",
            [
                {"role": "user", "content": "new1"},
                {"role": "assistant", "content": "new2"},
            ],
        )

        messages = db.get_all_messages("s1")
        assert len(messages) == 2
        assert messages[0].content == "new1"
        assert messages[1].content == "new2"

        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT message_count FROM sessions WHERE key = 's1'")
            ).fetchone()
            assert row[0] == 2

    def test_last_consolidated_position_updated(self, tmp_path: Path) -> None:
        """last_consolidated_position is persisted correctly."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)

        db.append_message("s1", role="user", content="msg1")
        db.update_last_consolidated_position("s1", 5)

        pos = db.get_last_consolidated_position("s1")
        assert pos == 5

    def test_messages_isolated_per_session(self, tmp_path: Path) -> None:
        """Different sessions have separate message histories."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)

        db.append_message("s1", role="user", content="msg1")
        db.append_message("s2", role="user", content="msg2")

        msgs1 = db.get_all_messages("s1")
        msgs2 = db.get_all_messages("s2")

        assert len(msgs1) == 1
        assert msgs1[0].content == "msg1"
        assert len(msgs2) == 1
        assert msgs2[0].content == "msg2"


class TestConcurrentAccess:
    def test_concurrent_get_or_create_no_duplicates(self, tmp_path: Path) -> None:
        """Concurrent get_or_create_session calls produce exactly one session row."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        n_threads = 10
        results: list[str] = []
        errors: list[Exception] = []

        def worker():
            try:
                session = db.get_or_create_session("race-session")
                results.append(session.key)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"
        assert len(results) == n_threads

        with db.engine.connect() as conn:
            count = conn.execute(
                sa.text("SELECT COUNT(*) FROM sessions WHERE key = 'race-session'")
            ).scalar()
            assert count == 1

    def test_concurrent_append_no_position_duplicates(self, tmp_path: Path) -> None:
        """Concurrent append_message calls produce unique sequential positions."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        n_threads = 10
        positions: list[int] = []
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                pos = db.append_message("race-session", role="user", content=f"msg-{idx}")
                positions.append(pos)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Unexpected errors: {errors}"
        assert len(positions) == n_threads

        assert sorted(positions) == list(range(1, n_threads + 1))

        messages = db.get_all_messages("race-session")
        assert len(messages) == n_threads
        msg_positions = [m.position for m in messages]
        assert len(set(msg_positions)) == n_threads

        with db.engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT message_count FROM sessions WHERE key = 'race-session'")
            ).fetchone()
            assert row[0] == n_threads

    def test_concurrent_mixed_session_creates_and_appends(self, tmp_path: Path) -> None:
        """Mix of get_or_create and append_message across threads stays consistent."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        n_threads = 8
        errors: list[Exception] = []

        def session_worker():
            try:
                s = db.get_or_create_session("mixed-session")
                assert s.key == "mixed-session"
            except Exception as exc:
                errors.append(exc)

        def append_worker(idx: int):
            try:
                db.append_message("mixed-session", role="user", content=f"msg-{idx}")
            except Exception as exc:
                errors.append(exc)

        threads = []
        for i in range(n_threads):
            if i % 2 == 0:
                threads.append(threading.Thread(target=session_worker))
            else:
                threads.append(threading.Thread(target=append_worker, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Unexpected errors: {errors}"

        messages = db.get_all_messages("mixed-session")
        msg_positions = sorted(m.position for m in messages)
        assert msg_positions == list(range(1, len(messages) + 1))
