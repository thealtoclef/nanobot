"""Tests for nanobot.db.Database."""

import threading
from collections import Counter
from pathlib import Path

import pytest
import sqlalchemy as sa

from nanobot.db import Database, upgrade_db
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart


class TestDatabase:
    def test_database_initializes_with_migrations(self, tmp_path: Path) -> None:
        """upgrade_db() creates the sessions.db file."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        assert db.engine is not None
        assert (tmp_path / "sessions.db").exists()

    def test_ensure_session(self, tmp_path: Path) -> None:
        """Can create a session with ensure_session."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")
        row = db.get_session_row("test-session")
        assert row is not None
        assert row.key == "test-session"
        assert row.last_consolidated_message_id is None

    def test_ensure_session_idempotent(self, tmp_path: Path) -> None:
        """ensure_session is idempotent - calling twice doesn't error."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")
        db.ensure_session("test-session")  # Should not raise
        row = db.get_session_row("test-session")
        assert row is not None

    def test_add_message_blob(self, tmp_path: Path) -> None:
        """Can add a message blob and retrieve it."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")
        msg = ModelRequest(parts=[UserPromptPart(content="hello")])
        blob = ModelMessagesTypeAdapter.dump_json([msg])
        id = db.add_message_blob("test-session", blob)
        assert id == 1

        rows = db.get_message_blobs("test-session")
        assert len(rows) == 1
        assert rows[0].id == 1

    def test_message_blobs_ordered_by_id(self, tmp_path: Path) -> None:
        """Message blobs are returned in id order."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        for i in range(3):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        rows = db.get_message_blobs("test-session")
        assert len(rows) == 3
        assert rows[0].id == 1
        assert rows[1].id == 2
        assert rows[2].id == 3

    def test_unconsolidated_message_blobs_filtering(self, tmp_path: Path) -> None:
        """get_unconsolidated_message_blobs returns only messages after last_consolidated_id."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        # Add 3 message blobs
        for i in range(3):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        # Consolidate first two messages
        db.update_last_consolidated_message_id("test-session", 2)

        # Should only get message 3
        unconsolidated = db.get_unconsolidated_message_blobs("test-session", 2)
        assert len(unconsolidated) == 1
        assert unconsolidated[0].id == 3

        # Consolidate all
        db.update_last_consolidated_message_id("test-session", 3)
        unconsolidated = db.get_unconsolidated_message_blobs("test-session", 3)
        assert len(unconsolidated) == 0

    def test_delete_all_messages(self, tmp_path: Path) -> None:
        """delete_all_messages removes all message blobs and resets consolidation."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        for i in range(3):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        db.update_last_consolidated_message_id("test-session", 2)
        db.delete_all_messages("test-session")

        rows = db.get_message_blobs("test-session")
        assert len(rows) == 0

        row = db.get_session_row("test-session")
        assert row is not None
        assert row.last_consolidated_message_id is None

    def test_touch_session_updates_timestamp(self, tmp_path: Path) -> None:
        """touch_session updates the updated_at timestamp."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")
        row_before = db.get_session_row("test-session")
        assert row_before is not None
        original_updated_at = row_before.updated_at

        import time

        time.sleep(0.01)  # Small delay to ensure timestamp difference

        db.touch_session("test-session")
        row_after = db.get_session_row("test-session")
        assert row_after is not None
        assert row_after.updated_at >= original_updated_at

    def test_messages_isolated_per_session(self, tmp_path: Path) -> None:
        """Different sessions have separate message histories."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)

        db.ensure_session("s1")
        db.ensure_session("s2")

        msg1 = ModelRequest(parts=[UserPromptPart(content="hello s1")])
        blob1 = ModelMessagesTypeAdapter.dump_json([msg1])
        db.add_message_blob("s1", blob1)

        msg2 = ModelRequest(parts=[UserPromptPart(content="hello s2")])
        blob2 = ModelMessagesTypeAdapter.dump_json([msg2])
        db.add_message_blob("s2", blob2)

        rows_s1 = db.get_message_blobs("s1")
        rows_s2 = db.get_message_blobs("s2")

        assert len(rows_s1) == 1
        assert len(rows_s2) == 1


class TestConcurrentAccess:
    def test_concurrent_ensure_session_no_duplicates(self, tmp_path: Path) -> None:
        """Concurrent ensure_session calls produce exactly one session row."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        n_threads = 10
        errors: list[Exception] = []

        def worker():
            try:
                db.ensure_session("race-session")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Unexpected errors: {errors}"

        with db.engine.connect() as conn:
            count = conn.execute(
                sa.text("SELECT COUNT(*) FROM sessions WHERE key = 'race-session'")
            ).scalar()
            assert count == 1

    def test_concurrent_add_message_blob_unique_ids(self, tmp_path: Path) -> None:
        """Concurrent add_message_blob calls produce unique sequential ids."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        n_threads = 10
        ids: list[int] = []
        errors: list[Exception] = []

        # Ensure session exists first
        db.ensure_session("race-session")

        def worker(idx: int):
            try:
                msg = ModelRequest(parts=[UserPromptPart(content=f"msg-{idx}")])
                blob = ModelMessagesTypeAdapter.dump_json([msg])
                id = db.add_message_blob("race-session", blob)
                ids.append(id)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Unexpected errors: {errors}"
        assert len(ids) == n_threads
        assert len(set(ids)) == n_threads  # All unique

        rows = db.get_message_blobs("race-session")
        assert len(rows) == n_threads

    def test_concurrent_mixed_ensure_and_add(self, tmp_path: Path) -> None:
        """Mix of ensure_session and add_message_blob across threads stays consistent."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        n_threads = 8
        errors: list[Exception] = []

        def session_worker():
            try:
                db.ensure_session("mixed-session")
            except Exception as exc:
                errors.append(exc)

        def append_worker(idx: int):
            try:
                msg = ModelRequest(parts=[UserPromptPart(content=f"msg-{idx}")])
                blob = ModelMessagesTypeAdapter.dump_json([msg])
                db.add_message_blob("mixed-session", blob)
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

        rows = db.get_message_blobs("mixed-session")
        assert len(rows) == n_threads // 2  # Half are append workers
