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
        assert row.current_history_id is None

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

        # Create a history entry and set it as current
        history_id = db.add_history("test-session", "Summary of first two messages", 2)
        db.update_current_history_id("test-session", history_id)

        # Should only get message 3
        summarized_through = db.get_summarized_through_message_id("test-session")
        assert summarized_through == 2

        unconsolidated = db.get_unconsolidated_message_blobs("test-session", summarized_through)
        assert len(unconsolidated) == 1
        assert unconsolidated[0].id == 3

        # Consolidate all
        history_id_all = db.add_history("test-session", "Summary of all messages", 3)
        db.update_current_history_id("test-session", history_id_all)
        summarized_through = db.get_summarized_through_message_id("test-session")
        unconsolidated = db.get_unconsolidated_message_blobs("test-session", summarized_through)
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

        history_id = db.add_history("test-session", "Summary", 2)
        db.update_current_history_id("test-session", history_id)
        db.delete_all_messages("test-session")

        rows = db.get_message_blobs("test-session")
        assert len(rows) == 0

        row = db.get_session_row("test-session")
        assert row is not None
        assert row.current_history_id is None

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


class TestHistoryAndFacts:
    def test_add_history(self, tmp_path: Path) -> None:
        """Can add a history entry and retrieve it."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        # Add some messages first
        for i in range(3):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        # Add history
        history_id = db.add_history("test-session", "Conversation summary", 3)
        assert history_id == 1

        # Link history to session
        db.update_current_history_id("test-session", history_id)

        # Verify via get_current_history_row
        history = db.get_current_history_row("test-session")
        assert history is not None
        assert history.id == history_id
        assert history.summary == "Conversation summary"
        assert history.summarized_through_message_id == 3

    def test_get_summarized_through_message_id(self, tmp_path: Path) -> None:
        """get_summarized_through_message_id returns correct value."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        for i in range(5):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        # No history yet
        assert db.get_summarized_through_message_id("test-session") is None

        # Add history through message 3
        history_id = db.add_history("test-session", "Summary 1", 3)
        db.update_current_history_id("test-session", history_id)
        assert db.get_summarized_through_message_id("test-session") == 3

        # Add another history through message 5
        history_id2 = db.add_history("test-session", "Summary 2", 5)
        db.update_current_history_id("test-session", history_id2)
        assert db.get_summarized_through_message_id("test-session") == 5

    def test_get_latest_history_summary(self, tmp_path: Path) -> None:
        """get_latest_history_summary returns current history's summary."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        # Add some messages first
        for i in range(5):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        # No history yet
        assert db.get_latest_history_summary("test-session") is None

        # Add first history
        history_id = db.add_history("test-session", "First summary", 2)
        db.update_current_history_id("test-session", history_id)
        assert db.get_latest_history_summary("test-session") == "First summary"

        # Add second history
        history_id2 = db.add_history("test-session", "Second summary", 4)
        db.update_current_history_id("test-session", history_id2)
        assert db.get_latest_history_summary("test-session") == "Second summary"

    def test_get_all_histories(self, tmp_path: Path) -> None:
        """get_all_histories returns all history rows ordered by id."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        for i in range(4):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        # Add multiple histories
        h1 = db.add_history("test-session", "Summary 1", 2)
        h2 = db.add_history("test-session", "Summary 2", 3)
        h3 = db.add_history("test-session", "Summary 3", 4)

        histories = db.get_all_histories("test-session")
        assert len(histories) == 3
        assert histories[0].id == h1
        assert histories[0].summary == "Summary 1"
        assert histories[1].id == h2
        assert histories[2].id == h3

    def test_add_fact(self, tmp_path: Path) -> None:
        """Can add a single fact and retrieve it."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        fact_id = db.add_fact("test-session", "User prefers dark mode", "preference")
        assert fact_id == 1

        facts = db.get_facts("test-session")
        assert len(facts) == 1
        assert facts[0].content == "User prefers dark mode"
        assert facts[0].category == "preference"

    def test_add_facts_bulk(self, tmp_path: Path) -> None:
        """add_facts bulk inserts multiple facts."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        facts = [
            ("User is admin", "role"),
            ("Theme is dark", "preference"),
            ("Email is test@example.com", "contact"),
        ]
        db.add_facts("test-session", facts)

        retrieved = db.get_facts("test-session")
        assert len(retrieved) == 3
        assert retrieved[0].content == "User is admin"
        assert retrieved[0].category == "role"
        assert retrieved[1].content == "Theme is dark"
        assert retrieved[1].category == "preference"
        assert retrieved[2].content == "Email is test@example.com"
        assert retrieved[2].category == "contact"

    def test_add_facts_empty_list(self, tmp_path: Path) -> None:
        """add_facts with empty list does nothing."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        db.add_facts("test-session", [])
        assert len(db.get_facts("test-session")) == 0

    def test_get_fact_digest(self, tmp_path: Path) -> None:
        """get_fact_digest builds correct formatted string."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        facts = [
            ("Likes pizza", "food"),
            ("Allergic to nuts", "health"),
        ]
        db.add_facts("test-session", facts)

        digest = db.get_fact_digest("test-session")
        assert "## Known Facts" in digest
        assert "[food] Likes pizza" in digest
        assert "[health] Allergic to nuts" in digest

    def test_get_fact_digest_truncation(self, tmp_path: Path) -> None:
        """get_fact_digest truncates at max_tokens * 4 chars."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        # Add a long fact
        long_content = "x" * 1000
        db.add_fact("test-session", long_content, "test")

        digest = db.get_fact_digest("test-session", max_tokens=10)
        # max_tokens=10 means max 40 chars
        assert len(digest) <= 43  # 40 + len("...") + header

    def test_get_fact_digest_empty(self, tmp_path: Path) -> None:
        """get_fact_digest returns empty string when no facts."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        digest = db.get_fact_digest("test-session")
        assert digest == ""

    def test_clear_session_for_new(self, tmp_path: Path) -> None:
        """clear_session_for_new deletes messages and resets history but keeps facts."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        # Add messages
        for i in range(3):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        # Add history
        history_id = db.add_history("test-session", "Old summary", 3)
        db.update_current_history_id("test-session", history_id)

        # Add facts
        db.add_facts("test-session", [("Some fact", "test")])

        # Clear session
        db.clear_session_for_new("test-session")

        # Messages should be gone
        assert len(db.get_message_blobs("test-session")) == 0

        # History should be reset
        assert db.get_current_history_row("test-session") is None

        # Facts should still exist
        facts = db.get_facts("test-session")
        assert len(facts) == 1
        assert facts[0].content == "Some fact"

    def test_facts_survive_clear_session(self, tmp_path: Path) -> None:
        """Facts persist after clear_session_for_new is called multiple times."""
        upgrade_db(tmp_path)
        db = Database(tmp_path)
        db.ensure_session("test-session")

        # Add initial facts
        db.add_fact("test-session", "Initial fact", "type1")

        # Clear once
        db.clear_session_for_new("test-session")

        # Add more messages and facts
        for i in range(2):
            msg = ModelRequest(parts=[UserPromptPart(content=f"msg{i}")])
            blob = ModelMessagesTypeAdapter.dump_json([msg])
            db.add_message_blob("test-session", blob)

        db.add_fact("test-session", "Second fact", "type2")

        # Clear again
        db.clear_session_for_new("test-session")

        # Both facts should survive
        facts = db.get_facts("test-session")
        assert len(facts) == 2
        contents = [f.content for f in facts]
        assert "Initial fact" in contents
        assert "Second fact" in contents
