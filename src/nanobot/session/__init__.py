"""Session management for conversation history, backed by SQLite via Database."""

from pathlib import Path
from typing import Any

import pendulum
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from nanobot.db import Database


class Session(BaseModel):
    """Pure data model — no DB reference, no messages field."""

    model_config = {"arbitrary_types_allowed": True}

    key: str
    current_history_id: int | None = None
    created_at: pendulum.DateTime
    updated_at: pendulum.DateTime


class SessionManager:
    def __init__(self, workspace: Path, db: Database | None = None):
        self.workspace = workspace
        self._db = db or Database(workspace)

    def _row_to_session(self, row: Any) -> Session:
        """Convert SessionRow to Session Pydantic model."""
        return Session(
            key=row.key,
            current_history_id=row.current_history_id,
            created_at=pendulum.from_timestamp(row.created_at / 1000.0, tz="UTC"),
            updated_at=pendulum.from_timestamp(row.updated_at / 1000.0, tz="UTC"),
        )

    def ensure_session(self, key: str) -> None:
        """Ensure session row exists. Idempotent."""
        self._db.ensure_session(key)

    def get_session(self, key: str) -> Session:
        """Get session metadata. Raises if not found."""
        row = self._db.get_session_row(key)
        if row is None:
            raise KeyError(f"Session {key!r} not found")
        return self._row_to_session(row)

    def get_all_messages(self, session_key: str) -> list[ModelMessage]:
        """Get ALL messages for session, in id order."""
        rows = self._db.get_message_blobs(session_key)
        result: list[ModelMessage] = []
        for row in rows:
            msgs = ModelMessagesTypeAdapter.validate_json(row.messages_json)
            result.extend(msgs)
        return result

    def get_unconsolidated_messages(self, session_key: str) -> list[ModelMessage]:
        """Get messages after the current history boundary (id > summarized_through_message_id)."""
        boundary = self._db.get_summarized_through_message_id(session_key)
        if boundary is None:
            rows = self._db.get_unconsolidated_message_blobs(session_key, None)
        else:
            rows = self._db.get_unconsolidated_message_blobs(session_key, boundary)
        result: list[ModelMessage] = []
        for row in rows:
            msgs = ModelMessagesTypeAdapter.validate_json(row.messages_json)
            result.extend(msgs)
        return result

    def get_unconsolidated_blobs_with_ids(
        self, session_key: str
    ) -> list[tuple[int, list[ModelMessage]]]:
        """Get unconsolidated blobs as (row_id, messages) pairs."""
        boundary = self._db.get_summarized_through_message_id(session_key)
        if boundary is None:
            rows = self._db.get_unconsolidated_message_blobs(session_key, None)
        else:
            rows = self._db.get_unconsolidated_message_blobs(session_key, boundary)
        result: list[tuple[int, list[ModelMessage]]] = []
        for row in rows:
            msgs = ModelMessagesTypeAdapter.validate_json(row.messages_json)
            result.append((row.id, msgs))
        return result

    def add_message(self, session_key: str, message: ModelMessage) -> int:
        """Append one message. Returns inserted row id."""
        blob = ModelMessagesTypeAdapter.dump_json([message])
        return self._db.add_message_blob(session_key, blob)

    def add_messages(self, session_key: str, messages: list[ModelMessage]) -> int:
        """Append messages from one turn as a single blob. Returns last inserted row id."""
        if not messages:
            return 0
        blob = ModelMessagesTypeAdapter.dump_json(messages)
        return self._db.add_message_blob(session_key, blob)

    def update_current_history_id(self, session_key: str, history_id: int) -> None:
        """Update consolidation boundary."""
        self._db.update_current_history_id(session_key, history_id)

    def delete_all_messages(self, session_key: str) -> None:
        """Delete all messages. Resets consolidation boundary."""
        self._db.delete_all_messages(session_key)

    def touch(self, session_key: str) -> None:
        """Update updated_at."""
        self._db.touch_session(session_key)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        return self._db.list_sessions()

    def get_current_history_row(self, session_key: str) -> Any:
        """Get current history row for session."""
        return self._db.get_current_history_row(session_key)

    def get_all_histories(self, session_key: str) -> list[Any]:
        """Get all history rows for session, ordered by id."""
        return self._db.get_all_histories(session_key)
