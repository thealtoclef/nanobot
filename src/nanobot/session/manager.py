"""Session management for conversation history, backed by SQLite via Database."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.db import Database
from nanobot.utils.helpers import find_legal_message_start


def _safe_json_loads(value: str | None) -> Any:
    """Deserialize a JSON string, with fallback to raw value for backward compat."""
    if not value:
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        # Backward compat: data stored as Python repr before JSON serialization fix
        logger.debug("json.loads fallback for value type {}", type(value).__name__)
        return value


@dataclass
class Session:
    """A conversation session."""

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to DB

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat(), **kwargs}
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated :]
        sliced = unconsolidated[-max_messages:]

        # Avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Drop orphan tool results at the front.
        start = find_legal_message_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules."""
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        start = find_legal_message_start(retained)
        if start:
            retained = retained[start:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions, backed by SQLite via Database.

    Sessions are stored in the sessions table; messages in the messages table.
    Database is the authoritative store for all message state.
    """

    def __init__(self, workspace: Path, db: Database | None = None):
        self.workspace = workspace
        self._db = db or Database(workspace)

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Always loads fresh from Database to ensure consistency.
        """
        db_sess = self._db.get_or_create_session(key)
        db_messages = self._db.get_all_messages(key)
        last_consolidated_pos = self._db.get_last_consolidated_position(key)

        # Build message dicts from DB rows
        msg_dicts: list[dict[str, Any]] = []
        for m in db_messages:
            msg_dict: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg_dict["tool_calls"] = _safe_json_loads(m.tool_calls)
            if m.tool_results:
                msg_dict["tool_results"] = _safe_json_loads(m.tool_results)
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            if m.name:
                msg_dict["name"] = m.name
            if m.tokens is not None:
                msg_dict["tokens"] = m.tokens
            msg_dicts.append(msg_dict)

        # Parse timestamps
        try:
            created_at = datetime.fromisoformat(db_sess.created_at)
        except (ValueError, TypeError):
            created_at = datetime.now()
        try:
            updated_at = datetime.fromisoformat(db_sess.last_activity_at)
        except (ValueError, TypeError):
            updated_at = datetime.now()

        return Session(
            key=key,
            messages=msg_dicts,
            created_at=created_at,
            updated_at=updated_at,
            metadata={},
            last_consolidated=last_consolidated_pos,
        )

    def save(self, session: Session) -> None:
        """Save a session to the database (messages + metadata in one transaction)."""
        self._db.save_all_messages(session.key, session.messages, session.last_consolidated)

    def touch(self, session_key: str) -> None:
        """Update last_activity_at without loading full session."""
        self._db.touch_session(session_key)

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions."""
        return self._db.list_sessions()
