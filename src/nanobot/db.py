"""SQLAlchemy models and Database wrapper for nanobot persistence."""

from __future__ import annotations

from typing import Any

import uuid
from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy
from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

_MAX_APPEND_RETRIES = 5


class Base(DeclarativeBase):
    pass


class DBSession(Base):
    """A conversation session stored in SQLite."""

    __tablename__ = "sessions"
    __table_args__ = ()

    key: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    last_activity_at: Mapped[str] = mapped_column(String, nullable=False)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    last_consolidated_position: Mapped[int] = mapped_column(Integer, default=0)

    messages: Mapped[list["Message"]] = relationship(
        "Message",
        order_by="Message.position",
        lazy="noload",
        cascade="all, delete-orphan",
    )
    memory_entries: Mapped[list["MemoryEntry"]] = relationship(
        "MemoryEntry",
        order_by="MemoryEntry.created_at",
        lazy="noload",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """A single message in a conversation session."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("session_key", "position", name="uq_messages_session_position"),
        Index("idx_messages_session", "session_key"),
        Index("idx_messages_position", "session_key", "position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(
        String,
        ForeignKey("sessions.key", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    tool_calls: Mapped[str | None] = mapped_column(Text)
    tool_results: Mapped[str | None] = mapped_column(Text)
    tool_call_id: Mapped[str | None] = mapped_column(String)
    name: Mapped[str | None] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String, nullable=False)
    tokens: Mapped[int | None] = mapped_column(Integer)


class MemoryEntry(Base):
    """A memory entry — either curated (long-term knowledge) or history (conversation summaries)."""

    __tablename__ = "memory_entries"
    __table_args__ = (
        UniqueConstraint("session_key", "category", "key", name="uq_memory_session_category_key"),
        Index("idx_memory_session", "session_key"),
        Index("idx_memory_category", "session_key", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(
        String,
        ForeignKey("sessions.key", ondelete="CASCADE"),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(String, nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)


def make_engine(workspace: Path) -> sqlalchemy.Engine:
    """Create a SQLite engine with WAL mode for the workspace."""
    db_path = workspace / "sessions.db"
    # Ensure directory exists
    workspace.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy.create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA foreign_keys=ON"))
        conn.commit()
    return engine


def make_session_factory(engine: sqlalchemy.Engine):
    """Create a session factory bound to the engine."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def upgrade_db(workspace: Path) -> None:
    """Run alembic migrations to bring DB schema up to date."""
    from alembic import command
    from alembic.config import Config
    import importlib.resources

    db_path = workspace / "sessions.db"
    # Ensure directory exists
    workspace.mkdir(parents=True, exist_ok=True)

    # Load alembic.ini from inside the package, then override sqlalchemy.url.
    ref = importlib.resources.files("nanobot")
    if hasattr(ref, "_paths"):
        alembic_ini = next(iter(ref._paths)) / "alembic.ini"
    else:
        alembic_ini = Path(ref) / "alembic.ini"

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(alembic_cfg, "head")


class Database:
    """Database wrapper for nanobot persistence."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.engine = make_engine(workspace)
        self.SessionFactory = make_session_factory(self.engine)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------

    def get_or_create_session(self, session_key: str) -> DBSession:
        """Get existing session or create a new one."""
        now = self._now()
        with self.SessionFactory() as db:
            stmt = (
                sqlite_insert(DBSession)
                .values(
                    key=session_key,
                    created_at=now,
                    last_activity_at=now,
                    message_count=0,
                    last_consolidated_position=0,
                )
                .on_conflict_do_update(
                    index_elements=["key"],
                    set_={"last_activity_at": now},
                )
            )
            db.execute(stmt)
            db.commit()
            row = db.get(DBSession, session_key)
            return row

    def touch_session(self, session_key: str) -> None:
        """Update last_activity_at without loading the full session."""
        with self.SessionFactory() as db:
            db.execute(
                update(DBSession)
                .where(DBSession.key == session_key)
                .values(last_activity_at=self._now())
            )
            db.commit()

    def update_last_consolidated_position(self, session_key: str, position: int) -> None:
        with self.SessionFactory() as db:
            db.execute(
                update(DBSession)
                .where(DBSession.key == session_key)
                .values(last_consolidated_position=position)
            )
            db.commit()

    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------

    def append_message(
        self,
        session_key: str,
        role: str,
        content: str | None = None,
        *,
        tool_calls: str | None = None,
        tool_results: str | None = None,
        tool_call_id: str | None = None,
        name: str | None = None,
        timestamp: str | None = None,
        tokens: int | None = None,
    ) -> int:
        """Append a message and return its position."""
        for _attempt in range(_MAX_APPEND_RETRIES):
            try:
                return self._try_append_message(
                    session_key,
                    role,
                    content,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    tool_call_id=tool_call_id,
                    name=name,
                    timestamp=timestamp,
                    tokens=tokens,
                )
            except IntegrityError:
                continue
        raise RuntimeError(
            f"Failed to append message for session {session_key!r} "
            f"after {_MAX_APPEND_RETRIES} retries"
        )

    def _try_append_message(
        self,
        session_key: str,
        role: str,
        content: str | None,
        *,
        tool_calls: str | None,
        tool_results: str | None,
        tool_call_id: str | None,
        name: str | None,
        timestamp: str | None,
        tokens: int | None,
    ) -> int:
        now = self._now()
        with self.SessionFactory() as db:
            stmt = (
                sqlite_insert(DBSession)
                .values(
                    key=session_key,
                    created_at=now,
                    last_activity_at=now,
                    message_count=0,
                    last_consolidated_position=0,
                )
                .on_conflict_do_nothing(index_elements=["key"])
            )
            db.execute(stmt)
            db.flush()

            max_pos = db.execute(
                text("SELECT COALESCE(MAX(position), 0) FROM messages WHERE session_key = :key"),
                {"key": session_key},
            ).scalar()
            position = max_pos + 1

            ts = timestamp or self._now()
            msg = Message(
                session_key=session_key,
                position=position,
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_results=tool_results,
                tool_call_id=tool_call_id,
                name=name,
                timestamp=ts,
                tokens=tokens,
            )
            db.add(msg)

            db.execute(
                update(DBSession)
                .where(DBSession.key == session_key)
                .values(message_count=position, last_activity_at=ts)
            )
            db.commit()
            return position

    def get_messages(
        self,
        session_key: str,
        after_position: int = 0,
        limit: int | None = None,
    ) -> list[Message]:
        """Get messages after a given position."""
        with self.SessionFactory() as db:
            q = (
                db.query(Message)
                .where(Message.session_key == session_key, Message.position > after_position)
                .order_by(Message.position)
            )
            if limit is not None:
                q = q.limit(limit)
            return list(q.all())

    def get_all_messages(self, session_key: str) -> list[Message]:
        """Get all messages for a session, ordered by position."""
        with self.SessionFactory() as db:
            return list(
                db.query(Message)
                .where(Message.session_key == session_key)
                .order_by(Message.position)
                .all()
            )

    def save_all_messages(
        self, session_key: str, messages: list[dict[str, Any]], last_consolidated: int = 0
    ) -> None:
        """Replace all messages for a session and update session metadata atomically.

        Used by SessionManager to persist the full session state. Deletes existing
        messages and inserts the provided list, preserving position numbers.

        ``tool_calls`` and ``tool_results`` are serialized as JSON strings so they
        can be reliably deserialized on read-back (avoiding Python repr issues).
        """
        import json

        now = self._now()
        with self.SessionFactory() as db:
            # Delete existing messages for this session
            db.query(Message).where(Message.session_key == session_key).delete()

            # Insert all current messages
            for pos, msg in enumerate(messages, start=1):
                tc = msg.get("tool_calls")
                tr = msg.get("tool_results")
                db.add(
                    Message(
                        session_key=session_key,
                        position=pos,
                        role=msg.get("role", "user"),
                        content=msg.get("content"),
                        tool_calls=json.dumps(tc) if tc is not None else None,
                        tool_results=json.dumps(tr) if tr is not None else None,
                        tool_call_id=msg.get("tool_call_id"),
                        name=msg.get("name"),
                        timestamp=msg.get("timestamp") or now,
                        tokens=msg.get("tokens"),
                    )
                )

            row = db.get(DBSession, session_key)
            if row:
                row.message_count = len(messages)
                row.last_activity_at = now
                row.last_consolidated_position = last_consolidated

            db.commit()

    def delete_all_messages(self, session_key: str) -> None:
        """Delete all messages for a session."""
        with self.SessionFactory() as db:
            db.query(Message).where(Message.session_key == session_key).delete()
            db.commit()

    # -------------------------------------------------------------------------
    # Memory entries
    # -------------------------------------------------------------------------

    def upsert_memory(
        self,
        session_key: str,
        category: str,
        key: str,
        content: str,
    ) -> None:
        """Insert or update a memory entry (upsert by session_key+category+key)."""
        now = self._now()
        with self.SessionFactory() as db:
            existing = (
                db.query(MemoryEntry)
                .filter(
                    MemoryEntry.session_key == session_key,
                    MemoryEntry.category == category,
                    MemoryEntry.key == key,
                )
                .first()
            )

            if existing:
                existing.content = content
                existing.updated_at = now
            else:
                db.add(
                    MemoryEntry(
                        session_key=session_key,
                        category=category,
                        key=key,
                        content=content,
                        created_at=now,
                        updated_at=now,
                    )
                )
            db.commit()

    def append_history(self, session_key: str, entry: str) -> str:
        """Append a history entry and return its key."""
        now = self._now()
        key = f"history_{now.replace(':', '-').replace('+', '-')}_{uuid.uuid4().hex[:8]}"
        with self.SessionFactory() as db:
            db.add(
                MemoryEntry(
                    session_key=session_key,
                    category="history",
                    key=key,
                    content=entry,
                    created_at=now,
                    updated_at=now,
                )
            )
            db.commit()
        return key

    def get_curated_memory(self, session_key: str) -> str | None:
        """Get the current curated memory for a session."""
        with self.SessionFactory() as db:
            row = (
                db.query(MemoryEntry)
                .filter(
                    MemoryEntry.session_key == session_key,
                    MemoryEntry.category == "curated",
                    MemoryEntry.key == "curated_memory",
                )
                .first()
            )
            return row.content if row else None

    def get_recent_history(self, session_key: str, limit: int = 20) -> list[str]:
        """Get recent history entries as content strings."""
        with self.SessionFactory() as db:
            rows = (
                db.query(MemoryEntry)
                .filter(
                    MemoryEntry.session_key == session_key,
                    MemoryEntry.category == "history",
                )
                .order_by(MemoryEntry.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.content for r in reversed(rows)]

    def get_last_consolidated_position(self, session_key: str) -> int:
        """Get the last consolidated position for a session."""
        with self.SessionFactory() as db:
            row = db.get(DBSession, session_key)
            return row.last_consolidated_position if row else 0

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with their metadata."""
        with self.SessionFactory() as db:
            rows = db.query(DBSession).order_by(DBSession.last_activity_at.desc()).all()
            return [
                {
                    "key": row.key,
                    "created_at": row.created_at,
                    "last_activity_at": row.last_activity_at,
                    "message_count": row.message_count,
                }
                for row in rows
            ]
