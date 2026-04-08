"""SQLAlchemy models and Database wrapper for nanobot persistence."""

from __future__ import annotations

from typing import Any

import uuid
from pathlib import Path

import pendulum
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


class SessionRow(Base):
    """A conversation session stored in SQLite."""

    __tablename__ = "sessions"
    __table_args__ = ()

    key: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[int] = mapped_column(Integer, nullable=False)
    last_consolidated_message_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )

    memory_entries: Mapped[list["MemoryEntry"]] = relationship(
        "MemoryEntry",
        order_by="MemoryEntry.created_at",
        lazy="noload",
        cascade="all, delete-orphan",
    )


class MessageRow(Base):
    """A message blob in a conversation session."""

    __tablename__ = "messages"
    __table_args__ = (Index("idx_messages_session", "session_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.key", ondelete="CASCADE"), nullable=False
    )
    messages_json: Mapped[bytes] = mapped_column(sqlalchemy.LargeBinary, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


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

    def _now_ms(self) -> int:
        """Return current unix timestamp in milliseconds."""
        return int(pendulum.now("UTC").float_timestamp * 1000)

    # -------------------------------------------------------------------------
    # SessionManager helpers
    # -------------------------------------------------------------------------

    def ensure_session(self, key: str) -> None:
        """Create session row if not exists. Idempotent."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            stmt = (
                sqlite_insert(SessionRow)
                .values(
                    key=key, created_at=now_ms, updated_at=now_ms, last_consolidated_message_id=None
                )
                .on_conflict_do_nothing(index_elements=["key"])
            )
            db.execute(stmt)
            db.commit()

    def get_session_row(self, key: str) -> SessionRow | None:
        """Get session row by key."""
        with self.SessionFactory() as db:
            return db.get(SessionRow, key)

    def touch_session(self, session_key: str) -> None:
        """Update updated_at timestamp."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            db.execute(
                update(SessionRow).where(SessionRow.key == session_key).values(updated_at=now_ms)
            )
            db.commit()

    def get_message_blobs(self, session_key: str) -> list[MessageRow]:
        """Get all message rows for a session, ordered by id."""
        with self.SessionFactory() as db:
            return list(
                db.query(MessageRow)
                .filter(MessageRow.session_key == session_key)
                .order_by(MessageRow.id)
                .all()
            )

    def get_unconsolidated_message_blobs(
        self, session_key: str, last_consolidated_id: int | None
    ) -> list[MessageRow]:
        """Get message rows after the last consolidation boundary."""
        with self.SessionFactory() as db:
            q = (
                db.query(MessageRow)
                .filter(MessageRow.session_key == session_key)
                .order_by(MessageRow.id)
            )
            if last_consolidated_id is not None:
                q = q.filter(MessageRow.id > last_consolidated_id)
            return list(q.all())

    def add_message_blob(self, session_key: str, messages_json: bytes) -> int:
        """Insert a message blob. Returns the inserted row id."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            row = MessageRow(
                session_key=session_key, messages_json=messages_json, created_at=now_ms
            )
            db.add(row)
            db.flush()
            inserted_id = row.id
            db.execute(
                update(SessionRow).where(SessionRow.key == session_key).values(updated_at=now_ms)
            )
            db.commit()
            return inserted_id

    def update_last_consolidated_message_id(self, session_key: str, message_id: int) -> None:
        """Update consolidation boundary."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            db.execute(
                update(SessionRow)
                .where(SessionRow.key == session_key)
                .values(last_consolidated_message_id=message_id, updated_at=now_ms)
            )
            db.commit()

    def delete_all_messages(self, session_key: str) -> None:
        """Delete all messages for a session. Resets consolidation boundary."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            db.query(MessageRow).filter(MessageRow.session_key == session_key).delete()
            db.execute(
                update(SessionRow)
                .where(SessionRow.key == session_key)
                .values(last_consolidated_message_id=None, updated_at=now_ms)
            )
            db.commit()

    def get_or_create_session(self, key: str) -> SessionRow:
        """Get or create session row. Returns SessionRow."""
        self.ensure_session(key)
        row = self.get_session_row(key)
        assert row is not None
        return row

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with their metadata."""
        with self.SessionFactory() as db:
            rows = db.query(SessionRow).order_by(SessionRow.updated_at.desc()).all()
            return [
                {
                    "key": row.key,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                    "last_consolidated_message_id": row.last_consolidated_message_id,
                }
                for row in rows
            ]

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
        now = pendulum.now("UTC").isoformat()
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
        now = pendulum.now("UTC").isoformat()
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
