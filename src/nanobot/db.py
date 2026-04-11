"""SQLAlchemy models and Database wrapper for nanobot persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pendulum
import sqlalchemy
from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
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
    current_history_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("histories.id", ondelete="SET NULL"), nullable=True, default=None
    )

    histories: Mapped[list["HistoryRow"]] = relationship(
        "HistoryRow",
        order_by="HistoryRow.id",
        lazy="noload",
        cascade="all, delete-orphan",
        foreign_keys="HistoryRow.session_key",
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


class HistoryRow(Base):
    """A conversation summary stored for a session."""

    __tablename__ = "histories"
    __table_args__ = (Index("idx_histories_session_id", "session_key", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.key", ondelete="CASCADE"), nullable=False
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    summarized_through_message_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)


class SubagentSessionRow(Base):
    """A background subagent session, linked to a parent main session."""

    __tablename__ = "subagent_sessions"
    __table_args__ = (
        Index("idx_subagent_sessions_parent", "parent_key"),
        Index("idx_subagent_sessions_label_parent", "label", "parent_key", unique=True),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)  # UUID v7
    parent_key: Mapped[str] = mapped_column(
        String, ForeignKey("sessions.key", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String, nullable=False)  # slug from LLM, unique per parent
    task: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    subagent_session_key: Mapped[str] = mapped_column(String, nullable=False)
    origin_channel: Mapped[str] = mapped_column(String, nullable=False)
    origin_chat_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    completed_at: Mapped[int | None] = mapped_column(Integer, nullable=True)


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
    import importlib.resources

    from alembic import command
    from alembic.config import Config

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
                .values(key=key, created_at=now_ms, updated_at=now_ms, current_history_id=None)
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

    def update_current_history_id(self, session_key: str, history_id: int) -> None:
        """Update current history id."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            db.execute(
                update(SessionRow)
                .where(SessionRow.key == session_key)
                .values(current_history_id=history_id, updated_at=now_ms)
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
                .values(current_history_id=None, updated_at=now_ms)
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
                    "current_history_id": row.current_history_id,
                }
                for row in rows
            ]

    # -------------------------------------------------------------------------
    # History methods
    # -------------------------------------------------------------------------

    def add_history(
        self, session_key: str, summary: str, summarized_through_message_id: int | None
    ) -> int:
        """Insert HistoryRow, return inserted id, update session updated_at."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            row = HistoryRow(
                session_key=session_key,
                summary=summary,
                summarized_through_message_id=summarized_through_message_id,
                created_at=now_ms,
            )
            db.add(row)
            db.flush()
            inserted_id = row.id
            db.execute(
                update(SessionRow).where(SessionRow.key == session_key).values(updated_at=now_ms)
            )
            db.commit()
            return inserted_id

    def get_current_history_row(self, session_key: str) -> HistoryRow | None:
        """Get session row, follow current_history_id FK, return HistoryRow or None."""
        with self.SessionFactory() as db:
            session = db.get(SessionRow, session_key)
            if session is None:
                return None
            if session.current_history_id is not None:
                return db.get(HistoryRow, session.current_history_id)
            return None

    def get_summarized_through_message_id(self, session_key: str) -> int | None:
        """Get current history row's summarized_through_message_id, or None."""
        history = self.get_current_history_row(session_key)
        return history.summarized_through_message_id if history else None

    def get_latest_history_summary(self, session_key: str) -> str | None:
        """Get current history row's summary text, or None."""
        history = self.get_current_history_row(session_key)
        return history.summary if history else None

    def get_all_histories(self, session_key: str) -> list[HistoryRow]:
        """All history rows for session ordered by id."""
        with self.SessionFactory() as db:
            return list(
                db.query(HistoryRow)
                .filter(HistoryRow.session_key == session_key)
                .order_by(HistoryRow.id)
                .all()
            )

    # -------------------------------------------------------------------------
    # Subagent session methods
    # -------------------------------------------------------------------------

    def create_subagent_session(
        self,
        id: str,
        parent_key: str,
        label: str,
        task: str,
        origin_channel: str,
        origin_chat_id: str,
    ) -> str:
        """Create a subagent session row. Returns the id (UUID)."""
        now_ms = self._now_ms()
        subagent_session_key = f"subagent:{id}"
        with self.SessionFactory() as db:
            row = SubagentSessionRow(
                id=id,
                parent_key=parent_key,
                label=label,
                task=task,
                status="running",
                subagent_session_key=subagent_session_key,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
                created_at=now_ms,
            )
            db.add(row)
            db.commit()
            return id

    def complete_subagent_session(
        self,
        id: str,
        status: str,
        result: str | None,
    ) -> None:
        """Update subagent session status and result."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            db.execute(
                update(SubagentSessionRow)
                .where(SubagentSessionRow.id == id)
                .values(status=status, result=result, completed_at=now_ms)
            )
            db.commit()

    def get_subagent_session(self, id: str) -> SubagentSessionRow | None:
        """Get a subagent session row by id (UUID)."""
        with self.SessionFactory() as db:
            return db.get(SubagentSessionRow, id)

    def get_subagent_session_by_label(
        self, label: str, parent_key: str
    ) -> SubagentSessionRow | None:
        """Get a subagent session row by label (slug) and parent_key."""
        with self.SessionFactory() as db:
            return (
                db.query(SubagentSessionRow)
                .filter(
                    SubagentSessionRow.label == label,
                    SubagentSessionRow.parent_key == parent_key,
                )
                .first()
            )

    def list_subagent_sessions(self, parent_key: str | None = None) -> list[SubagentSessionRow]:
        """List subagent sessions, optionally filtered by parent_key."""
        with self.SessionFactory() as db:
            q = db.query(SubagentSessionRow).order_by(SubagentSessionRow.created_at.desc())
            if parent_key is not None:
                q = q.filter(SubagentSessionRow.parent_key == parent_key)
            return list(q.all())

    def delete_subagent_sessions(self, parent_key: str) -> int:
        """Delete all subagent sessions for a parent. Also deletes orphaned subagent sessions rows.

        This cleans up both the subagent_sessions metadata rows AND the corresponding
        subagent:* session rows from the sessions table (which would otherwise become
        orphaned since subagent_session_key is just a String, not a FK).
        """
        with self.SessionFactory() as db:
            # Get subagent session keys before deleting (to clean up sessions table)
            subagent_keys = [
                row.subagent_session_key
                for row in db.query(SubagentSessionRow.subagent_session_key)
                .filter(SubagentSessionRow.parent_key == parent_key)
                .all()
            ]

            # Delete subagent_sessions rows
            count = (
                db.query(SubagentSessionRow)
                .filter(SubagentSessionRow.parent_key == parent_key)
                .delete()
            )

            # Delete orphaned subagent:* session rows and their messages
            for sk in subagent_keys:
                db.query(MessageRow).filter(MessageRow.session_key == sk).delete()
                db.query(SessionRow).filter(SessionRow.key == sk).delete()

            db.commit()
            return count

    def recover_interrupted_subagents(self) -> int:
        """Mark all 'running' subagent sessions as 'interrupted'. Returns count."""
        now_ms = self._now_ms()
        with self.SessionFactory() as db:
            rows = db.query(SubagentSessionRow).filter(SubagentSessionRow.status == "running").all()
            for row in rows:
                row.status = "interrupted"
                row.completed_at = now_ms
            db.commit()
            return len(rows)
