"""Initial schema.

Revision ID: 001
Revises:
Create Date: 2026-04-09

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # sessions table
    op.create_table(
        "sessions",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=False),
        sa.Column("last_consolidated_message_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("key"),
    )

    # messages table
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_key", sa.String(), nullable=False),
        sa.Column("messages_json", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["session_key"], ["sessions.key"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_messages_session", "messages", ["session_key"], unique=False)

    # memory_entries table (legacy, to be dropped in migration 002)
    op.create_table(
        "memory_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_key", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("updated_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["session_key"], ["sessions.key"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_memory_session", "memory_entries", ["session_key"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_memory_session", table_name="memory_entries")
    op.drop_table("memory_entries")
    op.drop_index("idx_messages_session", table_name="messages")
    op.drop_table("messages")
    op.drop_table("sessions")
