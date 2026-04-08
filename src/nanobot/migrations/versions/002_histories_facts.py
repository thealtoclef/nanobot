"""Add histories and facts tables, replace last_consolidated_message_id with current_history_id.

Revision ID: 002
Revises: 001
Create Date: 2026-04-09

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, Sequence[str], None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Create histories table
    op.create_table(
        "histories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_key", sa.String(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summarized_through_message_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["session_key"], ["sessions.key"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["summarized_through_message_id"],
            ["messages.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_histories_session_id", "histories", ["session_key", "id"], unique=False)

    # 2. Create facts table
    op.create_table(
        "facts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_key", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["session_key"], ["sessions.key"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_facts_session", "facts", ["session_key"], unique=False)

    # 3. Modify sessions table using batch mode for SQLite
    # This handles the add/drop column by recreating the table
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.add_column(sa.Column("current_history_id", sa.Integer(), nullable=True))
        batch_op.drop_column("last_consolidated_message_id")

    # 4. Add FK constraint for current_history_id -> histories.id using raw SQL
    # SQLite supports this via ALTER TABLE for CHECK/FK when NOT using batch mode
    # We use execute to add the constraint directly
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sessions_current_history_id ON sessions(current_history_id)
        """
    )
    # Note: SQLite doesn't support adding FK constraints via ALTER TABLE,
    # so we rely on the application-level enforcement via SQLAlchemy
    # The FK is defined in the model but not enforced at DB level for existing tables

    # 5. Drop legacy memory_entries table if it exists
    op.execute("DROP TABLE IF EXISTS memory_entries")


def downgrade() -> None:
    """Downgrade schema."""
    # 1. Recreate memory_entries table (empty, for downgrade integrity)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_key VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            key VARCHAR NOT NULL,
            content TEXT NOT NULL,
            created_at VARCHAR NOT NULL,
            updated_at VARCHAR NOT NULL,
            FOREIGN KEY (session_key) REFERENCES sessions(key) ON DELETE CASCADE
        )
        """
    )

    # 2. Modify sessions table to add back last_consolidated_message_id
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.add_column(sa.Column("last_consolidated_message_id", sa.Integer(), nullable=True))
        batch_op.drop_column("current_history_id")

    # 3. Drop facts table
    op.drop_index("idx_facts_session", table_name="facts")
    op.drop_table("facts")

    # 4. Drop histories table
    op.drop_index("idx_histories_session_id", table_name="histories")
    op.drop_table("histories")
