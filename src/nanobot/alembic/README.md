# Database Migrations

Uses SQLAlchemy + Alembic for schema migrations.

## Usage

Migrations are run automatically by `Database.__init__` on first access.

To run manually:

```bash
# Set workspace before running alembic commands
export NANOBOT_WORKSPACE=/path/to/workspace
uv run alembic upgrade head
uv run alembic revision -m "description"  # create new migration
uv run alembic downgrade -1               # rollback
```
