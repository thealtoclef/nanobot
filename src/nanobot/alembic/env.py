import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import our Base and set target_metadata for autogenerate
import sys
from pathlib import Path

# Add src to path so we can import nanobot.db
src_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(src_path))

from nanobot.db import Base

target_metadata = Base.metadata

# Allow database URL to be overridden at runtime via NANOBOT_WORKSPACE env var
# This lets us run migrations against the correct workspace DB
_workspace = os.environ.get("NANOBOT_WORKSPACE")
if _workspace:
    db_path = Path(_workspace) / "sessions.db"
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
