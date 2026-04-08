"""Pytest fixtures for nanobot tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.db import Database, upgrade_db


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Create a Database instance with migrations applied."""
    upgrade_db(tmp_path)
    return Database(tmp_path)
