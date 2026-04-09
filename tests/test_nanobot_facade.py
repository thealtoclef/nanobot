"""Tests for the Nanobot programmatic facade."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.db import upgrade_db
from nanobot.nanobot import Nanobot, RunResult


def _write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    upgrade_db(tmp_path)
    data = {
        "providers": {
            "openai": {
                "backend": "openai",
                "base_url": "https://api.openai.com",
                "api_key": "sk-test-key",
            }
        },
        "agent": {"models": [{"name": "gpt-4.1", "provider": "openai"}]},
    }
    if overrides:
        data.update(overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data))
    return config_path


def test_from_config_missing_file():
    with pytest.raises(FileNotFoundError):
        Nanobot.from_config("/nonexistent/config.json")


def test_from_config_creates_instance(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    assert bot._agent is not None
    assert bot._agent.workspace == tmp_path


def test_from_config_default_path():
    from nanobot.config.schema import Config, ProviderConfig, ModelConfig

    config = Config(
        providers={
            "openai": ProviderConfig(
                backend="openai", base_url="https://api.openai.com", api_key="sk-test"
            )
        },
        agent={"models": [ModelConfig(name="gpt-4", provider="openai")]},
    )
    with (
        patch("nanobot.config.loader.load_config") as mock_load,
        patch("nanobot.agents.talker.Agent") as mock_agent,
    ):
        mock_load.return_value = config
        mock_agent.return_value = MagicMock()
        bot = Nanobot.from_config()
        assert bot._agent is not None


@pytest.mark.asyncio
async def test_run_returns_result(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    bot._agent.run = AsyncMock(return_value=("Hello back!", []))

    result = await bot.run("hi")

    assert isinstance(result, RunResult)
    assert result.content == "Hello back!"
    bot._agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_none_response(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)
    bot._agent.run = AsyncMock(return_value=("", []))

    result = await bot.run("hi")
    assert result.content == ""


def test_workspace_override(tmp_path):
    config_path = _write_config(tmp_path)
    custom_ws = tmp_path / "custom_workspace"
    custom_ws.mkdir()

    bot = Nanobot.from_config(config_path, workspace=custom_ws)
    assert bot._agent.workspace == custom_ws


@pytest.mark.asyncio
async def test_run_custom_session_key(tmp_path):
    config_path = _write_config(tmp_path)
    bot = Nanobot.from_config(config_path, workspace=tmp_path)

    bot._agent.run = AsyncMock(return_value=("ok", []))

    await bot.run("hi", session_key="user-alice")
    bot._agent.run.assert_awaited_once()


def test_import_from_top_level():
    from nanobot import Nanobot as N, RunResult as R

    assert N is Nanobot
    assert R is RunResult
