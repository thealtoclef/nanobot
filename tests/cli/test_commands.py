"""Tests for the CLI commands (onboard, serve, channels)."""

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from nanobot.cli.commands import app
from nanobot.config.schema import Config

runner = CliRunner()


def _strip_ansi(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


@pytest.fixture
def mock_paths(tmp_path):
    """Mock config/workspace paths for test isolation."""
    import shutil

    with patch("nanobot.config.loader.get_config_path") as mock_cp, \
         patch("nanobot.config.loader.save_config") as mock_sc, \
         patch("nanobot.config.loader.load_config") as mock_lc, \
         patch("nanobot.cli.commands.get_workspace_path") as mock_ws:

        base_dir = tmp_path / "test_onboard_data"
        if base_dir.exists():
            shutil.rmtree(base_dir)
        base_dir.mkdir()

        config_file = base_dir / "config.json"
        workspace_dir = base_dir / "workspace"

        mock_cp.return_value = config_file
        mock_ws.return_value = workspace_dir
        mock_lc.side_effect = lambda _config_path=None: Config()

        def _save_config(config: Config, config_path: Path | None = None):
            target = config_path or config_file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(config.model_dump(by_alias=True)), encoding="utf-8")

        mock_sc.side_effect = _save_config

        yield config_file, workspace_dir, mock_ws

        if base_dir.exists():
            shutil.rmtree(base_dir)


def test_onboard_fresh_install(mock_paths):
    """No existing config — should create from scratch."""
    config_file, workspace_dir, mock_ws = mock_paths

    result = runner.invoke(app, ["onboard"])

    assert result.exit_code == 0
    assert "Created config" in result.stdout
    assert "Created workspace" in result.stdout
    assert "nanobot is ready" in result.stdout
    assert config_file.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "sessions.db").exists(), "sessions.db should be created by Database init"
    expected_workspace = Config().workspace_path
    assert mock_ws.call_args.args == (expected_workspace,)


def test_onboard_existing_config_refresh(mock_paths):
    """Config exists, user declines overwrite — should refresh (load-merge-save)."""
    config_file, workspace_dir, _ = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "existing values preserved" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "sessions.db").exists()


def test_onboard_existing_config_overwrite(mock_paths):
    """Config exists, user confirms overwrite — should reset to defaults."""
    config_file, workspace_dir, _ = mock_paths
    config_file.write_text('{"existing": true}')

    result = runner.invoke(app, ["onboard"], input="y\n")

    assert result.exit_code == 0
    assert "Config already exists" in result.stdout
    assert "Config reset to defaults" in result.stdout
    assert workspace_dir.exists()
    assert (workspace_dir / "sessions.db").exists()


def test_onboard_existing_workspace_safe_create(mock_paths):
    """Workspace exists — should not recreate, but still add missing templates."""
    config_file, workspace_dir, _ = mock_paths
    workspace_dir.mkdir(parents=True)
    config_file.write_text("{}")

    result = runner.invoke(app, ["onboard"], input="n\n")

    assert result.exit_code == 0
    assert "Created workspace" not in result.stdout
    assert "Created AGENTS.md" in result.stdout
    assert (workspace_dir / "AGENTS.md").exists()
    assert (workspace_dir / "sessions.db").exists()


def test_onboard_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["onboard", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output
    assert "--wizard" in stripped_output
    assert "--dir" not in stripped_output


def test_onboard_interactive_discard_does_not_save_or_create_workspace(mock_paths, monkeypatch):
    config_file, workspace_dir, _ = mock_paths

    from nanobot.cli.onboard import OnboardResult

    monkeypatch.setattr(
        "nanobot.cli.onboard.run_onboard",
        lambda initial_config: OnboardResult(config=initial_config, should_save=False),
    )

    result = runner.invoke(app, ["onboard", "--wizard"])

    assert result.exit_code == 0
    assert "No changes were saved" in result.stdout
    assert not config_file.exists()
    assert not workspace_dir.exists()


def test_onboard_uses_explicit_config_and_workspace_paths(tmp_path, monkeypatch):
    config_path = tmp_path / "instance" / "config.json"
    workspace_path = tmp_path / "workspace"

    monkeypatch.setattr("nanobot.channels.registry.discover_all", lambda: {})

    result = runner.invoke(
        app,
        ["onboard", "--config", str(config_path), "--workspace", str(workspace_path)],
    )

    assert result.exit_code == 0
    saved = Config.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    assert saved.workspace_path == workspace_path
    assert (workspace_path / "AGENTS.md").exists()
    stripped_output = _strip_ansi(result.stdout)
    compact_output = stripped_output.replace("\n", "")
    resolved_config = str(config_path.resolve())
    assert resolved_config in compact_output
    assert f"--config {resolved_config}" in compact_output


def test_onboard_wizard_preserves_explicit_config_in_next_steps(tmp_path, monkeypatch):
    config_path = tmp_path / "instance" / "config.json"
    workspace_path = tmp_path / "workspace"

    from nanobot.cli.onboard import OnboardResult

    monkeypatch.setattr(
        "nanobot.cli.onboard.run_onboard",
        lambda initial_config: OnboardResult(config=initial_config, should_save=True),
    )
    monkeypatch.setattr("nanobot.channels.registry.discover_all", lambda: {})

    result = runner.invoke(
        app,
        ["onboard", "--wizard", "--config", str(config_path), "--workspace", str(workspace_path)],
    )

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    compact_output = stripped_output.replace("\n", "")
    resolved_config = str(config_path.resolve())
    assert f'nanobot agent -m "Hello!" --config {resolved_config}' in compact_output
    assert f"nanobot gateway --config {resolved_config}" in compact_output


def test_agent_help_shows_workspace_and_config_options():
    result = runner.invoke(app, ["agent", "--help"])

    assert result.exit_code == 0
    stripped_output = _strip_ansi(result.stdout)
    assert "--workspace" in stripped_output
    assert "-w" in stripped_output
    assert "--config" in stripped_output
    assert "-c" in stripped_output


def test_channels_login_requires_channel_name():
    result = runner.invoke(app, ["channels", "login"])

    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# /stop command tests
# ---------------------------------------------------------------------------

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import cmd_stop


def _make_runner_mock() -> MagicMock:
    """Create a mock AgentRunner for testing commands."""
    runner = MagicMock()
    runner._active_tasks = {}
    runner.subagents.cancel_by_session = AsyncMock(return_value=0)
    return runner


def _make_ctx(
    runner_mock: MagicMock,
    session_key: str = "cli:test",
    content: str = "/stop",
) -> MagicMock:
    """Create a mock CommandContext for testing."""
    from dataclasses import dataclass
    from nanobot.bus.events import InboundMessage

    msg = MagicMock(spec=InboundMessage)
    msg.session_key = session_key
    msg.channel = "cli"
    msg.chat_id = "test"
    msg.metadata = {}

    ctx = MagicMock()
    ctx.msg = msg
    ctx.session = None
    ctx.key = session_key
    ctx.raw = content
    ctx.loop = runner_mock  # backward-compat alias
    ctx.runner = runner_mock
    return ctx


class TestCmdStop:
    @pytest.mark.asyncio
    async def test_stop_no_active_task(self):
        """No active tasks — should report no active task."""
        runner = _make_runner_mock()
        ctx = _make_ctx(runner)

        result = await cmd_stop(ctx)

        assert "No active task" in result.content

    @pytest.mark.asyncio
    async def test_stop_cancels_active_task(self):
        """Active task is cancelled when /stop is called."""
        runner = _make_runner_mock()
        cancelled = asyncio.Event()
        started = asyncio.Event()

        async def slow_task():
            started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        task = asyncio.create_task(slow_task())
        runner._active_tasks["cli:test"] = [task]
        await started.wait()  # ensure task is actually running before we cancel it

        ctx = _make_ctx(runner)
        await cmd_stop(ctx)

        # Let the cancellation propagate
        await asyncio.sleep(0)
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_stop_cancels_multiple_tasks(self):
        """Multiple active tasks are all cancelled."""
        runner = _make_runner_mock()
        events = [asyncio.Event(), asyncio.Event()]

        async def slow(idx):
            events[idx].set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise

        tasks = [asyncio.create_task(slow(i)) for i in range(2)]
        runner._active_tasks["cli:test"] = tasks

        # wait for all tasks to start
        await asyncio.gather(*[events[i].wait() for i in range(2)])

        ctx = _make_ctx(runner)
        await cmd_stop(ctx)

        # Let the cancellation propagate
        await asyncio.sleep(0)
        # tasks were removed from _active_tasks by cmd_stop
        assert runner._active_tasks.get("cli:test", []) == []

    @pytest.mark.asyncio
    async def test_stop_cancels_subagents(self):
        """subagents.cancel_by_session is called when /stop is invoked."""
        runner = _make_runner_mock()
        runner.subagents.cancel_by_session = AsyncMock(return_value=3)
        runner._active_tasks["cli:test"] = []

        ctx = _make_ctx(runner)
        result = await cmd_stop(ctx)

        runner.subagents.cancel_by_session.assert_awaited_once_with("cli:test")
        assert "3" in result.content

    @pytest.mark.asyncio
    async def test_stop_reports_total_cancelled(self):
        """Total count includes both active tasks and subagents."""
        runner = _make_runner_mock()
        runner.subagents.cancel_by_session = AsyncMock(return_value=2)

        async def slow():
            await asyncio.sleep(60)

        task = asyncio.create_task(slow())
        runner._active_tasks["cli:test"] = [task]

        ctx = _make_ctx(runner)
        result = await cmd_stop(ctx)

        # 1 active task + 2 subagents = 3
        assert "3" in result.content

