"""High-level programmatic interface to nanobot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

from nanobot.agents.talker import TalkerAgent
from nanobot.bus.queue import MessageBus
from nanobot.db import Database
from nanobot.session import SessionManager


@dataclass(slots=True)
class RunResult:
    """Result of a single agent run."""

    content: str
    tools_used: list[str]
    messages: list[ModelMessage]


class Nanobot:
    """Programmatic facade for running the nanobot agent.

    Usage::

        bot = Nanobot.from_config()
        result = await bot.run("Summarize this repo")
        print(result.content)
    """

    def __init__(
        self,
        agent: TalkerAgent,
        bus: MessageBus,
        workspace: Path,
        *,
        db: Database | None = None,
    ) -> None:
        self._agent = agent
        self._bus = bus
        self._workspace = workspace
        self._db = db or Database(workspace)
        self._sessions = SessionManager(workspace, db=self._db)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path | None = None,
        *,
        workspace: str | Path | None = None,
    ) -> Nanobot:
        """Create a Nanobot instance from a config file.

        Args:
            config_path: Path to ``config.json``.  Defaults to
                ``~/.nanobot/config.json``.
            workspace: Override the workspace directory from config.
        """
        from nanobot.config.loader import load_config

        resolved: Path | None = None
        if config_path is not None:
            resolved = Path(config_path).expanduser().resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Config not found: {resolved}")

        config = load_config(resolved)
        if workspace is not None:
            config.agent.workspace = str(Path(workspace).expanduser().resolve())

        agent_config = config.agent
        workspace_path = config.workspace_path

        # Initialize observability (logfire SDK + OTEL backends)
        obs = config.observability
        from nanobot.observability import setup

        setup(
            enabled=obs.enabled,
            log_level=obs.log_level,
            service_name=obs.service_name,
            traces_endpoint=obs.traces_endpoint,
            metrics_endpoint=obs.metrics_endpoint,
            logs_endpoint=obs.logs_endpoint,
        )

        # Resolve models from config
        from nanobot.config.provider_spec import resolve_agent_models

        resolved_models = resolve_agent_models(config)

        # Build skills directories
        skills_dirs = [
            workspace_path / "skills",
        ]

        # Create PydanticAI-based agent
        agent = TalkerAgent(
            workspace=workspace_path,
            models=resolved_models,
            max_iterations=agent_config.max_tool_iterations,
            max_tool_result_chars=agent_config.max_tool_result_chars,
            context_window_tokens=agent_config.context_window_tokens,
            timezone=agent_config.timezone,
            skills_directories=skills_dirs if skills_dirs else None,
        )

        bus = MessageBus()
        from nanobot.db import Database

        db = Database(workspace_path)
        return cls(agent, bus, workspace_path, db=db)

    async def run(
        self,
        message: str,
        *,
        session_key: str = "sdk:default",
        hooks: list[Any] | None = None,
    ) -> RunResult:
        """Run the agent once and return the result.

        Args:
            message: The user message to process.
            session_key: Session identifier for conversation isolation.
                Different keys get independent history.
            hooks: Optional lifecycle hooks for this run (not yet supported with pydanticAI-native agent).
        """
        self._sessions.ensure_session(session_key)
        model_messages = self._sessions.get_unconsolidated_messages(session_key)

        try:
            content, new_messages = await self._agent.run(
                message,
                message_history=model_messages,
            )
        except Exception:
            raise

        if new_messages:
            self._sessions.add_messages(session_key, new_messages)
        all_messages = self._sessions.get_all_messages(session_key)

        return RunResult(
            content=(content or ""),
            tools_used=[],
            messages=all_messages,
        )
