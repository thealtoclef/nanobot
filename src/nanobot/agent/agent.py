"""PydanticAI-based agent for nanobot.

NanobotAgent is the primary agent implementation using pydanticAI.
AgentRunner is a thin orchestrator layer around this agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.agent import Agent as PydanticAIAgent
from pydantic_ai.agent import RunContext
from pydantic_ai.capabilities.hooks import Hooks

from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_skills import SkillsCapability

from loguru import logger


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AgentDepsT = dict[str, Any]


# ---------------------------------------------------------------------------
# Bootstrap / instructions builder
# ---------------------------------------------------------------------------

BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]


def build_instructions(workspace: Path) -> str:
    """Build PydanticAI instructions from bootstrap files and memory."""
    from nanobot.agent._identity import build_identity

    parts = [build_identity(workspace)]

    for filename in BOOTSTRAP_FILES:
        path = workspace / filename
        if path.exists():
            parts.append(f"## {filename}\n\n{path.read_text(encoding='utf-8')}")

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Message conversion helpers
# ---------------------------------------------------------------------------


def _to_user_content(
    content: str | list[dict[str, Any]],
) -> str | list[Any]:
    """Convert session-stored content to PydanticAI-native UserContent items."""
    if isinstance(content, str):
        return content
    parts: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if block.get("type") == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if url:
                parts.append(ImageUrl(url=url))
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
            continue
        parts.append(str(block))
    return parts


class ToolAdapter:
    """Registers nanobot Tool instances as PydanticAI agent tools."""

    def __init__(self, agent: PydanticAIAgent, max_result_chars: int = 16000):
        self._agent = agent
        self._max_result_chars = max_result_chars
        self._tools: list[Any] = []

    def register(self, tool_instance: Any) -> None:
        """Register a Tool instance on the agent."""

        @self._agent.tool(
            name=tool_instance.name,
            description=tool_instance.description,
        )
        async def bound_tool(ctx: RunContext, **kwargs: Any) -> str:
            result = await tool_instance.execute(**kwargs)
            if isinstance(result, str) and len(result) > self._max_result_chars:
                half = self._max_result_chars // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - self._max_result_chars:,} chars truncated) ...\n\n"
                    + result[-half:]
                )
            return result

        self._tools.append(tool_instance)
        logger.debug("Registered tool: %s", tool_instance.name)


# ---------------------------------------------------------------------------
# NanobotAgent
# ---------------------------------------------------------------------------


class NanobotAgent:
    """PydanticAI-based agent for nanobot.

    This is the primary agent implementation. It wraps pydanticAI's Agent
    and provides:
    - Tool registration via ToolAdapter
    - Streaming via run_stream_events()
    - Session history management
    - MCP server support via builtin_tools
    - Hooks via pydanticAI's Hooks capability
    Usage::

        agent = NanobotAgent(
            workspace=Path("~/.nanobot/workspace"),
            models=[...],
        )
        result = await agent.run("Hello", session=session)
    """

    def __init__(
        self,
        workspace: Path,
        models: list[Any],
        *,
        max_iterations: int = 200,
        max_tool_result_chars: int = 16000,
        context_window_tokens: int = 65536,
        timezone: str = "UTC",
        skills_directories: list[Path] | None = None,
        system_prompt: str | None = None,
        # PydanticAI-native parameters
        builtin_tools: list[Any] | None = None,
        tools: list[Any] | None = None,
        hooks: Hooks | None = None,
        tool_timeout: float | None = None,
        retries: int = 1,
    ) -> None:
        self.workspace = workspace
        self.models = models
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.context_window_tokens = context_window_tokens
        self.timezone = timezone
        self._builtin_tools = builtin_tools or []
        self._tools = tools or []
        self._hooks = hooks
        self._tool_timeout = tool_timeout
        self._retries = retries
        self._tool_adapter: ToolAdapter | None = None

        # Build instructions
        if system_prompt:
            instructions = system_prompt
        else:
            instructions = build_instructions(workspace)

        # Build capabilities
        capabilities: list[Any] = []
        if skills_directories:
            skill_dirs = [str(d) for d in skills_directories]
            capabilities.append(SkillsCapability(directories=skill_dirs))
        if hooks:
            capabilities.append(hooks)

        # Wrap models in FallbackModel (always, even for single model)
        from pydantic_ai.models.fallback import FallbackModel

        if not models:
            raise ValueError("No models configured. Set agent.models in your config.")
        pydantic_model = FallbackModel(*models)

        self._agent: PydanticAIAgent = Agent(
            model=pydantic_model,
            instructions=instructions,
            capabilities=capabilities if capabilities else None,
            builtin_tools=self._builtin_tools if self._builtin_tools is not None else None,
            tools=self._tools if self._tools is not None else None,
            retries=retries,
            tool_timeout=tool_timeout,
        )

        model_names = ", ".join(getattr(m, "model_name", str(m)) for m in models)
        logger.info("NanobotAgent initialized with models=%s", model_names)

    @property
    def pydantic_agent(self) -> PydanticAIAgent:
        return self._agent

    @property
    def tool_adapter(self) -> ToolAdapter:
        if self._tool_adapter is None:
            self._tool_adapter = ToolAdapter(self._agent, self.max_tool_result_chars)
        return self._tool_adapter

    async def run(
        self,
        user_message: str | list[Any],
        *,
        message_history: list[ModelMessage] | None = None,
    ) -> tuple[str, list[ModelMessage]]:
        """Run the agent with a user message and history.

        This method is stateless — it does NOT touch the session object.
        Session persistence is the caller's responsibility.

        Args:
            user_message: The user's message.
            message_history: Pre-built PydanticAI message history.

        Returns:
            Tuple of (output_text, new_model_messages). Use
            ``result.new_messages()`` semantics — the returned list contains
            only messages generated during this run.
        """
        result = await self._agent.run(
            user_prompt=user_message,
            message_history=message_history,
        )

        output = result.output if result.output is not None else ""
        return output, result.new_messages()

    def run_stream(
        self,
        user_message: str | list[Any],
        *,
        message_history: list[ModelMessage] | None = None,
    ):
        """Streaming variant — returns an async context manager with stream_text(delta=True).

        After the context manager exits, ``result.new_messages()`` contains
        all messages generated during the stream run.

        Usage:
            async with agent.run_stream(...) as result:
                async for delta in result.stream_text(delta=True):
                    print(delta)
            new_msgs = result.new_messages()
        """
        return self._agent.run_stream(
            user_message,
            message_history=message_history,
        )
