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
    ModelMessage,
    ModelRequest,
    ModelResponse,
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


def session_messages_to_model_messages(
    session_messages: list[dict[str, Any]],
) -> list[ModelMessage]:
    """Convert nanobot session messages to PydanticAI ModelMessage list."""
    result: list[ModelMessage] = []

    for msg in session_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            result.append(ModelRequest(parts=[SystemPromptPart(content=content)]))
        elif role == "user":
            result.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            tc = msg.get("tool_calls")
            if tc:
                parts = [TextPart(content=content)] if content else []
                for call in tc:
                    parts.append(
                        ToolCallPart(
                            tool_name=call.get("name", call.get("function", {}).get("name", "")),
                            tool_call_id=call.get("id", ""),
                            args=call.get("arguments", {}),
                        )
                    )
                result.append(ModelResponse(parts=parts))
            else:
                result.append(ModelResponse(parts=[TextPart(content=content)]))
        elif role == "tool":
            result.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=msg.get("name", ""),
                            tool_call_id=msg.get("tool_call_id", ""),
                            content=content,
                        )
                    ]
                )
            )

    return result


def model_messages_to_session_messages(
    model_messages: list[ModelMessage],
) -> list[dict[str, Any]]:
    """Convert PydanticAI ModelMessage list back to nanobot session dicts.

    This is the inverse of ``session_messages_to_model_messages`` and is used
    to persist PydanticAI's ``result.new_messages()`` into the session store.
    """
    from pydantic_ai.messages import RetryPromptPart, ThinkingPart

    result: list[dict[str, Any]] = []

    for msg in model_messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    result.append({"role": "user", "content": part.content})
                elif isinstance(part, ToolReturnPart):
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "name": part.tool_name,
                            "content": part.content,
                        }
                    )
                elif isinstance(part, RetryPromptPart):
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "name": part.tool_name or "",
                            "content": part.content,
                        }
                    )
        elif isinstance(msg, ModelResponse):
            text_content = ""
            tool_calls: list[dict[str, Any]] = []
            thinking_content = ""
            for part in msg.parts:
                if isinstance(part, TextPart):
                    text_content = part.content
                elif isinstance(part, ToolCallPart):
                    tool_calls.append(
                        {
                            "id": part.tool_call_id,
                            "name": part.tool_name,
                            "arguments": part.args if isinstance(part.args, dict) else {},
                        }
                    )
                elif isinstance(part, ThinkingPart):
                    thinking_content = part.content
            entry: dict[str, Any] = {"role": "assistant", "content": text_content}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if thinking_content:
                entry["reasoning_content"] = thinking_content
            if not text_content and not tool_calls:
                continue
            result.append(entry)

    return result


def persist_new_messages(
    session: Any,
    new_model_messages: list,
    max_tool_result_chars: int = 16000,
) -> None:
    """Convert PydanticAI new_messages() to session dicts and append to session.

    Applies sanitization: tool result truncation, runtime context stripping,
    and multimodal block cleanup.
    """
    from datetime import datetime

    new_session_msgs = model_messages_to_session_messages(new_model_messages)

    for m in new_session_msgs:
        entry = dict(m)
        role, content = entry.get("role"), entry.get("content")

        if role == "assistant" and not content and not entry.get("tool_calls"):
            continue

        if role == "tool":
            if isinstance(content, str) and len(content) > max_tool_result_chars:
                from nanobot.utils.helpers import truncate_text

                entry["content"] = truncate_text(content, max_tool_result_chars)
            elif isinstance(content, list):
                filtered = _sanitize_persisted_blocks(
                    content, max_tool_result_chars, truncate_text=True
                )
                if not filtered:
                    continue
                entry["content"] = filtered

        elif role == "user":
            from nanobot.agent.context import ContextBuilder

            if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                parts = content.split("\n\n", 1)
                if len(parts) > 1 and parts[1].strip():
                    entry["content"] = parts[1]
                else:
                    continue
            if isinstance(content, list):
                filtered = _sanitize_persisted_blocks(
                    content, max_tool_result_chars, drop_runtime=True
                )
                if not filtered:
                    continue
                entry["content"] = filtered

        entry.setdefault("timestamp", datetime.now().isoformat())
        session.messages.append(entry)
    session.updated_at = datetime.now()


def _sanitize_persisted_blocks(
    content: list[dict[str, Any]],
    max_tool_result_chars: int = 16000,
    *,
    truncate_text: bool = False,
    drop_runtime: bool = False,
) -> list[dict[str, Any]]:
    from nanobot.utils.helpers import image_placeholder_text, truncate_text as _truncate_text
    from nanobot.agent.context import ContextBuilder

    filtered: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            filtered.append(block)
            continue
        if (
            drop_runtime
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
        ):
            continue
        if block.get("type") == "image_url" and block.get("image_url", {}).get(
            "url", ""
        ).startswith("data:image/"):
            path = (block.get("_meta") or {}).get("path", "")
            filtered.append({"type": "text", "text": image_placeholder_text(path)})
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text = block["text"]
            if truncate_text and len(text) > max_tool_result_chars:
                text = _truncate_text(text, max_tool_result_chars)
            filtered.append({**block, "text": text})
            continue
        filtered.append(block)
    return filtered


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
        user_message: str,
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
        user_message: str,
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

    async def run_stream_events(
        self,
        user_message: str,
        session: Any | None = None,
        *,
        message_history: list[ModelMessage] | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Stream all events including tool calls and results.

        Yields:
            AgentStreamEvent — can be FunctionToolCallEvent, FunctionToolResultEvent,
            PartStartEvent, PartDeltaEvent, FinalResultEvent, etc.
        """
        if message_history is None:
            session_msgs = session.get_history(max_messages=0) if session else []
            message_history = session_messages_to_model_messages(session_msgs)

        async for event in self._agent.run_stream_events(
            user_message,
            message_history=message_history,
        ):
            yield event

    def get_history(
        self,
        session: Any,
        max_messages: int = 500,
    ) -> list[ModelMessage]:
        """Get message history for a session in PydanticAI format."""
        session_msgs = session.get_history(max_messages=max_messages) if session else []
        return session_messages_to_model_messages(session_msgs)
