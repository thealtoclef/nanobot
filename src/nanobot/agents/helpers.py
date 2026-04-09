"""Shared utilities for agents."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Files that are loaded to build agent identity/instructions
BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]


def format_messages_for_text(messages: list) -> str:
    """Format ModelMessages into readable text for agent prompts.

    Used by SummarizerAgent, ExtractorAgent, and any agent that needs
    to present conversation history as text.
    """
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content if isinstance(c, str))
                    lines.append(f"USER: {content}")
                elif isinstance(part, SystemPromptPart):
                    lines.append(f"SYSTEM: {part.content}")
                elif isinstance(part, ToolReturnPart):
                    lines.append(f"TOOL [{part.tool_name}]: {part.content}")
                elif isinstance(part, RetryPromptPart):
                    lines.append(f"TOOL [{part.tool_name or 'unknown'}]: {part.content}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f"ASSISTANT: {part.content}")
                elif isinstance(part, ToolCallPart):
                    lines.append(f"ASSISTANT [call {part.tool_name}]: {part.args}")
    return "\n".join(lines)
