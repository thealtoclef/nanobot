"""Agent dependencies — per-run context passed via PydanticAI RunContext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentDeps:
    """Per-run dependencies injected into every agent run via ctx.deps.

    All fields have defaults to allow default-construction when deps_type is
    set on the agent but no deps are explicitly passed.
    """

    session_key: str = ""
    channel: str = "cli"
    chat_id: str = "direct"
    message_id: str | None = None
    mem0_client: Any = None
