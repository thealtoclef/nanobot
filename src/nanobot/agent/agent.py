"""Backward compat — use nanobot.agents.talker directly."""

from nanobot.agents.talker import (
    TalkerAgent,
    Talker as Talker,
    ToolAdapter,
    build_instructions,
    _to_user_content,
    BOOTSTRAP_FILES,
)

__all__ = [
    "TalkerAgent",
    "Talker",
    "ToolAdapter",
    "build_instructions",
    "_to_user_content",
    "BOOTSTRAP_FILES",
]
