"""Backward compat — use nanobot._identity directly."""

from nanobot._identity import build_identity, _PROMPTS_DIR

__all__ = ["build_identity", "_PROMPTS_DIR"]
