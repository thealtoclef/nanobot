"""Backward compat — use nanobot.agents, nanobot.runner, etc. directly."""

from nanobot.agents.talker import TalkerAgent
from nanobot.agents.talker import Talker as Talker  # compat alias
from nanobot.context import ContextBuilder
from nanobot.memory.compressor import HistoryCompressor as HistorySummarizer
from nanobot.memory.compressor import MemoryStore
from nanobot.runner import AgentRunner
from nanobot.skill_loader import SkillsLoader
from nanobot.subagent import SubagentManager

__all__ = [
    "AgentRunner",
    "ContextBuilder",
    "HistoryCompressor",
    "HistorySummarizer",
    "MemoryStore",
    "SkillsLoader",
    "SubagentManager",
    "Talker",
    "TalkerAgent",
]
