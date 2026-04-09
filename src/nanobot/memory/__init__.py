"""Memory system: storage + compression."""

from nanobot.agents.extractor import (
    ExtractorAgent,
    ExtractorDeps,
    ExtractorResult,
    FactItem,
    _extractor_agent,
)
from nanobot.agents.helpers import format_messages_for_text as format_messages_for_summarizer

# Backward compat module-level singletons
from nanobot.agents.summarizer import (
    SummarizerAgent,
    SummarizerDeps,
    SummarizerResult,
    _summarizer_agent,
)
from nanobot.memory.compressor import HistoryCompressor
from nanobot.memory.fact_store import FactStore
from nanobot.memory.history_store import HistoryStore

__all__ = [
    "HistoryCompressor",
    "HistoryStore",
    "FactStore",
    "SummarizerAgent",
    "SummarizerDeps",
    "SummarizerResult",
    "ExtractorAgent",
    "ExtractorDeps",
    "ExtractorResult",
    "FactItem",
    "format_messages_for_summarizer",
    "_summarizer_agent",
    "_extractor_agent",
]
