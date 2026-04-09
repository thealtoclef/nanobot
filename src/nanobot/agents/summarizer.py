"""SummarizerAgent — produces recursive conversation summaries."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class SummarizerResult(BaseModel):
    summary: str = Field(
        description=(
            "A comprehensive recursive summary. "
            "If there is an existing summary, incorporate it with the new conversation. "
            "Start with a timestamp [YYYY-MM-DD HH:MM]. "
            "Include enough detail for context recovery. "
            "If the existing summary contains errors or omissions, correct them."
        ),
    )


@dataclass
class SummarizerDeps:
    """Dependencies injected into the summarizer agent."""

    existing_summary: str
    formatted_messages: str


class SummarizerAgent:
    """Wraps a pydantic_ai.Agent for conversation summarization."""

    def __init__(self):
        self._agent = Agent(
            output_type=SummarizerResult,
            deps_type=SummarizerDeps,
            instructions=(
                "You are a conversation summarizer. Analyze the conversation and produce a comprehensive summary.\n"
                "If there is an existing summary, incorporate it with the new conversation.\n"
                "Start with a timestamp [YYYY-MM-DD HH:MM].\n"
                "Include enough detail for context recovery.\n"
                "If the existing summary contains errors or omissions, correct them in your new summary.\n"
                "Be concise but thorough. Focus on decisions, topics, and key information."
            ),
            retries=2,
        )

        @self._agent.instructions
        def _build_context(ctx: RunContext[SummarizerDeps]) -> str:
            parts = []
            if ctx.deps.existing_summary:
                parts.append(f"## Existing Summary\n{ctx.deps.existing_summary}")
            parts.append(f"## Conversation to Summarize\n{ctx.deps.formatted_messages}")
            return "\n\n".join(parts)

    @property
    def agent(self):
        """Access the underlying pydantic_ai.Agent for .override() etc."""
        return self._agent

    async def run(self, *, user_prompt: str, deps: SummarizerDeps, model):
        """Run the summarization agent."""
        return await self._agent.run(user_prompt=user_prompt, deps=deps, model=model)

    def override(self, **kwargs):
        """Pass-through to pydantic_ai.Agent.override()."""
        return self._agent.override(**kwargs)


# Module-level singleton for backward compat and convenience
_summarizer_agent = SummarizerAgent()
