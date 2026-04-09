"""ExtractorAgent — extracts facts from conversations."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class FactItem(BaseModel):
    content: str = Field(description="A single factual statement.")
    category: str = Field(description="Either 'fact' or 'preference'.")


class ExtractorResult(BaseModel):
    facts: list[FactItem] = Field(
        description="List of facts extracted from the conversation.",
        default_factory=list,
    )


@dataclass
class ExtractorDeps:
    """Dependencies injected into the extractor agent."""

    formatted_messages: str
    existing_facts: str


class ExtractorAgent:
    """Wraps a pydantic_ai.Agent for fact extraction."""

    def __init__(self):
        self._agent = Agent(
            output_type=ExtractorResult,
            deps_type=ExtractorDeps,
            instructions=(
                "You are a fact extractor. Analyze the conversation and extract individual facts.\n"
                "For each fact, provide:\n"
                "- content: A clear, self-contained factual statement\n"
                "- category: Either 'fact' (objective information) or 'preference' (user preferences)\n\n"
                "Rules:\n"
                "- Extract ONLY genuinely new facts not already in the existing facts list\n"
                "- Skip trivial observations (greetings, confirmations)\n"
                "- Each fact should be atomic and independently useful\n"
                "- If no new facts are found, return an empty list"
            ),
            retries=1,
        )

        @self._agent.instructions
        def _build_context(ctx: RunContext[ExtractorDeps]) -> str:
            parts = [f"## Conversation\n{ctx.deps.formatted_messages}"]
            if ctx.deps.existing_facts:
                parts.append(
                    f"## Existing Facts (do NOT duplicate these)\n{ctx.deps.existing_facts}"
                )
            return "\n\n".join(parts)

    @property
    def agent(self):
        """Access the underlying pydantic_ai.Agent for .override() etc."""
        return self._agent

    async def run(self, *, user_prompt: str, deps: ExtractorDeps, model):
        """Run the fact extraction agent."""
        return await self._agent.run(user_prompt=user_prompt, deps=deps, model=model)

    def override(self, **kwargs):
        """Pass-through to pydantic_ai.Agent.override()."""
        return self._agent.override(**kwargs)


# Module-level singleton for backward compat and convenience
_extractor_agent = ExtractorAgent()
