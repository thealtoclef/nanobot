"""mem0 memory client — thin wrapper bridging MemoryConfig → AsyncMemory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.config.schema import MemoryConfig

from mem0 import AsyncMemory


class Mem0Client:
    """Wraps mem0 AsyncMemory with nanobot config integration."""

    def __init__(self, config: MemoryConfig, workspace: Path):
        self._config = config
        self._workspace = workspace
        self._client: AsyncMemory | None = None

    def _build_mem0_config(self) -> dict[str, Any]:
        """Build mem0 config — passthrough llm/embedder/reranker, hardcode vector store."""
        config: dict[str, Any] = {
            "version": "v1.1",
            "llm": self._config.llm,
            "embedder": self._config.embedder,
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "nanobot_memory",
                    "path": str(self._workspace / "mem0_chroma"),
                },
            },
            "history_db_path": str(self._workspace / "memories.db"),
        }

        if self._config.reranker:
            config["reranker"] = self._config.reranker

        return config

    async def initialize(self) -> None:
        """Initialize AsyncMemory from config. Must be called before use."""
        self._client = AsyncMemory.from_config(self._build_mem0_config())

    @property
    def client(self) -> AsyncMemory:
        if self._client is None:
            raise RuntimeError("Mem0Client.initialize() must be called before use")
        return self._client

    async def add(
        self,
        session_key: str,
        messages: list[dict[str, str]],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add conversation messages to memory."""
        await self._client.add(
            user_id=session_key,
            messages=messages,
            metadata=metadata or {},
        )

    async def search(
        self,
        session_key: str,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories for a session, return top-K results."""
        result = await self._client.search(
            query=query,
            user_id=session_key,
            limit=limit,
        )
        return result.get("results", [])

    @staticmethod
    def format_memories_for_prompt(memories: list[dict[str, Any]]) -> str:
        """Format memories as a readable string block for system prompt injection."""
        if not memories:
            return ""
        lines = ["## Relevant Memories"]
        for m in memories:
            memory_text = m.get("memory", "")
            score = m.get("score")
            if score is not None:
                lines.append(f"- [{score:.2f}] {memory_text}")
            else:
                lines.append(f"- {memory_text}")
        return "\n".join(lines)
