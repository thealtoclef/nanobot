"""mem0 memory client — thin wrapper bridging MemoryConfig → AsyncMemory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.config.schema import MemoryConfig

try:
    from mem0 import AsyncMemory
except ImportError:
    AsyncMemory = None  # type: ignore[assignment,misc]


class Mem0Client:
    """Wraps mem0 AsyncMemory with nanobot config integration."""

    def __init__(self, config: MemoryConfig, workspace: Path):
        self._config = config
        self._workspace = workspace
        self._client: AsyncMemory | None = None

    def _build_vector_store_path(self) -> Path:
        path = self._config.vector_store_path.strip()
        if path:
            return Path(path)
        return self._workspace / "mem0_chroma"

    def _build_history_db_path(self) -> Path:
        path = self._config.history_db_path.strip()
        if path:
            return Path(path)
        return self._workspace / "memories.db"

    def _resolve_api_key(self, api_key: str, api_key_env: str) -> str:
        if api_key:
            return api_key
        if api_key_env:
            return os.getenv(api_key_env, "")
        return ""

    def _build_llm_config(self) -> dict[str, Any]:
        llm = self._config.llm
        provider = llm.provider
        result: dict[str, Any] = {"provider": provider}

        if provider == "ollama":
            result["config"] = {
                "model": llm.model or "llama3.1",
                "ollama_base_url": llm.ollama_base_url or "http://localhost:11434",
            }
        elif provider in ("openai", "anthropic"):
            api_key = self._resolve_api_key(llm.api_key, llm.api_key_env)
            result["config"] = {"model": llm.model or "gpt-4o-mini"}
            if api_key:
                result["config"]["api_key"] = api_key
            if llm.base_url:
                result["config"]["openai_base_url"] = llm.base_url

        return result

    def _build_embedder_config(self) -> dict[str, Any]:
        emb = self._config.embedder
        provider = emb.provider
        result: dict[str, Any] = {"provider": provider}

        if provider == "ollama":
            result["config"] = {
                "model": emb.model or "nomic-embed-text",
                "ollama_base_url": emb.ollama_base_url or "http://localhost:11434",
            }
        elif provider == "openai":
            api_key = self._resolve_api_key(emb.api_key, emb.api_key_env)
            result["config"] = {"model": emb.model or "text-embedding-3-small"}
            if api_key:
                result["config"]["api_key"] = api_key
            if emb.base_url:
                result["config"]["openai_base_url"] = emb.base_url

        return result

    def _build_mem0_config(self) -> dict[str, Any]:
        return {
            "version": "v1.1",
            "llm": self._build_llm_config(),
            "embedder": self._build_embedder_config(),
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "nanobot_memory",
                    "path": str(self._build_vector_store_path()),
                },
            },
            "history_db_path": str(self._build_history_db_path()),
        }

    async def initialize(self) -> None:
        """Initialize AsyncMemory from config. Must be called before use."""
        if AsyncMemory is None:
            raise RuntimeError(
                "mem0ai package is not installed. Install with: pip install nanobot[memory]"
            )
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
