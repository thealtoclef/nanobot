"""mem0 memory client — thin wrapper bridging MemoryConfig → AsyncMemory."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanobot.config.schema import MemoryConfig, MemoryLLMConfig

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

    def _resolve_api_key(self, api_key: str, api_key_env: str) -> str:
        if api_key:
            return api_key
        if api_key_env:
            return os.getenv(api_key_env, "")
        return ""

    def _build_llm_config(self) -> dict[str, Any]:
        llm = self._config.llm
        backend = llm.backend
        result: dict[str, Any] = {"provider": backend}

        api_key = self._resolve_api_key(llm.api_key, llm.api_key_env)
        result["config"] = {"model": llm.model or "gpt-4o-mini"}
        if api_key:
            result["config"]["api_key"] = api_key

        if backend == "ollama":
            result["config"]["ollama_base_url"] = llm.base_url
        else:
            result["config"]["openai_base_url"] = llm.base_url

        return result

    def _build_embedder_config(self) -> dict[str, Any]:
        emb = self._config.embedder
        backend = emb.backend
        result: dict[str, Any] = {"provider": backend}

        api_key = self._resolve_api_key(emb.api_key, emb.api_key_env)
        result["config"] = {"model": emb.model or "text-embedding-3-small"}
        if api_key:
            result["config"]["api_key"] = api_key

        if backend == "ollama":
            result["config"]["ollama_base_url"] = emb.base_url
        else:
            result["config"]["openai_base_url"] = emb.base_url

        return result

    def _build_reranker_config(self) -> dict[str, Any] | None:
        """Build reranker config if reranker_enabled is True in MemoryConfig."""
        if not self._config.reranker_enabled:
            return None

        reranker = self._config.reranker

        api_key = self._resolve_api_key(reranker.api_key, reranker.api_key_env)
        config: dict[str, Any] = {
            "provider": reranker.type,
            "model": reranker.model,
        }
        if api_key:
            config["api_key"] = api_key
        if reranker.top_k is not None:
            config["top_k"] = reranker.top_k
        if reranker.temperature is not None:
            config["temperature"] = reranker.temperature

        # For llm_reranker, pass nested LLM config
        if reranker.type == "llm_reranker" and reranker.llm is not None:
            config["llm"] = self._build_llm_config_for_reranker(reranker.llm)

        return config

    def _build_llm_config_for_reranker(self, llm: "MemoryLLMConfig") -> dict[str, Any]:
        """Build LLM config dict for llm_reranker nested config."""
        backend = llm.backend
        api_key = self._resolve_api_key(llm.api_key, llm.api_key_env)
        result: dict[str, Any] = {"provider": backend}
        result["config"] = {"model": llm.model or "gpt-4o-mini"}
        if api_key:
            result["config"]["api_key"] = api_key
        if backend == "ollama":
            result["config"]["ollama_base_url"] = llm.base_url
        else:
            result["config"]["openai_base_url"] = llm.base_url
        return result

    def _build_mem0_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "version": "v1.1",
            "llm": self._build_llm_config(),
            "embedder": self._build_embedder_config(),
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "nanobot_memory",
                    "path": str(self._workspace / "mem0_chroma"),
                },
            },
            "history_db_path": str(self._workspace / "memories.db"),
        }

        reranker = self._build_reranker_config()
        if reranker:
            config["reranker"] = reranker

        return config

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
