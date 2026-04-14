from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List

import openai
from chromadb.api.types import EmbeddingFunction

if TYPE_CHECKING:
    from nanobot.config.schema import EmbedderConfig


class CubeEmbeddingFunction(EmbeddingFunction[List[str]]):
    """Adapter bridging nanobot EmbedderConfig to ChromaDB's EmbeddingFunction protocol.

    Supports OpenAI-compatible backends via base_url.
    """

    def __init__(self, config: "EmbedderConfig") -> None:
        self.config = config
        self._client: "openai.AsyncOpenAI | None" = None

    def _get_client(self) -> "openai.AsyncOpenAI":
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.provider.get_api_key(),
                base_url=self.config.provider.base_url,
            )
        return self._client

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of texts. Required by ChromaDB EmbeddingFunction protocol."""
        return asyncio.run(self._embed_async(input))

    @staticmethod
    def name() -> str:
        """Return embedding function name for ChromaDB compatibility."""
        return "cube-embedder"

    @staticmethod
    def build_from_config(config: dict) -> "CubeEmbeddingFunction":
        """Build embedding function from serialized config."""
        raise NotImplementedError("CubeEmbeddingFunction cannot be rebuilt from config alone")

    def get_config(self) -> dict:
        """Return configuration dict for ChromaDB persistence."""
        return {
            "name": self.name(),
            "model": self.config.model,
            "provider": self.config.provider.backend,
        }

    async def _embed_async(self, texts: List[str]) -> List[List[float]]:
        if self.config.provider.backend != "openai":
            raise ValueError(
                f"Unsupported embedding backend: {self.config.provider.backend}. Only 'openai' backend is supported."
            )
        client = self._get_client()
        resp = await client.embeddings.create(model=self.config.model, input=texts)
        return [item.embedding for item in resp.data]
