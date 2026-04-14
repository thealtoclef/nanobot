from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import openai
from chromadb.api.types import Embeddings

if TYPE_CHECKING:
    from nanobot.config.schema import EmbedderConfig


class CubeEmbeddingFunction(Embeddings):
    """Adapter bridging nanobot EmbedderConfig to ChromaDB's Embeddings protocol.

    Supports OpenAI-compatible backends via base_url.
    """

    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self._client: "openai.AsyncOpenAI | None" = None

    def _get_client(self) -> "openai.AsyncOpenAI":
        if self._client is None:
            self._client = openai.AsyncOpenAI(
                api_key=self.config.provider.get_api_key(),
                base_url=self.config.provider.base_url,
            )
        return self._client

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return asyncio.run(self._embed_async(input))

    def embed_query(self, query: str) -> list[float]:
        return asyncio.run(self._embed_async([query]))[0]

    async def _embed_async(self, texts: list[str]) -> list[list[float]]:
        if self.config.provider.backend != "openai":
            raise ValueError(
                f"Unsupported embedding backend: {self.config.provider.backend}. Only 'openai' backend is supported."
            )
        client = self._get_client()
        resp = await client.embeddings.create(model=self.config.model, input=texts)
        return [item.embedding for item in resp.data]
