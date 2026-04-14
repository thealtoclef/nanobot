"""Tests for SqlMemory."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.config.schema import CubeMemoryConfig, EmbedderConfig, ProviderConfig
from nanobot.cube.sql_memory import SqlMemory


@pytest.fixture
def memory_config():
    embedder_cfg = EmbedderConfig(
        provider=ProviderConfig(
            backend="openai",
            base_url="https://api.openai.com",
            api_key="test-key",
        ),
        model="text-embedding-3-small",
    )
    return CubeMemoryConfig(
        enabled=True,
        max_results=5,
        embedder=embedder_cfg,
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestSqlMemoryInit:
    def test_directory_created(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            assert (temp_dir / "chroma").exists()

    def test_collection_exists(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            assert memory.is_available is True


class TestSqlMemoryStore:
    @pytest.mark.asyncio
    async def test_store_question_payload(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            await memory.store("How many orders?", '{"query": {"measures": ["orders.count"]}}')

            # Verify collection.add was called
            mock_collection.add.assert_called_once()
            call_kwargs = mock_collection.add.call_args
            documents = call_kwargs.kwargs.get("documents") or call_kwargs[1].get("documents")
            assert "How many orders?" in documents

    @pytest.mark.asyncio
    async def test_store_empty_question(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            await memory.store("", '{"query": {}}')

            # Should still call add with empty string
            mock_collection.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_special_chars(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            await memory.store(
                "What's the count? (special chars: @#$%)", '{"query": {"measures": ["test"]}}'
            )

            mock_collection.add.assert_called_once()


class TestSqlMemorySearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "documents": [["How many orders by status?"]],
                "metadatas": [[{"payload": '{"query": {"dimensions": ["status"]}}'}]],
                "distances": [[0.05]],
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            results = await memory.search("order status", limit=5)
            assert len(results) >= 1
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 3  # (question, payload, score)
            assert results[0][1] == '{"query": {"dimensions": ["status"]}}'

    @pytest.mark.asyncio
    async def test_search_limit_respected(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "documents": [["Query 0"]],
                "metadatas": [[{"payload": '{"query": 0}'}]],
                "distances": [[0.1]],
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            await memory.search("query", limit=2)

            # Verify n_results was passed correctly
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args
            n_results = call_kwargs.kwargs.get("n_results") or call_kwargs[1].get("n_results")
            assert n_results == 2

    @pytest.mark.asyncio
    async def test_search_no_results(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            results = await memory.search("nonexistent query xyz", limit=5)
            assert len(results) == 0


class TestSqlMemoryClear:
    @pytest.mark.asyncio
    async def test_clear_all(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            await memory.clear()

            mock_collection.delete.assert_called_once()
            call_kwargs = mock_collection.delete.call_args
            where_filter = call_kwargs.kwargs.get("where") or call_kwargs[1].get("where")
            assert where_filter == {}

    @pytest.mark.asyncio
    async def test_clear_empty(self, memory_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            memory = SqlMemory(
                persist_dir=temp_dir / "chroma",
                max_results=memory_config.max_results,
                embedder=memory_config.embedder,
                reranker=memory_config.reranker,
            )
            memory.initialize()

            await memory.clear()  # Should not raise


class TestSqlMemoryAvailable:
    def test_unavailable_when_collection_none(self, temp_dir):
        memory = SqlMemory(
            persist_dir=temp_dir / "chroma",
            max_results=5,
            embedder=None,
            reranker=None,
        )
        # Without initialization, collection is None
        assert memory.is_available is False
