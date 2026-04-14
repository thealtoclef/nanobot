"""Tests for CubeSchemaIndex."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nanobot.config.schema import CubeSchemaIndexConfig, EmbedderConfig, ProviderConfig
from nanobot.cube.schema_index import CubeSchemaIndex


@pytest.fixture
def index_config():
    embedder_cfg = EmbedderConfig(
        provider=ProviderConfig(
            backend="openai",
            base_url="https://api.openai.com",
            api_key="test-key",
        ),
        model="text-embedding-3-small",
    )
    return CubeSchemaIndexConfig(
        enabled=True,
        threshold=30_000,
        max_results=10,
        embedder=embedder_cfg,
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_cubes():
    return [
        {
            "name": "orders",
            "title": "Orders",
            "description": "Order transactions",
            "dimensions": [
                {
                    "name": "orders.order_id",
                    "type": "number",
                    "description": "Unique order identifier",
                },
                {"name": "orders.status", "type": "string", "description": "Current order status"},
            ],
            "measures": [
                {
                    "name": "orders.order_count",
                    "aggType": "count",
                    "description": "Total number of orders",
                },
            ],
        },
        {
            "name": "customers",
            "title": "Customers",
            "description": "Customer master data",
            "dimensions": [
                {
                    "name": "customers.customer_id",
                    "type": "number",
                    "description": "Unique customer identifier",
                },
            ],
            "measures": [
                {
                    "name": "customers.customer_count",
                    "aggType": "count",
                    "description": "Total number of customers",
                },
            ],
        },
    ]


class TestCubeSchemaIndexInit:
    def test_directory_created(self, index_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            assert (temp_dir / "chroma").exists()

    def test_collection_exists(self, index_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            assert index.is_available is True


class TestCubeSchemaIndexIndex:
    @pytest.mark.asyncio
    async def test_index_cubes_from_meta(self, index_config, temp_dir, sample_cubes):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            await index.index_cubes(sample_cubes)

            # Verify collection.delete and collection.add were called
            mock_collection.delete.assert_called_once()
            assert mock_collection.add.called

    @pytest.mark.asyncio
    async def test_reindex_clears_old_data(self, index_config, temp_dir, sample_cubes):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            await index.index_cubes(sample_cubes)
            await index.index_cubes(sample_cubes[:1])  # Re-index with only first cube

            # delete should be called twice (once for each index_cubes call)
            assert mock_collection.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_index_empty_cubes(self, index_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            await index.index_cubes([])  # Should not raise

            # delete should still be called
            mock_collection.delete.assert_called_once()


class TestCubeSchemaIndexSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, index_config, temp_dir, sample_cubes):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            # Mock query response
            mock_collection.query.return_value = {
                "documents": [["order data"]],
                "metadatas": [[{"cube_name": "orders", "item_type": "cube"}]],
                "distances": [[0.1]],
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            results = await index.search("order status", limit=10)
            assert len(results) >= 1
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 3  # (text, metadata, score)

    @pytest.mark.asyncio
    async def test_search_limit_respected(self, index_config, temp_dir, sample_cubes):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "documents": [["order data"]],
                "metadatas": [[{"cube_name": "orders", "item_type": "cube"}]],
                "distances": [[0.1]],
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            results = await index.search("orders", limit=1)

            # Verify n_results was passed correctly
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args
            assert call_kwargs.kwargs.get("n_results") == 1 or call_kwargs[1].get("n_results") == 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, index_config, temp_dir, sample_cubes):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "documents": [],
                "metadatas": [],
                "distances": [],
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            results = await index.search("nonexistent cube xyz", limit=10)
            assert len(results) == 0


class TestCubeSchemaIndexClear:
    @pytest.mark.asyncio
    async def test_clear_all_items(self, index_config, temp_dir, sample_cubes):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            await index.index_cubes(sample_cubes)
            await index.clear()

            mock_collection.delete.assert_called()

    @pytest.mark.asyncio
    async def test_clear_empty(self, index_config, temp_dir):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

            index = CubeSchemaIndex(
                persist_dir=temp_dir / "chroma",
                max_results=index_config.max_results,
                embedder=index_config.embedder,
                reranker=index_config.reranker,
            )
            index.initialize()

            await index.clear()  # Should not raise
