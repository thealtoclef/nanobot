"""Tests for mem0 integration in nanobot."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.config.schema import MemoryConfig
from nanobot.context import ContextBuilder
from nanobot.db import Database, upgrade_db
from nanobot.memory.mem0_client import Mem0Client


# ---------------------------------------------------------------------------
# TestMem0ClientFormatting
# ---------------------------------------------------------------------------


class TestMem0ClientFormatting:
    def test_format_memories_empty(self):
        result = Mem0Client.format_memories_for_prompt([])
        assert result == ""

    def test_format_memories_with_score(self):
        memories = [
            {"memory": "User likes Python", "score": 0.85},
            {"memory": "User is a developer", "score": 0.72},
        ]
        result = Mem0Client.format_memories_for_prompt(memories)
        assert "## Relevant Memories" in result
        assert "[0.85] User likes Python" in result
        assert "[0.72] User is a developer" in result

    def test_format_memories_without_score(self):
        memories = [{"memory": "User likes pizza"}]
        result = Mem0Client.format_memories_for_prompt(memories)
        assert "## Relevant Memories" in result
        assert "- User likes pizza" in result


# ---------------------------------------------------------------------------
# TestMem0ClientConfig
# ---------------------------------------------------------------------------


class TestMem0ClientConfig:
    def test_default_config(self, tmp_path: Path):
        config = MemoryConfig()
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()

        assert mem0_config["version"] == "v1.1"
        assert mem0_config["vector_store"]["provider"] == "chroma"
        assert mem0_config["vector_store"]["config"]["path"] == str(tmp_path / "chroma")
        assert mem0_config["vector_store"]["config"]["collection_name"] == "mem0"
        assert mem0_config["history_db_path"] == str(tmp_path / "mem0.db")
        assert mem0_config["llm"] == {}
        assert mem0_config["embedder"] == {}
        assert "reranker" not in mem0_config

    def test_llm_passthrough(self, tmp_path: Path):
        config = MemoryConfig(
            llm={"provider": "openai", "config": {"model": "gpt-4o", "api_key": "sk-test"}}
        )
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()

        assert mem0_config["llm"]["provider"] == "openai"
        assert mem0_config["llm"]["config"]["model"] == "gpt-4o"
        assert mem0_config["llm"]["config"]["api_key"] == "sk-test"

    def test_embedder_passthrough(self, tmp_path: Path):
        config = MemoryConfig(
            embedder={
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            }
        )
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()

        assert mem0_config["embedder"]["provider"] == "openai"
        assert mem0_config["embedder"]["config"]["model"] == "text-embedding-3-small"

    def test_reranker_included_when_non_empty(self, tmp_path: Path):
        config = MemoryConfig(
            reranker={"provider": "llm_reranker", "config": {"model": "gpt-4o-mini", "top_k": 5}}
        )
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()

        assert "reranker" in mem0_config
        assert mem0_config["reranker"]["provider"] == "llm_reranker"
        assert mem0_config["reranker"]["config"]["model"] == "gpt-4o-mini"
        assert mem0_config["reranker"]["config"]["top_k"] == 5

    def test_reranker_excluded_when_empty(self, tmp_path: Path):
        config = MemoryConfig(reranker={})
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()

        assert "reranker" not in mem0_config


# ---------------------------------------------------------------------------
# TestContextBuilderMemoryBlock
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestMemoryDisabledPath
# ---------------------------------------------------------------------------
# TestMemoryDisabledPath
# ---------------------------------------------------------------------------


class TestMemoryDisabledPath:
    def test_runner_imports_without_mem0(self):
        """AgentRunner should be importable even when mem0 is not installed."""
        # This test verifies that without the [memory] extra, the runner
        # still imports without error
        from nanobot.runner import AgentRunner

        # AgentRunner class exists and is callable
        assert AgentRunner is not None

    def test_memory_config_defaults_disabled(self):
        """MemoryConfig should default to enabled=False."""
        from nanobot.config.schema import MemoryConfig

        config = MemoryConfig()
        assert config.enabled is False
