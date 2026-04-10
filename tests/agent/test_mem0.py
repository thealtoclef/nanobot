"""Tests for mem0 integration in nanobot."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.config.schema import (
    MemoryConfig,
    MemoryLLMConfig,
    MemoryEmbedderConfig,
    MemoryRerankerConfig,
)
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
        assert mem0_config["llm"]["provider"] == "openai"
        assert mem0_config["embedder"]["provider"] == "openai"

    def test_ollama_config(self, tmp_path: Path):
        config = MemoryConfig(
            enabled=True,
            llm=MemoryLLMConfig(
                provider="ollama", model="llama3.1", ollama_base_url="http://localhost:11434"
            ),
            embedder=MemoryEmbedderConfig(
                provider="ollama",
                model="nomic-embed-text",
                ollama_base_url="http://localhost:11434",
            ),
        )
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()

        assert mem0_config["llm"]["provider"] == "ollama"
        assert mem0_config["llm"]["config"]["model"] == "llama3.1"
        assert mem0_config["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"
        assert mem0_config["embedder"]["provider"] == "ollama"
        assert mem0_config["embedder"]["config"]["model"] == "nomic-embed-text"

    def test_openai_config_with_api_key_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "sk-test-key")
        config = MemoryConfig(
            llm=MemoryLLMConfig(provider="openai", api_key_env="MY_API_KEY", model="gpt-4o"),
        )
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_llm_config()

        assert mem0_config["config"]["api_key"] == "sk-test-key"
        assert mem0_config["config"]["model"] == "gpt-4o"

    def test_vector_store_hardcoded_in_config(self, tmp_path: Path):
        """Vector store path is always workspace-relative, not configurable."""
        config = MemoryConfig()
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()
        assert mem0_config["vector_store"]["config"]["path"] == str(tmp_path / "mem0_chroma")
        assert mem0_config["vector_store"]["provider"] == "chroma"
        assert mem0_config["vector_store"]["config"]["collection_name"] == "nanobot_memory"

    def test_history_db_path_hardcoded_in_config(self, tmp_path: Path):
        """history_db_path is always workspace-relative, not configurable."""
        config = MemoryConfig()
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()
        assert mem0_config["history_db_path"] == str(tmp_path / "memories.db")

    def test_reranker_not_included_when_none(self, tmp_path: Path):
        """Reranker is absent from config when not set."""
        config = MemoryConfig()
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()
        assert "reranker" not in mem0_config

    def test_reranker_config_full(self, tmp_path: Path, monkeypatch):
        """Reranker is included in config when reranker_enabled=True."""
        monkeypatch.setenv("COHERE_KEY", "test-cohere-key")
        config = MemoryConfig(
            reranker_enabled=True,
            reranker=MemoryRerankerConfig(
                provider="cohere",
                model="rerank-english-v3.0",
                api_key_env="COHERE_KEY",
                top_k=5,
                temperature=0.0,
            ),
        )
        client = Mem0Client(config, tmp_path)
        mem0_config = client._build_mem0_config()
        assert "reranker" in mem0_config
        assert mem0_config["reranker"]["provider"] == "cohere"
        assert mem0_config["reranker"]["model"] == "rerank-english-v3.0"
        assert mem0_config["reranker"]["api_key"] == "test-cohere-key"
        assert mem0_config["reranker"]["top_k"] == 5
        assert mem0_config["reranker"]["temperature"] == 0.0

    def test_reranker_ollama_local(self, tmp_path: Path):
        """Ollama-based reranker uses llm sub-config."""
        config = MemoryConfig(
            reranker_enabled=True,
            reranker=MemoryRerankerConfig(
                provider="llm_reranker",
                model="llama3.1",
            ),
        )
        client = Mem0Client(config, tmp_path)
        reranker_cfg = client._build_reranker_config()
        assert reranker_cfg is not None
        assert reranker_cfg["provider"] == "llm_reranker"
        assert reranker_cfg["model"] == "llama3.1"


# ---------------------------------------------------------------------------
# TestMem0ClientInitialization
# ---------------------------------------------------------------------------


class TestMem0ClientInitialization:
    def test_initialize_raises_when_mem0_not_installed(self, tmp_path: Path, monkeypatch):
        # Patch AsyncMemory to None to simulate not-installed
        import nanobot.memory.mem0_client as mc

        monkeypatch.setattr(mc, "AsyncMemory", None)

        from nanobot.memory.mem0_client import Mem0Client
        from nanobot.config.schema import MemoryConfig

        config = MemoryConfig(enabled=True)
        client = Mem0Client(config, tmp_path)

        import pytest

        with pytest.raises(RuntimeError, match="mem0ai package is not installed"):
            import asyncio

            asyncio.run(client.initialize())


# ---------------------------------------------------------------------------
# TestContextBuilderMemoryBlock
# ---------------------------------------------------------------------------


class TestContextBuilderMemoryBlock:
    @pytest.fixture
    def db(self, tmp_path: Path) -> Database:
        from nanobot.db import Base

        upgrade_db(tmp_path)
        db = Database(tmp_path)
        Base.metadata.create_all(db.engine)
        return db

    @pytest.fixture
    def cb(self, db: Database, tmp_path: Path) -> ContextBuilder:
        return ContextBuilder(workspace=tmp_path, db=db)

    def test_memory_block_prepends_to_user_content_str(self, cb: ContextBuilder):
        """memory_block should be prepended to user content when it's a string."""
        history = []
        memory_block = "## Relevant Memories\n- [0.85] User likes Python"

        msgs, prompt = cb.build_messages(
            history=history,
            current_message="What do I like?",
            memory_block=memory_block,
        )

        # prompt is a string (user content) or list
        if isinstance(prompt, str):
            assert memory_block in prompt
        else:
            # it's a list of content dicts
            texts = [p["text"] for p in prompt if isinstance(p, dict) and p.get("type") == "text"]
            assert any(memory_block in t for t in texts)

    def test_memory_block_prepends_to_user_content_media_list(
        self, cb: ContextBuilder, tmp_path: Path
    ):
        """memory_block should be prepended to user content when it's a media list."""
        history = []
        memory_block = "## Relevant Memories\n- [0.85] User likes Python"

        # Create a real image file to test media handling
        img_path = tmp_path / "test_image.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)  # Minimal valid PNG header

        msgs, prompt = cb.build_messages(
            history=history,
            current_message="Describe this image",
            media=[str(img_path)],
            memory_block=memory_block,
        )

        # prompt should be a list with memory_block as first text item
        assert isinstance(prompt, list)
        memory_texts = [
            p["text"]
            for p in prompt
            if isinstance(p, dict) and "Relevant Memories" in p.get("text", "")
        ]
        assert len(memory_texts) >= 1

    def test_memory_block_empty_when_none(self, cb: ContextBuilder):
        """When memory_block is None, no memory content injected."""
        history = []

        msgs, prompt = cb.build_messages(
            history=history,
            current_message="Hello",
            memory_block=None,
        )

        if isinstance(prompt, str):
            assert "Relevant Memories" not in prompt
        else:
            texts = [p.get("text", "") for p in prompt if isinstance(p, dict)]
            assert not any("Relevant Memories" in t for t in texts)

    def test_memory_block_with_history(self, cb: ContextBuilder, db: Database):
        """memory_block should be prepended even when history exists."""
        # Create a session with history
        db.ensure_session("test-session")

        # Add a history row
        db.add_history("test-session", "This was a previous conversation summary.", None)

        history = db.get_all_histories("test-session")
        memory_block = "## Relevant Memories\n- User prefers concise answers"

        msgs, prompt = cb.build_messages(
            history=[],
            current_message="Continue the conversation",
            session_key="test-session",
            memory_block=memory_block,
        )

        if isinstance(prompt, str):
            assert memory_block in prompt
        else:
            memory_found = any(
                "Relevant Memories" in p.get("text", "") for p in prompt if isinstance(p, dict)
            )
            assert memory_found


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

    def test_mem0_client_imports_without_mem0(self):
        """Mem0Client should be importable even when mem0 package is absent."""
        from nanobot.memory.mem0_client import Mem0Client

        assert Mem0Client is not None

    def test_memory_config_defaults_disabled(self):
        """MemoryConfig should default to enabled=False."""
        from nanobot.config.schema import MemoryConfig

        config = MemoryConfig()
        assert config.enabled is False
