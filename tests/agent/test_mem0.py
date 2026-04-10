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
        assert mem0_config["vector_store"]["config"]["path"] == str(tmp_path / "mem0_chroma")
        assert mem0_config["vector_store"]["config"]["collection_name"] == "nanobot_memory"
        assert mem0_config["history_db_path"] == str(tmp_path / "memories.db")
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

    def test_memory_config_defaults_disabled(self):
        """MemoryConfig should default to enabled=False."""
        from nanobot.config.schema import MemoryConfig

        config = MemoryConfig()
        assert config.enabled is False
