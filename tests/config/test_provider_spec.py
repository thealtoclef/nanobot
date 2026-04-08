"""Tests for nanobot.config.provider_spec — backend-to-model resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nanobot.config.provider_spec import BACKEND_CLASSES, resolve_agent_models, resolve_model
from nanobot.config.schema import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)


# ---------------------------------------------------------------------------
# BACKEND_CLASSES registry
# ---------------------------------------------------------------------------


class TestBackendClasses:
    """Validate the BACKEND_CLASSES registry structure."""

    def test_has_anthropic_and_openai(self) -> None:
        assert "anthropic" in BACKEND_CLASSES
        assert "openai" in BACKEND_CLASSES

    def test_entries_are_two_tuples(self) -> None:
        for backend, pair in BACKEND_CLASSES.items():
            assert isinstance(pair, tuple), f"{backend} value is not a tuple"
            assert len(pair) == 2, f"{backend} tuple must be (model_cls, provider_cls)"

    def test_entries_contain_types(self) -> None:
        for backend, (model_cls, provider_cls) in BACKEND_CLASSES.items():
            assert isinstance(model_cls, type), f"{backend} model_cls is not a type"
            assert isinstance(provider_cls, type), f"{backend} provider_cls is not a type"


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    """Unit tests for resolve_model()."""

    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend 'nonexistent'"):
            resolve_model(
                model_name="test-model",
                backend="nonexistent",
                base_url="http://localhost",
                api_key="key",
                temperature=0.5,
                max_tokens=1024,
            )

    def test_error_message_lists_supported_backends(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            resolve_model(
                model_name="m",
                backend="bogus",
                base_url="http://x",
                api_key="k",
                temperature=0.1,
                max_tokens=100,
            )
        msg = str(exc_info.value)
        assert "anthropic" in msg
        assert "openai" in msg

    @patch.dict(
        "nanobot.config.provider_spec.BACKEND_CLASSES",
        {"anthropic": (MagicMock(name="AnthropicModel"), MagicMock(name="AnthropicProvider"))},
    )
    def test_anthropic_backend_creates_correct_classes(self) -> None:
        mock_model_cls, mock_prov_cls = BACKEND_CLASSES["anthropic"]
        mock_prov_instance = MagicMock()
        mock_prov_cls.return_value = mock_prov_instance
        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance

        result = resolve_model(
            model_name="claude-sonnet-4-5",
            backend="anthropic",
            base_url="https://api.anthropic.com",
            api_key="sk-test-key",
            temperature=0.3,
            max_tokens=4096,
        )

        mock_prov_cls.assert_called_once_with(
            api_key="sk-test-key",
            base_url="https://api.anthropic.com",
        )
        mock_model_cls.assert_called_once_with(
            "claude-sonnet-4-5",
            provider=mock_prov_instance,
            settings={"temperature": 0.3, "max_tokens": 4096},
        )
        assert result is mock_model_instance

    @patch.dict(
        "nanobot.config.provider_spec.BACKEND_CLASSES",
        {"openai": (MagicMock(name="OpenAIChatModel"), MagicMock(name="OpenAIProvider"))},
    )
    def test_openai_backend_creates_correct_classes(self) -> None:
        mock_model_cls, mock_prov_cls = BACKEND_CLASSES["openai"]
        mock_prov_instance = MagicMock()
        mock_prov_cls.return_value = mock_prov_instance
        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance

        result = resolve_model(
            model_name="gpt-5",
            backend="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-openai-key",
            temperature=0.7,
            max_tokens=2048,
        )

        mock_prov_cls.assert_called_once_with(
            api_key="sk-openai-key",
            base_url="https://api.openai.com/v1",
        )
        mock_model_cls.assert_called_once_with(
            "gpt-5",
            provider=mock_prov_instance,
            settings={"temperature": 0.7, "max_tokens": 2048},
        )
        assert result is mock_model_instance

    @patch.dict(
        "nanobot.config.provider_spec.BACKEND_CLASSES",
        {"openai": (MagicMock(name="OpenAIChatModel"), MagicMock(name="OpenAIProvider"))},
    )
    def test_empty_api_key_passed_as_is(self) -> None:
        _, mock_prov_cls = BACKEND_CLASSES["openai"]
        mock_prov_cls.return_value = MagicMock()

        resolve_model(
            model_name="deepseek-chat",
            backend="openai",
            base_url="https://api.deepseek.com/v1",
            api_key="",
            temperature=0.1,
            max_tokens=8192,
        )

        mock_prov_cls.assert_called_once_with(
            api_key="",
            base_url="https://api.deepseek.com/v1",
        )


# ---------------------------------------------------------------------------
# resolve_agent_models
# ---------------------------------------------------------------------------


class TestResolveAgentModels:
    """Unit tests for resolve_agent_models()."""

    def _make_config(self, models: list[dict], providers: dict) -> Config:
        """Build a Config with the given models and provider dicts."""
        provider_configs = {k: ProviderConfig.model_validate(v) for k, v in providers.items()}
        return Config(
            agent=AgentConfig(
                models=[ModelConfig(**m) for m in models],
            ),
            providers=ProvidersConfig.model_validate(provider_configs),
        )

    @patch("nanobot.config.provider_spec.resolve_model")
    def test_resolves_all_models(self, mock_resolve) -> None:
        mock_resolve.return_value = MagicMock()
        config = self._make_config(
            models=[
                {"name": "gpt-5", "provider": "myopenai", "temperature": 0.5, "max_tokens": 4096},
                {
                    "name": "claude-sonnet-4-5",
                    "provider": "myanthropic",
                    "temperature": 0.2,
                    "max_tokens": 8192,
                },
            ],
            providers={
                "myopenai": {
                    "backend": "openai",
                    "base_url": "https://api.openai.com",
                    "api_key": "k1",
                },
                "myanthropic": {
                    "backend": "anthropic",
                    "base_url": "https://api.anthropic.com",
                    "api_key": "k2",
                },
            },
        )

        result = resolve_agent_models(config)

        assert mock_resolve.call_count == 2
        assert len(result) == 2

    @patch("nanobot.config.provider_spec.resolve_model")
    def test_passes_correct_args_to_resolve_model(self, mock_resolve) -> None:
        mock_resolve.return_value = MagicMock(name="resolved_model")
        config = self._make_config(
            models=[
                {"name": "gpt-5", "provider": "myprov", "temperature": 0.5, "max_tokens": 4096},
            ],
            providers={
                "myprov": {
                    "backend": "openai",
                    "base_url": "https://api.example.com",
                    "api_key": "sk-key",
                },
            },
        )

        resolve_agent_models(config)

        mock_resolve.assert_called_once_with(
            model_name="gpt-5",
            backend="openai",
            base_url="https://api.example.com",
            api_key="sk-key",
            temperature=0.5,
            max_tokens=4096,
        )

    def test_unknown_provider_raises_value_error(self) -> None:
        config = self._make_config(
            models=[
                {
                    "name": "gpt-5",
                    "provider": "nonexistent_provider",
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
            ],
            providers={
                "myprov": {
                    "backend": "openai",
                    "base_url": "https://api.example.com",
                    "api_key": "k",
                },
            },
        )

        with pytest.raises(ValueError, match="Unknown provider 'nonexistent_provider'"):
            resolve_agent_models(config)

    @patch("nanobot.config.provider_spec.resolve_model")
    def test_empty_models_list_returns_empty(self, mock_resolve) -> None:
        config = self._make_config(
            models=[],
            providers={
                "myprov": {
                    "backend": "openai",
                    "base_url": "https://api.example.com",
                    "api_key": "k",
                },
            },
        )

        result = resolve_agent_models(config)

        assert result == []
        mock_resolve.assert_not_called()
