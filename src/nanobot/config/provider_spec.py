"""Provider backend to pydantic-ai model class mapping.

This is the only hardcoded coupling between provider backend names and
pydantic-ai model classes in the system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

if TYPE_CHECKING:
    from nanobot.config.schema import Config

# Backend name -> (model_class, provider_class)
BACKEND_CLASSES: dict[str, tuple[type, type]] = {
    "anthropic": (AnthropicModel, AnthropicProvider),
    "openai": (OpenAIChatModel, OpenAIProvider),
}


def resolve_model(
    model_name: str,
    backend: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Any:
    """Construct a pydantic-ai model instance from config.

    Args:
        model_name: Model name (e.g. "claude-sonnet-4-5").
        backend: Backend name (e.g. "anthropic").
        base_url: API base URL.
        api_key: API key (empty string if not provided).
        temperature: Model temperature setting.
        max_tokens: Max tokens setting.

    Returns:
        A pydantic-ai model instance.
    """
    pair = BACKEND_CLASSES.get(backend)
    if pair is None:
        supported = ", ".join(sorted(BACKEND_CLASSES))
        raise ValueError(
            f"Unknown backend '{backend}'. Supported backends: {supported}.\n"
            f"Note: For OpenAI-compatible providers (deepseek, groq, etc.), use "
            f"backend='openai' and set base_url to the provider's API endpoint.\n"
            f"For example: backend='openai', base_url='https://api.deepseek.com/v1'"
        )
    model_cls, provider_cls = pair

    settings: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Always create provider — base_url is required, api_key="" is passed as-is (no env fallback).
    provider = provider_cls(api_key=api_key, base_url=base_url)

    return model_cls(
        model_name,
        provider=provider,
        settings=settings,
    )


def resolve_agent_models(config: "Config") -> list[Any]:
    """Resolve all agent models from config.

    Args:
        config: The application configuration.

    Returns:
        A list of pydantic-ai model instances.
    """
    resolved: list[Any] = []
    for mc in config.agent.models:
        provider_cfg = getattr(config.providers, mc.provider, None)
        if provider_cfg is None:
            raise ValueError(f"Unknown provider '{mc.provider}' referenced by model '{mc.name}'")
        resolved.append(
            resolve_model(
                model_name=mc.name,
                backend=provider_cfg.backend,
                base_url=provider_cfg.base_url,
                api_key=provider_cfg.get_api_key(),
                temperature=mc.temperature,
                max_tokens=mc.max_tokens,
            )
        )
    return resolved
