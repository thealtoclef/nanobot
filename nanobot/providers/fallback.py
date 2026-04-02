"""Fallback provider — tries the primary, then fallback models in order."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from nanobot.providers.base import LLMProvider, LLMResponse


class FallbackProvider(LLMProvider):
    """Wraps a primary provider with an ordered list of fallback providers.

    Each provider (primary and every fallback) uses its own
    ``chat_with_retry`` / ``chat_stream_with_retry``, so transient-error
    retries happen *per provider* before moving to the next fallback.
    """

    def __init__(
        self,
        primary: LLMProvider,
        primary_model: str,
        fallbacks: list[tuple[LLMProvider, str]],
    ) -> None:
        # Skip ABC __init__ — we don't need api_key/api_base on the wrapper.
        super().__init__()
        self._primary = primary
        self._primary_model = primary_model
        self._fallbacks = fallbacks  # [(provider_instance, model_string), ...]
        # Inherit generation settings from primary.
        self.generation = primary.generation

    # ------------------------------------------------------------------
    # Abstract method implementations — delegate to primary
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        return await self._primary.chat(
            messages=messages, tools=tools,
            model=model or self._primary_model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )

    def get_default_model(self) -> str:
        return self._primary_model

    # ------------------------------------------------------------------
    # Retry-with-fallback overrides
    # ------------------------------------------------------------------

    async def chat_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kw: dict[str, Any] = dict(
            messages=messages, tools=tools,
            model=model or self._primary_model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
        )
        response = await self._primary.chat_with_retry(**kw)
        if response.finish_reason != "error":
            return response

        return await self._try_fallbacks(
            method="chat_with_retry", kw=kw, primary_error=response,
        )

    async def chat_stream_with_retry(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: object = LLMProvider._SENTINEL,
        temperature: object = LLMProvider._SENTINEL,
        reasoning_effort: object = LLMProvider._SENTINEL,
        tool_choice: str | dict[str, Any] | None = None,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        kw: dict[str, Any] = dict(
            messages=messages, tools=tools,
            model=model or self._primary_model,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort, tool_choice=tool_choice,
            on_content_delta=on_content_delta,
        )
        response = await self._primary.chat_stream_with_retry(**kw)
        if response.finish_reason != "error":
            return response

        return await self._try_fallbacks(
            method="chat_stream_with_retry", kw=kw, primary_error=response,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _try_fallbacks(
        self,
        method: str,
        kw: dict[str, Any],
        primary_error: LLMResponse,
    ) -> LLMResponse:
        """Iterate through fallback providers until one succeeds."""
        for fb_provider, fb_model in self._fallbacks:
            logger.warning(
                "Primary model failed ({}), trying fallback: {}",
                (primary_error.content or "")[:80], fb_model,
            )
            fb_kw = {**kw, "model": fb_model}
            # Let each fallback provider apply its own generation defaults
            # by passing _SENTINEL for settings that were not explicitly overridden.
            fn = getattr(fb_provider, method)
            response = await fn(**fb_kw)
            if response.finish_reason != "error":
                return response
            logger.warning(
                "Fallback {} also failed: {}", fb_model, (response.content or "")[:80],
            )
        # All fallbacks exhausted — return last error.
        return primary_error
