"""Configuration schema using Pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ChannelsConfig(Base):
    """Configuration for chat channels.

    Built-in and plugin channel configs are stored as extra fields (dicts).
    Each channel parses its own config in __init__.
    Per-channel "streaming": true enables streaming output (requires send_delta impl).
    """

    model_config = ConfigDict(extra="allow")

    send_progress: bool = True
    send_tool_hints: bool = False
    send_max_retries: int = Field(default=3, ge=0, le=10)


class ProviderConfig(Base):
    """LLM provider configuration.

    api_key and api_key_env are mutually exclusive — set one or the other, not both.
    """

    backend: Literal["anthropic", "openai"] = Field(
        description="Backend provider name. Use 'openai' for OpenAI-compatible providers "
        "(deepseek, groq, etc.) by setting base_url accordingly."
    )
    base_url: str = Field(description="API base URL")
    api_key: str = Field(default="", description="Direct API key")
    api_key_env: str = Field(default="", description="Env var name for API key")

    @model_validator(mode="after")
    def check_mutually_exclusive(self) -> "ProviderConfig":
        if self.api_key and self.api_key_env:
            raise ValueError(
                "api_key and api_key_env are mutually exclusive. Set one or the other, not both."
            )
        return self

    def get_api_key(self) -> str:
        """Resolve API key — returns direct value or fetches from env."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env, "")
        return ""


class ProvidersConfig(Base):
    """Configuration for LLM providers.

    Providers are defined by dynamic keys — no fixed fields.
    Each key maps to a ProviderConfig.
    """

    openai: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(
            backend="openai",
            base_url="https://api.openai.com",
        )
    )

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _validate_providers(cls, data: dict) -> dict:
        """Ensure each provider value is a ProviderConfig instance, not a raw dict."""
        if not isinstance(data, dict):
            return data
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result[key] = ProviderConfig.model_validate(value)
            else:
                result[key] = value
        return result


class ModelConfig(Base):
    """A model entry used in the fallback chain."""

    name: str = Field(description="Model name (e.g. claude-sonnet-4-5)")
    provider: str = Field(description="Provider key referencing providers config")
    temperature: float = 0.1
    max_tokens: int = 8192


class AgentConfig(Base):
    """Agent configuration — models list + agent-level settings."""

    models: list[ModelConfig] = Field(
        default_factory=lambda: [ModelConfig(name="gpt-5", provider="openai")]
    )
    context_window_tokens: int = 65_536
    max_tool_iterations: int = 200
    max_tool_result_chars: int = 16_000
    reasoning_effort: str | None = None
    timezone: str = "UTC"
    workspace: str = "~/.nanobot/workspace"


class HeartbeatConfig(Base):
    """Heartbeat service configuration."""

    enabled: bool = True
    interval_s: int = 30 * 60
    keep_recent_messages: int = 8


class ApiConfig(Base):
    """OpenAI-compatible API server configuration."""

    host: str = "127.0.0.1"
    port: int = 8900
    timeout: float = 120.0


class GatewayConfig(Base):
    """Gateway/server configuration."""

    host: str = "0.0.0.0"
    port: int = 18790
    heartbeat: HeartbeatConfig = Field(default_factory=HeartbeatConfig)


class WebSearchConfig(Base):
    """Web search tool configuration."""

    provider: str = "duckduckgo"
    api_key: str = ""
    base_url: str = ""
    max_results: int = 5


class WebToolsConfig(Base):
    """Web tools configuration."""

    enable: bool = True
    proxy: str | None = None
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ObservabilityConfig(Base):
    """Observability configuration via logfire SDK + OTEL backends."""

    enabled: bool = False
    log_level: str = "INFO"
    service_name: str = "nanobot"
    traces_endpoint: str = ""
    metrics_endpoint: str = ""
    logs_endpoint: str = ""


class ExecToolConfig(Base):
    """Shell exec tool configuration."""

    enable: bool = True
    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP server connection configuration (stdio or HTTP)."""

    type: Literal["stdio", "sse", "streamableHttp"] | None = None
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    tool_timeout: int = 30
    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])


class ToolsConfig(Base):
    """Tools configuration."""

    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class Config(BaseSettings):
    """Root configuration for nanobot."""

    agent: AgentConfig = Field(default_factory=AgentConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @property
    def workspace_path(self) -> Path:
        """Get expanded workspace path."""
        return Path(self.agent.workspace).expanduser()

    model_config = ConfigDict(env_prefix="NANOBOT_", env_nested_delimiter="__")
