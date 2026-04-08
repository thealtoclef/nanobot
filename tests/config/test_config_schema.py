from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from nanobot.config.schema import (
    AgentConfig,
    ApiConfig,
    ChannelsConfig,
    Config,
    ExecToolConfig,
    GatewayConfig,
    HeartbeatConfig,
    MCPServerConfig,
    ModelConfig,
    ObservabilityConfig,
    ProviderConfig,
    ProvidersConfig,
    ToolsConfig,
    WebSearchConfig,
    WebToolsConfig,
)


class TestProviderConfig:
    def test_accepts_valid_config(self) -> None:
        pc = ProviderConfig(backend="openai", base_url="https://api.openai.com", api_key="sk-test")
        assert pc.backend == "openai"
        assert pc.api_key == "sk-test"

    def test_default_api_key_empty(self) -> None:
        pc = ProviderConfig(backend="openai", base_url="https://api.openai.com")
        assert pc.api_key == ""

    def test_rejects_both_api_key_and_env(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ProviderConfig(
                backend="openai",
                base_url="https://api.openai.com",
                api_key="sk-test",
                api_key_env="MY_API_KEY",
            )

    def test_get_api_key_returns_direct(self) -> None:
        pc = ProviderConfig(
            backend="openai", base_url="https://api.openai.com", api_key="sk-direct"
        )
        assert pc.get_api_key() == "sk-direct"

    def test_get_api_key_returns_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_PROVIDER_KEY", "sk-from-env")
        pc = ProviderConfig(
            backend="openai", base_url="https://api.openai.com", api_key_env="MY_PROVIDER_KEY"
        )
        assert pc.get_api_key() == "sk-from-env"

    def test_get_api_key_returns_empty_when_none_set(self) -> None:
        pc = ProviderConfig(backend="openai", base_url="https://api.openai.com")
        assert pc.get_api_key() == ""

    def test_accepts_anthropic_backend(self) -> None:
        pc = ProviderConfig(backend="anthropic", base_url="https://api.anthropic.com")
        assert pc.backend == "anthropic"


class TestProvidersConfig:
    def test_default_has_openai(self) -> None:
        pc = ProvidersConfig()
        assert pc.openai.backend == "openai"

    def test_accepts_extra_providers(self) -> None:
        pc = ProvidersConfig.model_validate(
            {
                "deepseek": {
                    "backend": "openai",
                    "base_url": "https://api.deepseek.com",
                    "api_key": "sk-ds",
                }
            }
        )
        assert hasattr(pc, "deepseek")

    def test_validates_provider_configs(self) -> None:
        with pytest.raises(ValidationError):
            ProvidersConfig.model_validate({"bad_provider": {"backend": "openai"}})


class TestModelConfig:
    def test_defaults(self) -> None:
        mc = ModelConfig(name="gpt-5", provider="openai")
        assert mc.temperature == 0.1
        assert mc.max_tokens == 8192

    def test_custom_values(self) -> None:
        mc = ModelConfig(name="claude", provider="anthropic", temperature=0.5, max_tokens=4096)
        assert mc.temperature == 0.5
        assert mc.max_tokens == 4096


class TestAgentConfig:
    def test_defaults(self) -> None:
        ac = AgentConfig()
        assert ac.context_window_tokens == 65_536
        assert ac.max_tool_iterations == 200
        assert ac.timezone == "UTC"
        assert ac.workspace == "~/.nanobot/workspace"

    def test_custom_models(self) -> None:
        ac = AgentConfig(models=[ModelConfig(name="gpt-5", provider="openai")])
        assert len(ac.models) == 1
        assert ac.models[0].name == "gpt-5"


class TestChannelsConfig:
    def test_defaults(self) -> None:
        cc = ChannelsConfig()
        assert cc.send_progress is True
        assert cc.send_tool_hints is False
        assert cc.send_max_retries == 3

    def test_accepts_extra_channel_fields(self) -> None:
        cc = ChannelsConfig.model_validate({"telegram": {"token": "123"}})
        assert hasattr(cc, "telegram")

    def test_send_max_retries_bounds(self) -> None:
        ChannelsConfig(send_max_retries=0)
        ChannelsConfig(send_max_retries=10)
        with pytest.raises(ValidationError):
            ChannelsConfig(send_max_retries=11)
        with pytest.raises(ValidationError):
            ChannelsConfig(send_max_retries=-1)


class TestHeartbeatConfig:
    def test_defaults(self) -> None:
        hc = HeartbeatConfig()
        assert hc.enabled is True
        assert hc.interval_s == 1800
        assert hc.keep_recent_messages == 8


class TestApiConfig:
    def test_defaults(self) -> None:
        ac = ApiConfig()
        assert ac.host == "127.0.0.1"
        assert ac.port == 8900
        assert ac.timeout == 120.0


class TestGatewayConfig:
    def test_defaults(self) -> None:
        gc = GatewayConfig()
        assert gc.host == "0.0.0.0"
        assert gc.port == 18790
        assert isinstance(gc.heartbeat, HeartbeatConfig)


class TestWebSearchConfig:
    def test_defaults(self) -> None:
        wsc = WebSearchConfig()
        assert wsc.provider == "duckduckgo"
        assert wsc.max_results == 5


class TestWebToolsConfig:
    def test_defaults(self) -> None:
        wtc = WebToolsConfig()
        assert wtc.enable is True
        assert wtc.proxy is None
        assert isinstance(wtc.search, WebSearchConfig)


class TestExecToolConfig:
    def test_defaults(self) -> None:
        etc = ExecToolConfig()
        assert etc.enable is True
        assert etc.timeout == 60
        assert etc.path_append == ""


class TestMCPServerConfig:
    def test_stdio_config(self) -> None:
        mc = MCPServerConfig(command="npx", args=["-y", "@mcp/server"])
        assert mc.command == "npx"
        assert mc.args == ["-y", "@mcp/server"]

    def test_http_config(self) -> None:
        mc = MCPServerConfig(url="https://mcp.example.com", headers={"Authorization": "Bearer x"})
        assert mc.url == "https://mcp.example.com"

    def test_default_enabled_tools(self) -> None:
        mc = MCPServerConfig()
        assert mc.enabled_tools == ["*"]

    def test_custom_enabled_tools(self) -> None:
        mc = MCPServerConfig(enabled_tools=["read_file", "write_file"])
        assert mc.enabled_tools == ["read_file", "write_file"]


class TestToolsConfig:
    def test_defaults(self) -> None:
        tc = ToolsConfig()
        assert tc.restrict_to_workspace is False
        assert isinstance(tc.web, WebToolsConfig)
        assert isinstance(tc.exec, ExecToolConfig)
        assert tc.mcp_servers == {}


class TestObservabilityConfig:
    def test_defaults(self) -> None:
        oc = ObservabilityConfig()
        assert oc.enabled is False
        assert oc.service_name == "nanobot"


class TestConfig:
    def test_defaults(self) -> None:
        config = Config()
        assert isinstance(config.agent, AgentConfig)
        assert isinstance(config.providers, ProvidersConfig)
        assert isinstance(config.channels, ChannelsConfig)
        assert isinstance(config.api, ApiConfig)
        assert isinstance(config.gateway, GatewayConfig)
        assert isinstance(config.tools, ToolsConfig)
        assert isinstance(config.observability, ObservabilityConfig)

    def test_workspace_path_expansion(self) -> None:
        config = Config()
        assert config.workspace_path == Path.home() / ".nanobot" / "workspace"

    def test_custom_workspace_path(self) -> None:
        config = Config(agent=AgentConfig(workspace="/tmp/nanobot-ws"))
        assert config.workspace_path == Path("/tmp/nanobot-ws")

    def test_camel_case_accepted(self) -> None:
        config = Config.model_validate({"api": {"host": "0.0.0.0", "port": 9000}})
        assert config.api.host == "0.0.0.0"
        assert config.api.port == 9000


class TestConfigFromDict:
    def test_full_config_round_trip(self) -> None:
        data = {
            "agent": {
                "models": [{"name": "gpt-5", "provider": "openai", "temperature": 0.3}],
                "contextWindowTokens": 32000,
                "timezone": "US/Eastern",
            },
            "providers": {
                "openai": {
                    "backend": "openai",
                    "baseUrl": "https://api.openai.com",
                    "apiKey": "sk-test",
                }
            },
            "channels": {
                "sendProgress": False,
                "telegram": {"token": "BOT_TOKEN", "enabled": True},
            },
            "tools": {
                "restrictToWorkspace": True,
                "exec": {"enable": False},
            },
        }
        config = Config.model_validate(data)
        assert config.agent.models[0].temperature == 0.3
        assert config.agent.context_window_tokens == 32000
        assert config.agent.timezone == "US/Eastern"
        assert config.channels.send_progress is False
        assert config.tools.restrict_to_workspace is True
        assert config.tools.exec.enable is False
