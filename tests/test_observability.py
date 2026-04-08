"""Tests for nanobot.observability — logfire SDK + OTEL setup."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from nanobot.observability import setup


class TestSetupDisabled:
    """When observability is disabled, only loguru is configured."""

    @patch("nanobot.observability.logger")
    def test_configures_loguru_stderr_handler(self, mock_logger) -> None:
        setup(enabled=False, log_level="DEBUG")
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        call_kwargs = mock_logger.add.call_args
        assert call_kwargs[1]["level"] == "DEBUG"

    @patch("nanobot.observability.logger")
    def test_does_not_call_logfire_when_disabled(self, mock_logger) -> None:
        with patch("nanobot.observability.logfire") as mock_logfire:
            setup(enabled=False)
            mock_logfire.configure.assert_not_called()
            mock_logfire.instrument_pydantic_ai.assert_not_called()
            mock_logfire.instrument_httpx.assert_not_called()

    @patch("nanobot.observability.logger")
    def test_returns_early_no_env_vars_set(self, mock_logger) -> None:
        with patch.dict(os.environ, {}, clear=True):
            setup(enabled=False)
            assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ


class TestSetupEnabled:
    """When observability is enabled, logfire is configured."""

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_configures_logfire(self, mock_logfire, mock_logger) -> None:
        setup(enabled=True, service_name="test-svc")
        mock_logfire.configure.assert_called_once_with(service_name="test-svc")

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_instruments_pydantic_ai_and_httpx(self, mock_logfire, mock_logger) -> None:
        setup(enabled=True)
        mock_logfire.instrument_pydantic_ai.assert_called_once()
        mock_logfire.instrument_httpx.assert_called_once_with(capture_all=True)

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_sets_traces_endpoint_env_var(self, mock_logfire, mock_logger) -> None:
        with patch.dict(os.environ, {}, clear=True):
            setup(
                enabled=True,
                traces_endpoint="http://localhost:4318/v1/traces",
            )
            assert (
                os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
                == "http://localhost:4318/v1/traces"
            )

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_sets_metrics_endpoint_env_var(self, mock_logfire, mock_logger) -> None:
        with patch.dict(os.environ, {}, clear=True):
            setup(
                enabled=True,
                metrics_endpoint="http://localhost:4318/v1/metrics",
            )
            assert (
                os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]
                == "http://localhost:4318/v1/metrics"
            )

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_sets_logs_endpoint_env_var(self, mock_logfire, mock_logger) -> None:
        with patch.dict(os.environ, {}, clear=True):
            setup(
                enabled=True,
                logs_endpoint="http://localhost:4318/v1/logs",
            )
            assert os.environ["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == "http://localhost:4318/v1/logs"

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_does_not_set_env_vars_when_empty(self, mock_logfire, mock_logger) -> None:
        with patch.dict(os.environ, {}, clear=True):
            setup(enabled=True, traces_endpoint="", metrics_endpoint="", logs_endpoint="")
            assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in os.environ
            assert "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT" not in os.environ
            assert "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT" not in os.environ

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_adds_logfire_loguru_handler(self, mock_logfire, mock_logger) -> None:
        mock_handler = MagicMock()
        mock_logfire.loguru_handler.return_value = mock_handler

        setup(enabled=True, log_level="WARNING")

        handler_call = mock_logger.add.call_args_list[-1]
        assert handler_call[0][0] is mock_handler
        assert handler_call[1]["level"] == "WARNING"
        assert handler_call[1]["format"] == "{message}"

    @patch("nanobot.observability.logger")
    @patch("nanobot.observability.logfire")
    def test_logs_info_when_enabled(self, mock_logfire, mock_logger) -> None:
        setup(enabled=True)
        mock_logger.info.assert_called_once()
        assert "enabled" in mock_logger.info.call_args[0][0].lower()


class TestSetupDefaults:
    """Default parameter values."""

    def test_default_enabled_is_false(self) -> None:
        import inspect

        sig = inspect.signature(setup)
        assert sig.parameters["enabled"].default is False

    def test_default_log_level_is_info(self) -> None:
        import inspect

        sig = inspect.signature(setup)
        assert sig.parameters["log_level"].default == "INFO"

    def test_default_service_name_is_nanobot(self) -> None:
        import inspect

        sig = inspect.signature(setup)
        assert sig.parameters["service_name"].default == "nanobot"

    def test_default_endpoints_are_empty(self) -> None:
        import inspect

        sig = inspect.signature(setup)
        assert sig.parameters["traces_endpoint"].default == ""
        assert sig.parameters["metrics_endpoint"].default == ""
        assert sig.parameters["logs_endpoint"].default == ""
