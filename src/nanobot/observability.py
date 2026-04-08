"""Observability setup via logfire SDK + OTEL-compatible backends.

logfire is pydantic-ai's native observability SDK and is already a dependency
of pydantic-ai. It routes spans, metrics, and logs to any OTEL-compatible
backend via standard OTLP environment variables or explicit endpoints.

Usage::

    from nanobot.observability import setup

    setup(config.observability)
"""

from __future__ import annotations

import os
import sys

import logfire
from loguru import logger


def setup(
    enabled: bool = False,
    log_level: str = "INFO",
    service_name: str = "nanobot",
    traces_endpoint: str = "",
    metrics_endpoint: str = "",
    logs_endpoint: str = "",
) -> None:
    """Configure logging level and optional logfire observability.

    This must be called BEFORE any pydantic-ai agent is created.

    Args:
        enabled: Master switch. If False, basic console logging only.
        log_level: loguru log level (DEBUG, INFO, WARNING, ERROR).
        service_name: Service name used in OTEL resource.
        traces_endpoint: OTLP traces endpoint (e.g. "http://localhost:4318/v1/traces").
        metrics_endpoint: OTLP metrics endpoint.
        logs_endpoint: OTLP logs endpoint.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    if not enabled:
        logger.info(
            "Observability is disabled. Set observability.enabled=true in config to enable."
        )
        return

    # Set OTEL env vars if endpoints are provided (logfire reads these)
    if traces_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = traces_endpoint
    if metrics_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"] = metrics_endpoint
    if logs_endpoint:
        os.environ["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] = logs_endpoint

    logfire.configure(service_name=service_name)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)

    # Route loguru logs through logfire so internal logs are exported
    logger.add(
        logfire.loguru_handler(),
        level=log_level,
        format="{message}",
    )

    logger.info("Observability enabled")
