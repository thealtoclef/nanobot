"""Fixtures for Cube tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_health_ok():
    """Mock GET /readyz → 200 {"health": "HEALTH"}"""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"health": "HEALTH"}
    return response


@pytest.fixture
def mock_health_fail():
    """Mock GET /readyz → connection error"""
    return ConnectionError("Connection refused")


@pytest.fixture
def mock_health_down():
    """Mock GET /readyz → 500 {"health": "DOWN"}"""
    response = MagicMock()
    response.status_code = 500
    response.json.return_value = {"health": "DOWN"}
    return response


@pytest.fixture
def mock_live_ok():
    """Mock GET /livez → 200 {"health": "HEALTH"}"""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"health": "HEALTH"}
    return response


@pytest.fixture
def mock_meta_response():
    """Mock GET /v1/meta → sample cube schema JSON"""
    return {
        "cubes": [
            {
                "name": "orders",
                "title": "Orders",
                "description": "Order transactions",
                "dimensions": [
                    {
                        "name": "orders.order_id",
                        "type": "number",
                        "description": "Unique order identifier",
                    },
                    {
                        "name": "orders.status",
                        "type": "string",
                        "description": "Current order status",
                    },
                ],
                "measures": [
                    {
                        "name": "orders.order_count",
                        "aggType": "count",
                        "description": "Total number of orders",
                    },
                ],
            },
            {
                "name": "customers",
                "title": "Customers",
                "description": "Customer master data",
                "dimensions": [
                    {
                        "name": "customers.customer_id",
                        "type": "number",
                        "description": "Unique customer identifier",
                    },
                    {
                        "name": "customers.name",
                        "type": "string",
                        "description": "Customer full name",
                    },
                ],
                "measures": [
                    {
                        "name": "customers.customer_count",
                        "aggType": "count",
                        "description": "Total number of customers",
                    },
                ],
            },
        ]
    }


@pytest.fixture
def mock_load_success():
    """Mock POST /v1/load → sample data response"""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {"orders.status": "completed", "orders.order_count": 150},
            {"orders.status": "pending", "orders.order_count": 42},
        ]
    }
    return response


@pytest.fixture
def mock_load_error():
    """Mock POST /v1/load → error response"""
    response = MagicMock()
    response.status_code = 400
    response.raise_for_status.side_effect = Exception("400 Bad Request")
    return response


@pytest.fixture
def mock_load_continue_wait():
    """Mock POST /v1/load → 202 {"error": "Continue wait"}"""
    response = MagicMock()
    response.status_code = 202
    response.json.return_value = {"error": "Continue wait"}
    return response
