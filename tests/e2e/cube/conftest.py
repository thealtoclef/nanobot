"""E2E Cube test configuration using pytest-docker."""

import pytest


@pytest.fixture(scope="session")
def docker_compose_file() -> str:
    """Return path to docker-compose.yml."""
    import os

    return os.path.join(os.path.dirname(__file__), "fixtures", "compose.yaml")


@pytest.fixture(scope="session")
def cube_service_url(docker_ip, docker_services):
    """Return the Cube service URL from docker network.

    Uses port_for to get the actual mapped port.
    Waits for the Cube API /readyz endpoint to respond.
    """
    import socket

    # Get the mapped port for the cube service
    port = docker_services.port_for("cube", 4000)

    # Wait for the Cube API to be responsive by checking /readyz endpoint
    def check():
        try:
            with socket.create_connection(("localhost", port), timeout=2):
                # TCP connection works, now check HTTP endpoint
                import http.client

                conn = http.client.HTTPConnection("localhost", port, timeout=5)
                try:
                    conn.request("GET", "/readyz")
                    resp = conn.getresponse()
                    return resp.status == 200
                finally:
                    conn.close()
        except (OSError, socket.timeout, Exception):
            return False

    docker_services.wait_until_responsive(check=check, timeout=90, pause=2)

    return f"http://localhost:{port}"


@pytest.fixture(scope="session")
async def cube_container(docker_services):
    """Start the Cube container and ensure it's ready."""
    docker_services.start("cube")
    yield
    docker_services.stop("cube")


@pytest.fixture
async def cube_service_init(cube_service_url, cube_container):
    """Create an initialized CubeService for testing."""
    from nanobot.config.schema import CubeConfig
    from nanobot.cube.service import CubeService

    service = CubeService(
        CubeConfig(
            cube_url=cube_service_url,
            token="secret",
            cubejs_api_path="/cubejs-api",
            timeout=30.0,
            request_span_enabled=True,
            continue_wait_retry_interval=1.0,
            continue_wait_retry_max_attempts=5,
        )
    )
    await service.initialize()
    yield service
    await service.close()
