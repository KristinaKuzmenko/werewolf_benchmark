"""
Pytest configuration for A2A conformance tests.
"""
import httpx
import pytest

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


def pytest_addoption(parser):
    parser.addoption(
        "--agent-url",
        default="http://localhost:9009",
        help="Agent URL (default: http://localhost:9009)",
    )


@pytest.fixture(scope="session")
def agent(request):
    """Agent URL fixture. Skips tests if agent is not running."""
    url = request.config.getoption("--agent-url")

    try:
        response = httpx.get(f"{url}/.well-known/agent-card.json", timeout=2)
        if response.status_code != 200:
            pytest.skip(
                f"Agent at {url} returned status {response.status_code}. Start agent to run integration tests."
            )
    except Exception as e:
        pytest.skip(f"Could not connect to agent at {url}: {e}. Start agent to run integration tests.")

    return url
