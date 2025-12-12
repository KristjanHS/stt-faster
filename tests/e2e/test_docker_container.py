"""End-to-end tests for Docker container build and runtime."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.docker,
    pytest.mark.slow,
]

COMPOSE_FILE = Path("docker/docker-compose.yml")
PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_docker_compose(
    *args: str,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """Run docker compose command with standard options."""
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=capture_output,
        text=text,
    )
    if result.stdout:
        logger.debug("stdout: %s", result.stdout)
    if result.stderr:
        logger.debug("stderr: %s", result.stderr)
    return result


def exec_in_container(
    *cmd: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a command inside the app container."""
    return run_docker_compose("exec", "-T", "app", *cmd, check=check)


def _check_docker_daemon() -> None:
    """Ensure Docker daemon is reachable before running tests."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        if result.returncode != 0:
            pytest.fail("Docker daemon not reachable (docker info failed).")
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.fail(f"Docker daemon unavailable: {exc}")


@pytest.fixture(scope="module")
def docker_container() -> Generator[None, None, None]:
    """Build and start the Docker container, tear down after tests."""
    _check_docker_daemon()

    logger.info("Building Docker image...")
    run_docker_compose("build")

    logger.info("Starting container...")
    run_docker_compose("up", "-d")

    # Wait for container to become healthy
    max_wait = 60  # seconds
    interval = 2
    elapsed = 0

    while elapsed < max_wait:
        result = run_docker_compose("ps", "-q", "app")
        if result.stdout.strip():
            # Check if healthy
            inspect_result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Health.Status}}", result.stdout.strip()],
                capture_output=True,
                text=True,
                check=False,
            )
            health_status = inspect_result.stdout.strip()
            logger.info("Container health status: %s", health_status)

            if health_status == "healthy":
                logger.info("Container is healthy")
                break
        time.sleep(interval)
        elapsed += interval
    else:
        # Timeout - dump logs and fail
        logger.error("Container failed to become healthy within %d seconds", max_wait)
        logs = run_docker_compose("logs", check=False)
        logger.error("Container logs:\n%s", logs.stdout)
        pytest.fail("Container did not become healthy in time")

    yield

    # Teardown
    logger.info("Stopping and removing container...")
    # Never use -v flag for production containers per project rules
    run_docker_compose("down", check=False)


class TestDockerRuntime:
    """Test Docker container runtime behavior."""

    def test_healthcheck_command(self, docker_container: None) -> None:
        """Verify the healthcheck command works."""
        result = exec_in_container("python", "-m", "backend.main", "--healthcheck")
        assert result.returncode == 0
        assert "Healthcheck OK" in result.stdout


class TestDockerPythonEnvironment:
    """Test Python environment inside container."""

    def test_non_root_user(self, docker_container: None) -> None:
        """Verify container runs as non-root user."""
        result = exec_in_container("whoami")
        assert result.stdout.strip() == "appuser"

    def test_hf_cache_directory(self, docker_container: None) -> None:
        """Verify HuggingFace cache directory exists."""
        result = exec_in_container("ls", "-ld", "/hf_cache")
        assert "appuser" in result.stdout


class TestDockerVolumes:
    """Test Docker volume mounts."""

    def test_logs_volume_mount(self, docker_container: None) -> None:
        """Verify logs directory is mounted and writable."""
        # Try to write a test file
        test_file = "docker_test_" + str(int(time.time())) + ".log"
        result = exec_in_container("touch", f"/app/logs/{test_file}")
        assert result.returncode == 0

        # Verify it exists on host
        host_log_path = PROJECT_ROOT / "logs" / test_file
        assert host_log_path.exists(), "Log file not created on host"

        # Cleanup
        host_log_path.unlink()
