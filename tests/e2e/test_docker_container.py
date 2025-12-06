"""End-to-end tests for Docker container build and runtime.

These tests verify that:
1. The Docker image builds successfully
2. The container starts and becomes healthy
3. Commands can be executed inside the container
4. The Python environment is correctly configured
5. Tests can run inside the container
"""

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

# Path to docker-compose file
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


@pytest.fixture(scope="module")
def docker_container() -> Generator[None, None, None]:
    """Build and start the Docker container, tear down after tests."""
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


class TestDockerBuild:
    """Test Docker image build process."""

    def test_dockerfile_exists(self) -> None:
        """Verify Dockerfile exists."""
        dockerfile = PROJECT_ROOT / "docker" / "app.Dockerfile"
        assert dockerfile.exists(), f"Dockerfile not found at {dockerfile}"

    def test_compose_file_exists(self) -> None:
        """Verify docker-compose file exists."""
        assert COMPOSE_FILE.exists(), f"Docker compose file not found at {COMPOSE_FILE}"

    def test_build_succeeds(self, docker_container: None) -> None:
        """Verify the Docker image builds successfully."""
        # The docker_container fixture builds the image
        # If we get here, the build succeeded
        result = run_docker_compose("images", "app")
        assert "stt-faster-dev" in result.stdout


class TestDockerRuntime:
    """Test Docker container runtime behavior."""

    def test_container_is_running(self, docker_container: None) -> None:
        """Verify the container is running."""
        result = run_docker_compose("ps", "-q", "app")
        container_id = result.stdout.strip()
        assert container_id, "Container is not running"

    def test_container_is_healthy(self, docker_container: None) -> None:
        """Verify the container passes health checks."""
        result = run_docker_compose("ps")
        assert "(healthy)" in result.stdout, "Container is not healthy"

    def test_healthcheck_command(self, docker_container: None) -> None:
        """Verify the healthcheck command works."""
        result = exec_in_container("python", "-m", "backend.main", "--healthcheck")
        assert result.returncode == 0
        assert "Healthcheck OK" in result.stdout

    def test_main_process_running(self, docker_container: None) -> None:
        """Verify the main process is running."""
        logs = run_docker_compose("logs", "app")
        assert "backend.main started" in logs.stdout


class TestDockerPythonEnvironment:
    """Test Python environment inside container."""

    def test_python_version(self, docker_container: None) -> None:
        """Verify correct Python version is installed."""
        result = exec_in_container("python", "--version")
        assert "Python 3.12" in result.stdout

    def test_python_from_venv(self, docker_container: None) -> None:
        """Verify Python is running from the virtual environment."""
        result = exec_in_container("which", "python")
        assert "/opt/venv/bin/python" in result.stdout

    def test_venv_in_path(self, docker_container: None) -> None:
        """Verify venv is in Python path."""
        result = exec_in_container("python", "-c", "import sys; print(sys.path)")
        assert "/opt/venv/lib/python3.12/site-packages" in result.stdout

    def test_backend_module_accessible(self, docker_container: None) -> None:
        """Verify the backend module can be imported."""
        result = exec_in_container("python", "-c", "import backend; print(backend.__file__)")
        assert "/app/backend" in result.stdout

    def test_non_root_user(self, docker_container: None) -> None:
        """Verify container runs as non-root user."""
        result = exec_in_container("whoami")
        assert result.stdout.strip() == "appuser"

    def test_app_directory_structure(self, docker_container: None) -> None:
        """Verify app directory structure is correct."""
        result = exec_in_container("ls", "-1", "/app")
        expected_dirs = ["backend", "frontend", "tests", "scripts", "logs", "reports"]
        for directory in expected_dirs:
            assert directory in result.stdout, f"Missing directory: {directory}"

    def test_hf_cache_directory(self, docker_container: None) -> None:
        """Verify HuggingFace cache directory exists."""
        result = exec_in_container("ls", "-ld", "/hf_cache")
        assert "appuser" in result.stdout


class TestDockerTestExecution:
    """Test that tests can run inside the container."""

    def test_pytest_available(self, docker_container: None) -> None:
        """Verify pytest is installed and accessible."""
        result = exec_in_container("/opt/venv/bin/python", "-m", "pytest", "--version")
        assert "pytest" in result.stdout

    def test_unit_tests_discoverable(self, docker_container: None) -> None:
        """Verify unit tests can be discovered."""
        result = exec_in_container(
            "/opt/venv/bin/python",
            "-m",
            "pytest",
            "tests/unit",
            "--collect-only",
            "-q",
        )
        assert "test" in result.stdout
        # Should have discovered some tests
        assert "collected" in result.stdout

    def test_sample_unit_test_runs(self, docker_container: None) -> None:
        """Verify a sample unit test can run successfully."""
        result = exec_in_container(
            "/opt/venv/bin/python",
            "-m",
            "pytest",
            "tests/unit/test_main.py::TestHealthcheckMode::test_healthcheck_exits_successfully",
            "-v",
        )
        assert result.returncode == 0
        assert "PASSED" in result.stdout


class TestDockerVolumes:
    """Test Docker volume mounts."""

    def test_backend_volume_mount(self, docker_container: None) -> None:
        """Verify backend code is accessible (via volume mount)."""
        result = exec_in_container("ls", "-la", "/app/backend")
        assert "main.py" in result.stdout

    def test_tests_volume_mount(self, docker_container: None) -> None:
        """Verify tests are accessible (via volume mount)."""
        result = exec_in_container("ls", "-la", "/app/tests")
        assert "unit" in result.stdout
        assert "integration" in result.stdout

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


class TestDockerImageOptimization:
    """Test Docker image size and optimization."""

    def test_no_cache_in_apt_lists(self, docker_container: None) -> None:
        """Verify apt lists are cleaned up to reduce image size."""
        result = exec_in_container("ls", "/var/lib/apt/lists/", check=False)
        # Should be mostly empty (just lock files)
        list_count = len([line for line in result.stdout.split("\n") if line.strip()])
        assert list_count <= 3, "apt/lists not cleaned up properly"

    def test_debian_snapshot_configured(self, docker_container: None) -> None:
        """Verify Debian snapshot sources are configured for reproducibility."""
        result = exec_in_container("cat", "/etc/apt/sources.list.d/debian.sources")
        assert "snapshot.debian.org" in result.stdout
        assert "Check-Valid-Until: no" in result.stdout
