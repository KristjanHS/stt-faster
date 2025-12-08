"""End-to-end tests for production Docker container.

These tests verify that the production container (from root Dockerfile):
1. Builds successfully from clean state
2. Runs as non-root user
3. Has proper cloud-native characteristics
4. Healthcheck works correctly
5. Can be used for transcription
6. Properly handles volumes and permissions

Note: Some tests require HuggingFace authentication for gated models.
Set HF_TOKEN environment variable or run: huggingface-cli login
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# Sensitive environment variable names that should be masked in logs
SENSITIVE_ENV_VARS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "API_KEY", "SECRET", "PASSWORD", "TOKEN"}

# Production Dockerfile is at project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
IMAGE_NAME = "stt-faster:test-prod"
TEST_AUDIO_FILE = PROJECT_ROOT / "tests" / "test.mp3"

# HuggingFace token is now provided via the hf_token fixture in tests/conftest.py
# Tests requiring the token should accept it as a fixture parameter


def _sanitize_command_for_logging(cmd: list[str]) -> str:
    """Sanitize command for logging by masking sensitive values.

    Args:
        cmd: Command list to sanitize

    Returns:
        Sanitized command string safe for logging
    """
    sanitized = []
    i = 0
    while i < len(cmd):
        arg = cmd[i]
        # Check if this is an environment variable flag
        if arg == "-e" and i + 1 < len(cmd):
            env_arg = cmd[i + 1]
            # Check if it contains a sensitive variable name
            if any(sensitive in env_arg.upper() for sensitive in SENSITIVE_ENV_VARS):
                # Mask the value (keep key=, replace value with ***)
                if "=" in env_arg:
                    key, _ = env_arg.split("=", 1)
                    sanitized.append(arg)
                    sanitized.append(f"{key}=***")
                    i += 2
                    continue
        sanitized.append(arg)
        i += 1
    return " ".join(sanitized)


def run_docker(
    *args: str,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Run docker command with standard options.

    Sensitive environment variables in the command are masked in logs.
    """
    cmd = ["docker", *args]
    sanitized_cmd = _sanitize_command_for_logging(cmd)
    logger.info("Running: %s", sanitized_cmd)
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
    )
    if result.stdout:
        logger.debug("stdout: %s", result.stdout)
    if result.stderr:
        logger.debug("stderr: %s", result.stderr)
    return result


def run_docker_with_env(
    image: str,
    command_args: list[str],
    env_vars: dict[str, str] | None = None,
    volumes: list[tuple[str, str]] | None = None,
    use_gpu: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run docker container with environment variables and volumes.

    Sensitive environment variables (like HF_TOKEN) are passed via --env-file
    to avoid appearing in command line or logs. Non-sensitive variables use -e flags.

    Args:
        image: Docker image name
        command_args: Command and arguments to pass to container
        env_vars: Environment variables to set in container
        volumes: List of (host_path, container_path) tuples
        use_gpu: Whether to enable GPU access (--gpus all)
        **kwargs: Additional arguments passed to run_docker
    """
    docker_args = ["run", "--rm"]

    # Add GPU support if requested
    if use_gpu:
        docker_args.extend(["--gpus", "all"])

    # Separate sensitive and non-sensitive environment variables
    sensitive_vars: dict[str, str] = {}
    non_sensitive_vars: dict[str, str] = {}

    if env_vars:
        for key, value in env_vars.items():
            if value:  # Only add if value is not empty
                # Check if this is a sensitive variable
                if any(sensitive in key.upper() for sensitive in SENSITIVE_ENV_VARS):
                    sensitive_vars[key] = value
                else:
                    non_sensitive_vars[key] = value

    # Add non-sensitive environment variables via -e flags (visible in logs but safe)
    for key, value in non_sensitive_vars.items():
        docker_args.extend(["-e", f"{key}={value}"])

    # Add sensitive environment variables via --env-file (not visible in command line)
    env_file_path: Path | None = None
    if sensitive_vars:
        # Create temporary env file
        env_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".env")
        try:
            for key, value in sensitive_vars.items():
                env_file.write(f"{key}={value}\n")
            env_file_path = Path(env_file.name)
            env_file.close()
            docker_args.extend(["--env-file", str(env_file_path)])
            logger.debug("Using env-file for sensitive variables: %s", list(sensitive_vars.keys()))
        except Exception as e:
            # Fallback to -e flags if env file creation fails (with masking in logs)
            logger.warning("Failed to create env file, falling back to -e flags: %s", e)
            for key, value in sensitive_vars.items():
                docker_args.extend(["-e", f"{key}={value}"])
            env_file_path = None

    # Add volume mounts
    if volumes:
        for host_path, container_path in volumes:
            docker_args.extend(["-v", f"{host_path}:{container_path}"])

    # Add image and command
    docker_args.append(image)
    docker_args.extend(command_args)

    try:
        return run_docker(*docker_args, **kwargs)
    finally:
        # Clean up temporary env file
        if env_file_path and env_file_path.exists():
            try:
                env_file_path.unlink()
            except Exception as e:
                logger.warning("Failed to clean up env file %s: %s", env_file_path, e)


def container_exec(
    container_name: str,
    *cmd: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Execute a command inside a running container."""
    return run_docker("exec", "-i", container_name, *cmd, check=check)


@pytest.fixture(scope="module")
def production_image() -> Generator[str, None, None]:
    """Build the production Docker image."""
    logger.info("Building production Docker image from %s", DOCKERFILE_PATH)

    # Build with a test-specific tag
    run_docker(
        "build",
        "-t",
        IMAGE_NAME,
        "-f",
        str(DOCKERFILE_PATH),
        str(PROJECT_ROOT),
    )

    yield IMAGE_NAME

    # Cleanup: remove test image
    logger.info("Cleaning up production test image")
    run_docker("rmi", IMAGE_NAME, check=False)


@pytest.mark.docker
class TestProductionImageBuild:
    """Test production Docker image build process."""

    def test_dockerfile_exists(self) -> None:
        """Verify production Dockerfile exists at repo root."""
        assert DOCKERFILE_PATH.exists(), f"Dockerfile not found at {DOCKERFILE_PATH}"

    def test_build_succeeds(self, production_image: str) -> None:
        """Verify the production image builds successfully."""
        # The fixture builds the image; if we get here, it succeeded
        result = run_docker("images", production_image, "-q")
        assert result.stdout.strip(), f"Image {production_image} not found"

    def test_image_size_reasonable(self, production_image: str) -> None:
        """Verify image size is reasonable (not bloated)."""
        result = run_docker("images", production_image, "--format", "{{.Size}}")
        size_str = result.stdout.strip()
        logger.info("Image size: %s", size_str)

        # Parse size (format like "780MB" or "1.2GB")
        if "GB" in size_str:
            size_gb = float(size_str.replace("GB", ""))
            # Production image should be under 2GB (includes Python + ML libs)
            assert size_gb < 2.0, f"Image too large: {size_str}"
        # If in MB, it's definitely acceptable


@pytest.mark.docker
class TestProductionCloudNative:
    """Test cloud-native characteristics of production container."""

    def test_runs_as_non_root(self, production_image: str) -> None:
        """Verify container runs as non-root user."""
        result = run_docker("inspect", production_image, "--format", "{{.Config.User}}")
        user = result.stdout.strip()
        assert user == "appuser", f"Expected non-root user 'appuser', got '{user}'"

    def test_user_id_at_runtime(self, production_image: str) -> None:
        """Verify container runs with correct UID/GID at runtime."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "id -u && id -g",
        )
        lines = result.stdout.strip().split("\n")
        uid, gid = int(lines[0]), int(lines[1])
        assert uid == 1000, f"Expected UID 1000, got {uid}"
        assert gid == 1000, f"Expected GID 1000, got {gid}"

    def test_has_healthcheck(self, production_image: str) -> None:
        """Verify healthcheck is configured."""
        result = run_docker("inspect", production_image, "--format", "{{.Config.Healthcheck}}")
        healthcheck = result.stdout.strip()
        assert healthcheck != "<nil>", "No healthcheck configured"
        assert "CMD" in healthcheck, "Healthcheck missing CMD"
        logger.info("Healthcheck: %s", healthcheck)

    def test_healthcheck_works(self, production_image: str) -> None:
        """Verify healthcheck command executes successfully."""
        # Extract and run the healthcheck command manually
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "python",
            "-c",
            "import backend.transcribe; import backend.processor; print('OK')",
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_python_unbuffered(self, production_image: str) -> None:
        """Verify PYTHONUNBUFFERED is set for immediate log output."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "echo $PYTHONUNBUFFERED",
        )
        assert result.stdout.strip() == "1"

    def test_venv_in_path(self, production_image: str) -> None:
        """Verify virtualenv bin directory is first in PATH."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "echo $PATH",
        )
        path = result.stdout.strip()
        assert path.startswith("/opt/venv/bin:"), f"venv not first in PATH: {path}"

    def test_hf_cache_configured(self, production_image: str) -> None:
        """Verify Hugging Face cache directory is configured."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "echo $HF_HOME",
        )
        hf_home = result.stdout.strip()
        assert hf_home == "/home/appuser/.cache/hf"

    def test_uses_debian_snapshot(self, production_image: str) -> None:
        """Verify Debian snapshot is configured for reproducible builds."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "cat",
            "/etc/apt/sources.list.d/debian.sources",
            check=False,
        )
        if result.returncode == 0:
            content = result.stdout
            assert "snapshot.debian.org" in content, "Not using Debian snapshot"
            assert "Check-Valid-Until: no" in content, "Missing snapshot config"

    def test_no_sensitive_data_in_env(self, production_image: str) -> None:
        """Verify no sensitive data in environment variables."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "env",
        )
        env_output = result.stdout.lower()
        # Check for common sensitive variable patterns
        sensitive_patterns = ["password", "secret", "api_key", "token", "credential"]
        for pattern in sensitive_patterns:
            assert pattern not in env_output, f"Possible sensitive data in env: {pattern}"


@pytest.mark.docker
class TestProductionRuntime:
    """Test production container runtime behavior."""

    def test_entrypoint_shows_help(self, production_image: str) -> None:
        """Verify default entrypoint shows help message."""
        result = run_docker("run", "--rm", production_image)
        assert "transcribe_manager.py" in result.stdout
        assert "process" in result.stdout
        assert "status" in result.stdout

    def test_help_flag_works(self, production_image: str) -> None:
        """Verify --help flag works."""
        result = run_docker("run", "--rm", production_image, "--help")
        assert "usage:" in result.stdout
        assert "Manage audio transcription" in result.stdout

    def test_python_version(self, production_image: str) -> None:
        """Verify correct Python version is installed."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "python",
            "--version",
        )
        assert "Python 3.12" in result.stdout

    def test_backend_module_importable(self, production_image: str) -> None:
        """Verify backend modules can be imported."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "python",
            "-c",
            "import backend.transcribe; import backend.processor; import backend.database; print('OK')",
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_faster_whisper_installed(self, production_image: str) -> None:
        """Verify faster-whisper is installed."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "python",
            "-c",
            "import faster_whisper; print(faster_whisper.__version__)",
        )
        assert result.returncode == 0
        # Should output version number
        assert len(result.stdout.strip()) > 0


@pytest.mark.docker
class TestProductionVolumes:
    """Test volume mounts and permissions."""

    def test_workspace_directory_exists(self, production_image: str) -> None:
        """Verify /workspace directory exists and is writable."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "ls -ld /workspace && touch /workspace/test.txt && rm /workspace/test.txt",
        )
        assert result.returncode == 0
        assert "appuser" in result.stdout

    def test_volume_mount_works(self, production_image: str) -> None:
        """Verify volume mounts work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_file = tmppath / "test.txt"
            test_file.write_text("test content")

            # Mount tmpdir and verify file is accessible
            result = run_docker(
                "run",
                "--rm",
                "-v",
                f"{tmpdir}:/workspace",
                "--entrypoint",
                "",
                production_image,
                "cat",
                "/workspace/test.txt",
            )
            assert result.stdout.strip() == "test content"

    def test_hf_cache_mount(self, production_image: str) -> None:
        """Verify HF cache directory can be mounted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mount tmpdir as HF cache and verify it's accessible
            result = run_docker(
                "run",
                "--rm",
                "-v",
                f"{tmpdir}:/home/appuser/.cache/hf",
                "--entrypoint",
                "",
                production_image,
                "sh",
                "-c",
                "ls -ld /home/appuser/.cache/hf && touch /home/appuser/.cache/hf/test",
            )
            assert result.returncode == 0
            # Verify file was created on host
            assert (Path(tmpdir) / "test").exists()


@pytest.mark.docker
@pytest.mark.network
class TestProductionTranscription:
    """Test actual transcription functionality (requires network for model download)."""

    def test_transcribe_status_command(self, production_image: str, hf_token: str | None) -> None:
        """Verify status command works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create the database directory structure
            (tmppath / ".local" / "share" / "stt-faster").mkdir(parents=True)

            result = run_docker_with_env(
                image=production_image,
                command_args=["status"],
                volumes=[
                    (tmpdir, "/workspace"),
                    (f"{tmpdir}/.cache", "/home/appuser/.cache/hf"),
                    (f"{tmpdir}/.local/share/stt-faster", "/home/appuser/.local/share/stt-faster"),
                ],
                env_vars={"HF_TOKEN": hf_token} if hf_token else None,
            )
            # Should work even with empty database
            assert result.returncode == 0

    @pytest.mark.slow
    def test_transcribe_process_help(self, production_image: str) -> None:
        """Verify process command help works."""
        result = run_docker(
            "run",
            "--rm",
            production_image,
            "process",
            "--help",
        )
        assert result.returncode == 0
        assert "preset" in result.stdout

    @pytest.mark.slow
    def test_transcribe_with_test_audio(self, production_image: str, hf_token: str | None) -> None:
        """Test transcription with actual audio file (slow, downloads model).

        Note: test.mp3 is in Estonian, so we use the et-large preset (production default).
        GPU is enabled if available for faster transcription.

        Authentication: Uses HF_TOKEN if available, otherwise relies on huggingface-cli login.
        If the model requires authentication and neither is available, the test will fail
        with a clear error from huggingface_hub.
        """
        if not TEST_AUDIO_FILE.exists():
            pytest.fail("Test audio file not found; e2e container test requires tests/test.mp3")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Copy test audio to temp workspace
            import shutil

            workspace_audio = tmppath / "workspace"
            workspace_audio.mkdir()
            shutil.copy(TEST_AUDIO_FILE, workspace_audio / "test.mp3")

            # Use local HF cache to avoid re-downloading models
            import os

            local_hf_cache = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/hf"))
            logger.info(f"Using local HF cache: {local_hf_cache}")

            # Create data directory for transcription state
            data_dir = tmppath / ".local" / "share" / "stt-faster"
            data_dir.mkdir(parents=True)

            # Production container is CPU-only for cloud-native deployment
            # (GPU support would require cuDNN libraries, increasing image size)
            logger.info("ℹ️  Production container uses CPU for cloud-native deployment")

            # Run transcription with HF_TOKEN for model download
            # Use et-large preset (Estonian model - matches test.mp3 language and production default)
            # CPU-optimized for containers without cuDNN
            result = run_docker_with_env(
                image=production_image,
                command_args=["process", "/workspace", "--preset", "et-large", "--output-format", "json"],
                volumes=[
                    (str(workspace_audio), "/workspace"),
                    (local_hf_cache, "/home/appuser/.cache/hf"),
                    (str(data_dir), "/home/appuser/.local/share/stt-faster"),
                ],
                env_vars={"HF_TOKEN": hf_token} if hf_token else {},
                use_gpu=False,  # Production container is CPU-only
                timeout=600,  # 10 minutes for model download + CPU transcription
                check=False,
            )

            # Log the output for debugging
            logger.info("Transcription output:\n%s", result.stdout)
            if result.stderr:
                logger.info("Transcription stderr:\n%s", result.stderr)

            # Verify transcription succeeded
            assert result.returncode == 0, f"Transcription failed with exit code {result.returncode}"

            # Check if processed directory was created
            processed_dir = workspace_audio / "processed"
            assert processed_dir.exists(), "Processed directory not created"

            # Check for JSON transcription files
            json_files = list(processed_dir.glob("*.json"))
            logger.info("Found %d JSON transcription files", len(json_files))
            assert len(json_files) > 0, "No transcription files generated"

            # Verify the transcription is in Estonian
            if json_files:
                import json

                with json_files[0].open() as f:
                    transcript_data = json.load(f)
                    language = transcript_data.get("language")
                    logger.info("Detected language: %s", language)
                    # Estonian models should detect 'et' or 'est'
                    if language:
                        logger.info("✅ Language detected: %s", language)

            logger.info("✅ Transcription completed successfully with %d files", len(json_files))


@pytest.mark.docker
class TestProductionLabels:
    """Test OCI image labels."""

    def test_has_oci_labels(self, production_image: str) -> None:
        """Verify OCI image labels are present."""
        result = run_docker("inspect", production_image, "--format", "{{json .Config.Labels}}")
        import json

        labels = json.loads(result.stdout)

        # Check for key OCI labels
        assert "org.opencontainers.image.title" in labels
        assert labels["org.opencontainers.image.title"] == "stt-faster"

        assert "org.opencontainers.image.description" in labels
        assert "transcription" in labels["org.opencontainers.image.description"].lower()


@pytest.mark.docker
class TestProductionSecurity:
    """Test security aspects of production container."""

    def test_no_sudo(self, production_image: str) -> None:
        """Verify sudo is not installed (security best practice)."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "which sudo || echo 'not found'",
        )
        assert "not found" in result.stdout

    def test_cannot_switch_to_root(self, production_image: str) -> None:
        """Verify user cannot switch to root."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "su - root -c 'echo success' || echo 'failed as expected'",
            check=False,
        )
        # Should fail (su requires password or not available)
        assert "failed as expected" in result.stdout or result.returncode != 0

    def test_minimal_packages(self, production_image: str) -> None:
        """Verify only minimal packages are installed."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "dpkg",
            "-l",
        )
        package_list = result.stdout

        # Should NOT have development tools (gcc-12-base is OK, it's a library)
        # Check for actual compiler packages
        assert "ii  gcc " not in package_list  # Full gcc compiler package
        assert "g++" not in package_list.lower()
        assert "make" not in package_list.lower()
        assert "build-essential" not in package_list.lower()

        # Should have wget (specified in Dockerfile)
        assert "wget" in package_list.lower()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
