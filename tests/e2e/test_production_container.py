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
DIARIZE_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "audio" / "two_speakers_10s.wav"

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

    def test_build_succeeds(self, production_image: str) -> None:
        """Verify the production image builds successfully."""
        # The fixture builds the image; if we get here, it succeeded
        result = run_docker("images", production_image, "-q")
        assert result.stdout.strip(), f"Image {production_image} not found"


@pytest.mark.docker
class TestProductionCloudNative:
    """Test cloud-native characteristics of production container."""

    def test_runs_as_non_root(self, production_image: str) -> None:
        """Verify container runs as non-root user."""
        result = run_docker("inspect", production_image, "--format", "{{.Config.User}}")
        user = result.stdout.strip()
        assert user == "appuser", f"Expected non-root user 'appuser', got '{user}'"

    def test_user_id_at_runtime(self, production_image: str) -> None:
        """Verify container runs as non-root user (security behavior, not specific ID)."""
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "sh",
            "-c",
            "id -u && id -g && whoami",
        )
        lines = result.stdout.strip().split("\n")
        uid, gid = int(lines[0]), int(lines[1])
        username = lines[2]

        # Test behavior: user must be non-root (security requirement)
        assert uid != 0, "Container must not run as root (UID 0)"
        assert gid != 0, "Container must not run as root group (GID 0)"
        assert username != "root", "Container must not run as root user"

    def test_healthcheck_works(self, production_image: str) -> None:
        """Verify configured healthcheck command executes successfully."""
        inspect_result = run_docker("inspect", production_image, "--format", "{{json .Config.Healthcheck}}")
        import json

        healthcheck = json.loads(inspect_result.stdout)
        test_cmd = healthcheck.get("Test")
        assert test_cmd, "Healthcheck test command missing"

        if test_cmd[0] == "NONE":
            pytest.fail("Healthcheck test command is disabled")

        if test_cmd[0] == "CMD":
            command = test_cmd[1:]
            result = run_docker("run", "--rm", "--entrypoint", "", production_image, *command)
        elif test_cmd[0] == "CMD-SHELL":
            command = " ".join(test_cmd[1:])
            result = run_docker(
                "run",
                "--rm",
                "--entrypoint",
                "",
                production_image,
                "sh",
                "-c",
                command,
            )
        else:
            pytest.fail(f"Unknown healthcheck test type: {test_cmd[0]}")

        assert result.returncode == 0

    def test_no_sensitive_data_in_env(self, production_image: str) -> None:
        """Verify no sensitive data leaks into environment or logs (security behavior)."""
        # Test behavior: sensitive data should not appear in environment
        result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "env",
        )
        env_output = result.stdout.lower()
        sensitive_patterns = ["password", "secret", "api_key", "token", "credential"]
        for pattern in sensitive_patterns:
            assert pattern not in env_output, f"Possible sensitive data in env: {pattern}"

        # Test behavior: sensitive data should not appear in logs when running commands
        # This verifies the actual security behavior, not just env var presence
        log_result = run_docker(
            "run",
            "--rm",
            "--entrypoint",
            "",
            production_image,
            "python",
            "-c",
            (
                "import os; import sys; "
                "print('ENV_CHECK:', '|'.join(k for k in os.environ.keys() "
                "if any(p in k.upper() for p in ['PASS', 'SECRET', 'TOKEN', 'KEY', 'CRED']))); "
                "sys.exit(0 if not any(p in str(os.environ).upper() "
                "for p in ['PASSWORD', 'SECRET', 'API_KEY', 'TOKEN']) else 1)"
            ),
        )
        assert log_result.returncode == 0, "Sensitive data detected in environment during execution"


@pytest.mark.docker
class TestProductionRuntime:
    """Test production container runtime behavior."""

    def test_entrypoint_help_works(self, production_image: str) -> None:
        """Verify the container entrypoint runs successfully without external dependencies."""
        result = run_docker("run", "--rm", production_image, "--help")
        assert result.returncode == 0


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


@pytest.mark.docker
@pytest.mark.network
class TestProductionDiarization:
    """Container × diarization combined — mirrors `transcribe_*_Desk.bat` flags.

    On-host diarization is covered by `tests/integration/test_diarize_with_pyannote.py`.
    This class exercises the container path setup.bat builds and the Windows
    transcribe bats invoke: `docker run stt-faster:latest process /workspace
    --diarize --num-speakers 2 ...` — transcription is implicit (no separate
    no-diarize test, since `--diarize` runs the same whisper path plus pyannote).
    """

    @pytest.mark.slow
    def test_diarize_two_speakers_in_container(
        self,
        production_image: str,
        hf_token: str | None,
    ) -> None:
        if not hf_token:
            pytest.skip(
                "HF_TOKEN/HUGGING_FACE_HUB_TOKEN required for pyannote diarization — see docs/diarization_setup.md"
            )
        if not DIARIZE_FIXTURE.exists():
            pytest.skip(f"Fixture {DIARIZE_FIXTURE} absent — see tests/fixtures/audio/README.md")

        import os
        import re
        import shutil

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            workspace = tmppath / "workspace"
            workspace.mkdir()
            shutil.copy(DIARIZE_FIXTURE, workspace / DIARIZE_FIXTURE.name)

            local_hf_cache = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/hf"))
            data_dir = tmppath / ".local" / "share" / "stt-faster"
            data_dir.mkdir(parents=True)

            # Mirror the English Windows bats (transcribe_english_*.bat):
            # `process /workspace --preset turbo --language en
            #  --output-format txt --diarize --num-speakers 2`.
            result = run_docker_with_env(
                image=production_image,
                command_args=[
                    "process",
                    "/workspace",
                    "--preset",
                    "turbo",
                    "--language",
                    "en",
                    "--output-format",
                    "txt",
                    "--diarize",
                    "--num-speakers",
                    "2",
                ],
                volumes=[
                    (str(workspace), "/workspace"),
                    (local_hf_cache, "/home/appuser/.cache/hf"),
                    (str(data_dir), "/home/appuser/.local/share/stt-faster"),
                ],
                env_vars={"HF_TOKEN": hf_token},
                use_gpu=False,
                timeout=900,
                check=False,
            )

            logger.info("Diarize stdout:\n%s", result.stdout)
            if result.stderr:
                logger.info("Diarize stderr:\n%s", result.stderr)
            assert result.returncode == 0, f"Container diarize exited {result.returncode}"

            processed = workspace / "processed"
            assert processed.exists(), "Processed directory not created"
            txt_files = list(processed.rglob("*.txt"))
            assert txt_files, "No .txt transcription files generated"

            content = txt_files[0].read_text(encoding="utf-8")
            speaker_labels = set(re.findall(r"SPEAKER_\d{2}", content))
            assert len(speaker_labels) >= 2, (
                f"Expected 2+ distinct SPEAKER_NN labels in {txt_files[0]}, got {speaker_labels!r}"
            )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "--tb=short"])
