"""Integration test configuration and shared fixtures."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def _migrate_hf_transfer_env() -> None:
    """Prefer new Hugging Face high-performance env and drop deprecated one.

    Hugging Face deprecated HF_HUB_ENABLE_HF_TRANSFER in favor of
    HF_XET_HIGH_PERFORMANCE. Removing the legacy variable prevents noisy
    deprecation warnings during test runs while keeping the intended behavior.
    """
    legacy = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if legacy and "HF_XET_HIGH_PERFORMANCE" not in os.environ:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = legacy


@pytest.fixture
def cli_test_folder(tmp_path: Path) -> Path:
    """Create a temporary folder with test audio files.

    Creates empty placeholder audio files for CLI testing without
    requiring actual model processing.
    """
    test_folder = tmp_path / "test_audio"
    test_folder.mkdir()

    # Create test audio files (empty for now, as real models would be too heavy)
    (test_folder / "audio1.wav").touch()
    (test_folder / "audio2.mp3").touch()
    (test_folder / "audio3.m4a").touch()

    return test_folder


@pytest.fixture
def real_audio_test_folder(tmp_path: Path) -> Path:
    """Create a temporary folder with the actual test.mp3 file.

    Copies the real test audio file if it exists.
    """
    test_folder = tmp_path / "test_audio_real"
    test_folder.mkdir()

    # Copy the actual test.mp3 file to the test folder
    source_mp3 = Path("tests/test.mp3")
    if source_mp3.exists():
        dest_mp3 = test_folder / "test.mp3"
        shutil.copy2(source_mp3, dest_mp3)

    return test_folder


@pytest.fixture
def cli_test_db(tmp_path: Path) -> Path:
    """Create a temporary database path for CLI testing."""
    return tmp_path / "test_cli.db"


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if GPU is available for integration tests.

    Returns:
        True if GPU/CUDA is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def ctranslate2_device_count() -> int:
    """Get number of CUDA devices available to ctranslate2.

    Returns:
        Number of CUDA devices (0 if none available or CUDA not supported).
    """
    try:
        import ctranslate2

        try:
            return ctranslate2.get_cuda_device_count()
        except (AttributeError, RuntimeError):
            return 0
    except ImportError:
        return 0


@pytest.fixture(scope="session")
def tiny_whisper_model() -> "WhisperModel | None":
    """Cached tiny Whisper model for integration tests.

    Downloads and caches the tiny Whisper model (39M parameters) for use in
    integration tests that need real model inference without full E2E overhead.

    Behavior controlled by USE_CACHED_MODEL environment variable:
    - "true" (default in integration tests): Download/use cached tiny model
    - "false": Return None (tests should fall back to mocks)

    Cache location: ~/.cache/hf/hub/ (standard HuggingFace cache)

    Returns:
        WhisperModel instance if successful, None if disabled or download fails.

    Example usage in tests:
        ```python
        def test_with_optional_model(tiny_whisper_model):
            if tiny_whisper_model is None:
                # Fall back to mock
                with patch("backend.transcribe.pick_model") as mock:
                    mock.return_value = MagicMock()
                    # ... test with mock
            else:
                # Use real cached model
                result = transcribe(audio_path, model=tiny_whisper_model)
                # ... test with real model
        ```
    """
    # Check environment variable (default to true for integration tests)
    use_cached = os.getenv("USE_CACHED_MODEL", "true").lower() in ("true", "1", "yes")

    if not use_cached:
        LOGGER.info("USE_CACHED_MODEL=false, skipping tiny model cache")
        return None

    try:
        from faster_whisper import WhisperModel

        LOGGER.info("Loading tiny Whisper model for integration tests (one-time download if not cached)...")

        # Use tiny model: smallest/fastest for testing
        # Device: CPU (most compatible), compute_type: int8 (fastest on CPU)
        model = WhisperModel("tiny", device="cpu", compute_type="int8")

        LOGGER.info("Tiny Whisper model loaded successfully (cached for future test runs)")
        return model

    except Exception as e:
        LOGGER.warning(
            "Failed to load tiny Whisper model for integration tests: %s. "
            "Tests will fall back to mocks. This is expected if running offline or in restricted environments.",
            e,
            exc_info=True,
        )
        return None
