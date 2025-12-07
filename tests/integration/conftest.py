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
    pass

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
def real_audio_sample() -> Path:
    """Ensure the canonical real audio sample is present for integrations."""
    sample_path = Path("tests/test.mp3")
    if not sample_path.exists():
        pytest.fail("tests/test.mp3 is required for real-audio integrations; add the fixture asset.")
    return sample_path


@pytest.fixture(scope="session")
def ffmpeg_tooling() -> tuple[str, str]:
    """Ensure ffmpeg and ffprobe are available for audio processing tests."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.fail("ffmpeg and ffprobe are required for loudnorm integration; install ffmpeg.")
    return ffmpeg, ffprobe


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
