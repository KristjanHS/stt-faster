"""Integration test configuration and shared fixtures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


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
