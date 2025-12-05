"""Fixtures for E2E tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def e2e_test_folder(tmp_path: Path) -> Path:
    """Create a temporary folder with test.mp3 for E2E transcription tests.

    Returns:
        Path to temporary folder containing test.mp3
    """
    # Copy test.mp3 from tests directory if it exists
    source_mp3 = Path(__file__).parent.parent / "test.mp3"

    if source_mp3.exists():
        import shutil

        dest_mp3 = tmp_path / "test.mp3"
        shutil.copy(source_mp3, dest_mp3)

    return tmp_path


@pytest.fixture
def e2e_test_db(tmp_path: Path) -> Path:
    """Create a temporary database path for E2E tests.

    Returns:
        Path to temporary database file
    """
    return tmp_path / "test_transcriptions.db"
