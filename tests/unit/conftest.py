from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from backend.database import TranscriptionDatabase


@pytest.fixture(scope="session", autouse=True)
def _disable_network_for_unit_tests() -> None:
    """Block real sockets for unit tests; allow Unix sockets for pytest internals."""
    from pytest_socket import disable_socket

    disable_socket(allow_unix_socket=True)


@pytest.fixture
def temp_db() -> Iterator[TranscriptionDatabase]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db = TranscriptionDatabase(db_path)
    yield db
    db.close()

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_folder() -> Iterator[Path]:
    """Create a temporary folder for testing."""
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
