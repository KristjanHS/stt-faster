"""Unit tests for transcription processor."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.transcription.database import TranscriptionDatabase
from scripts.transcription.processor import TranscriptionProcessor


@pytest.fixture
def temp_db() -> TranscriptionDatabase:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    db = TranscriptionDatabase(db_path)
    yield db
    db.close()

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def temp_folder() -> Path:
    """Create a temporary folder for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir

    # Cleanup
    import shutil

    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def test_processor_initialization(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test that processor initializes correctly."""
    processor = TranscriptionProcessor(temp_db, temp_folder)

    assert processor.db == temp_db
    assert processor.input_folder == temp_folder
    assert processor.preset == "et-large"
    assert processor.processed_folder.exists()
    assert processor.failed_folder.exists()


def test_processor_custom_preset(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test processor with custom preset."""
    processor = TranscriptionProcessor(temp_db, temp_folder, preset="turbo")
    assert processor.preset == "turbo"


def test_scan_folder_empty(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test scanning an empty folder."""
    processor = TranscriptionProcessor(temp_db, temp_folder)
    files = processor.scan_folder()
    assert len(files) == 0


def test_scan_folder_with_audio_files(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test scanning a folder with audio files."""
    # Create some test audio files
    (temp_folder / "audio1.wav").touch()
    (temp_folder / "audio2.mp3").touch()
    (temp_folder / "audio3.m4a").touch()
    (temp_folder / "not_audio.txt").touch()

    processor = TranscriptionProcessor(temp_db, temp_folder)
    files = processor.scan_folder()

    assert len(files) == 3
    assert any("audio1.wav" in f for f in files)
    assert any("audio2.mp3" in f for f in files)
    assert any("audio3.m4a" in f for f in files)
    assert not any("not_audio.txt" in f for f in files)


def test_scan_folder_ignores_subdirectories(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test that scanning ignores files in subdirectories."""
    # Create files in root
    (temp_folder / "audio1.wav").touch()

    # Create files in subdirectories
    subfolder = temp_folder / "subfolder"
    subfolder.mkdir()
    (subfolder / "audio2.wav").touch()

    processor = TranscriptionProcessor(temp_db, temp_folder)
    files = processor.scan_folder()

    assert len(files) == 1
    assert "audio1.wav" in files[0]


def test_get_files_to_process(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test getting list of files to process from folder."""
    # Create test audio files
    (temp_folder / "audio1.wav").touch()
    (temp_folder / "audio2.wav").touch()

    processor = TranscriptionProcessor(temp_db, temp_folder)
    files = processor.get_files_to_process()

    assert len(files) == 2
    assert any("audio1.wav" in f for f in files)
    assert any("audio2.wav" in f for f in files)


def test_get_files_returns_all_regardless_of_db(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test that get_files_to_process returns all files regardless of database state."""
    audio_file = temp_folder / "audio1.wav"
    audio_file.touch()

    # Mark file as completed in database
    temp_db.add_file(str(audio_file), "completed")

    processor = TranscriptionProcessor(temp_db, temp_folder)
    files = processor.get_files_to_process()

    # File should still be returned - file location is source of truth
    assert len(files) == 1
    assert str(audio_file) in files


@patch("scripts.transcription.processor.transcribe_to_json")
def test_process_file_success(
    mock_transcribe: MagicMock,
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test successfully processing a file."""
    # Create test audio file
    audio_file = temp_folder / "audio1.wav"
    audio_file.touch()

    # Register file in database
    temp_db.add_file(str(audio_file), "pending")

    processor = TranscriptionProcessor(temp_db, temp_folder)
    result = processor.process_file(str(audio_file))

    assert result is True
    mock_transcribe.assert_called_once()

    # Verify status updated
    status = temp_db.get_status(str(audio_file))
    assert status is not None
    assert status["status"] == "completed"

    # Verify file moved to processed folder
    assert not audio_file.exists()
    assert (processor.processed_folder / "audio1.wav").exists()


@patch("scripts.transcription.processor.transcribe_to_json")
def test_process_file_failure(
    mock_transcribe: MagicMock,
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test handling of transcription failure."""
    # Create test audio file
    audio_file = temp_folder / "audio1.wav"
    audio_file.touch()

    # Register file in database
    temp_db.add_file(str(audio_file), "pending")

    # Make transcription fail
    mock_transcribe.side_effect = RuntimeError("Test error")

    processor = TranscriptionProcessor(temp_db, temp_folder)
    result = processor.process_file(str(audio_file))

    assert result is False

    # Verify status updated to failed
    status = temp_db.get_status(str(audio_file))
    assert status is not None
    assert status["status"] == "failed"
    assert "RuntimeError" in status["error_message"]

    # Verify file moved to failed folder
    assert not audio_file.exists()
    assert (processor.failed_folder / "audio1.wav").exists()


def test_process_file_not_found(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test processing a non-existent file."""
    audio_file = temp_folder / "nonexistent.wav"

    # Register file in database (but don't create it)
    temp_db.add_file(str(audio_file), "pending")

    processor = TranscriptionProcessor(temp_db, temp_folder)
    result = processor.process_file(str(audio_file))

    assert result is False

    # Verify status updated to failed
    status = temp_db.get_status(str(audio_file))
    assert status is not None
    assert status["status"] == "failed"
    assert "not found" in status["error_message"].lower()


@patch("scripts.transcription.processor.transcribe_to_json")
def test_process_all_files(
    mock_transcribe: MagicMock,
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test processing all files in provided list."""
    # Create test audio files
    audio1 = temp_folder / "audio1.wav"
    audio2 = temp_folder / "audio2.wav"
    audio1.touch()
    audio2.touch()

    processor = TranscriptionProcessor(temp_db, temp_folder)
    results = processor.process_all_files([str(audio1), str(audio2)])

    assert results["succeeded"] == 2
    assert results["failed"] == 0


@patch("scripts.transcription.processor.transcribe_to_json")
def test_process_folder(
    mock_transcribe: MagicMock,
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test full folder processing workflow."""
    # Create test audio files
    audio1 = temp_folder / "audio1.wav"
    audio2 = temp_folder / "audio2.wav"
    audio1.touch()
    audio2.touch()

    processor = TranscriptionProcessor(temp_db, temp_folder)
    results = processor.process_folder()

    assert results["files_found"] == 2
    assert results["succeeded"] == 2
    assert results["failed"] == 0


@patch("scripts.transcription.processor.transcribe_to_json")
@patch("scripts.transcription.processor.shutil.move")
def test_process_file_move_failure_keeps_pending_status(
    mock_move: MagicMock,
    mock_transcribe: MagicMock,
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test that if file move fails, database status is NOT updated to completed.

    This is a regression test for the bug where database status was updated
    before moving files, causing orphaned files that won't be reprocessed.
    """
    # Create test audio file
    audio_file = temp_folder / "audio1.wav"
    audio_file.touch()

    # Register file
    temp_db.add_file(str(audio_file), "pending")

    # Mock transcribe to succeed but move to fail
    mock_transcribe.return_value = None
    mock_move.side_effect = OSError("Permission denied")

    processor = TranscriptionProcessor(temp_db, temp_folder)
    result = processor.process_file(str(audio_file))

    # Processing should fail
    assert result is False

    # Verify status is 'failed', NOT 'completed'
    status = temp_db.get_status(str(audio_file))
    assert status is not None
    assert status["status"] == "failed"
    assert "Permission denied" in status["error_message"]

    # Verify file is still in original location (not moved)
    assert audio_file.exists()
