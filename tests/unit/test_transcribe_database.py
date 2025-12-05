"""Unit tests for transcription database operations."""

from pathlib import Path

from backend.database import TranscriptionDatabase


def test_database_initialization(temp_db: TranscriptionDatabase) -> None:
    """Test that database initializes correctly."""
    assert temp_db.conn is not None
    assert Path(temp_db.db_path).exists()


def test_add_file(temp_db: TranscriptionDatabase) -> None:
    """Test adding a file to the database."""
    temp_db.add_file("/path/to/audio.wav", "pending")

    status = temp_db.get_status("/path/to/audio.wav")
    assert status is not None
    assert status["file_path"] == "/path/to/audio.wav"
    assert status["status"] == "pending"
    assert status["error_message"] is None


def test_add_duplicate_file(temp_db: TranscriptionDatabase) -> None:
    """Test that adding the same file twice doesn't raise an error."""
    temp_db.add_file("/path/to/audio.wav", "pending")
    temp_db.add_file("/path/to/audio.wav", "pending")  # Should not raise

    all_files = temp_db.get_all_files()
    assert len(all_files) == 1


def test_update_status(temp_db: TranscriptionDatabase) -> None:
    """Test updating file status."""
    temp_db.add_file("/path/to/audio.wav", "pending")
    temp_db.update_status("/path/to/audio.wav", "completed")

    status = temp_db.get_status("/path/to/audio.wav")
    assert status is not None
    assert status["status"] == "completed"


def test_update_status_with_error(temp_db: TranscriptionDatabase) -> None:
    """Test updating file status with error message."""
    temp_db.add_file("/path/to/audio.wav", "pending")
    error_msg = "Test error message"
    temp_db.update_status("/path/to/audio.wav", "failed", error_msg)

    status = temp_db.get_status("/path/to/audio.wav")
    assert status is not None
    assert status["status"] == "failed"
    assert status["error_message"] == error_msg


def test_get_status_nonexistent(temp_db: TranscriptionDatabase) -> None:
    """Test getting status of a file that doesn't exist."""
    status = temp_db.get_status("/nonexistent/file.wav")
    assert status is None


def test_get_files_by_status(temp_db: TranscriptionDatabase) -> None:
    """Test filtering files by status."""
    temp_db.add_file("/path/to/audio1.wav", "pending")
    temp_db.add_file("/path/to/audio2.wav", "pending")
    temp_db.add_file("/path/to/audio3.wav", "completed")
    temp_db.add_file("/path/to/audio4.wav", "failed")

    pending_files = temp_db.get_files_by_status("pending")
    assert len(pending_files) == 2

    completed_files = temp_db.get_files_by_status("completed")
    assert len(completed_files) == 1

    failed_files = temp_db.get_files_by_status("failed")
    assert len(failed_files) == 1


def test_get_all_files(temp_db: TranscriptionDatabase) -> None:
    """Test getting all files."""
    temp_db.add_file("/path/to/audio1.wav", "pending")
    temp_db.add_file("/path/to/audio2.wav", "completed")
    temp_db.add_file("/path/to/audio3.wav", "failed")

    all_files = temp_db.get_all_files()
    assert len(all_files) == 3


def test_get_summary(temp_db: TranscriptionDatabase) -> None:
    """Test getting summary statistics."""
    temp_db.add_file("/path/to/audio1.wav", "pending")
    temp_db.add_file("/path/to/audio2.wav", "pending")
    temp_db.add_file("/path/to/audio3.wav", "completed")
    temp_db.add_file("/path/to/audio4.wav", "failed")

    summary = temp_db.get_summary()
    assert summary["pending"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1


def test_context_manager(temp_db: TranscriptionDatabase) -> None:
    """Test using database as a context manager."""
    db_path = temp_db.db_path
    temp_db.close()  # Close the fixture database

    with TranscriptionDatabase(db_path) as db:
        db.add_file("/path/to/audio.wav", "pending")
        assert db.conn is not None

    # Connection should be closed after exiting context
    # We can't directly check if connection is closed, but we can verify the file exists
    assert Path(db_path).exists()

    # Cleanup
    Path(db_path).unlink(missing_ok=True)
