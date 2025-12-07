"""Unit tests for transcription processor."""

import shutil
from pathlib import Path

from backend.database import TranscriptionDatabase
from backend.transcribe import TranscriptionMetrics
from backend.processor import TranscriptionProcessor


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


class RecordingTranscribe:
    def __init__(self, *, should_fail: bool = False, metrics: TranscriptionMetrics | None = None) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []
        self.should_fail = should_fail
        self.metrics = metrics

    def __call__(
        self, file_path: str, json_path: str, preset: str, language: str | None = None
    ) -> TranscriptionMetrics | None:
        self.calls.append((file_path, json_path, preset, language))
        if self.should_fail:
            raise RuntimeError("Test error")
        Path(json_path).write_text("{}")
        return self.metrics or self._default_metrics(file_path, preset, language)

    @staticmethod
    def _default_metrics(file_path: str, preset: str, language: str | None) -> TranscriptionMetrics:
        applied_language = language or "et"
        return TranscriptionMetrics(
            audio_path=file_path,
            preset=preset,
            requested_language=language,
            applied_language=applied_language,
            detected_language=applied_language,
            language_probability=0.5,
            audio_duration=60.0,
            total_processing_time=1.0,
            transcribe_duration=0.5,
            preprocess_duration=0.1,
            preprocess_enabled=False,
            preprocess_profile="test",
            target_sample_rate=16000,
            target_channels=1,
            preprocess_snr_before=None,
            preprocess_snr_after=None,
            preprocess_steps=[],
            speed_ratio=120.0,
        )


class RecordingMover:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.should_fail = should_fail

    def __call__(self, src: str, dst: str):
        self.calls.append((src, dst))
        if self.should_fail:
            raise OSError("Permission denied")
        return shutil.move(src, dst)


def test_process_file_success(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test successfully processing a file."""
    # Create test audio file
    audio_file = temp_folder / "audio1.wav"
    audio_file.touch()

    # Register file in database
    temp_db.add_file(str(audio_file), "pending")

    transcribe = RecordingTranscribe()
    processor = TranscriptionProcessor(temp_db, temp_folder, transcribe_fn=transcribe)
    result = processor.process_file(str(audio_file))

    assert result.status == "completed"
    assert result.metrics is not None
    assert len(transcribe.calls) == 1

    # Verify status updated
    status = temp_db.get_status(str(audio_file))
    assert status is not None
    assert status["status"] == "completed"

    # Verify file moved to processed folder
    assert not audio_file.exists()
    assert (processor.processed_folder / "audio1.wav").exists()


def test_process_file_failure(
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
    transcribe = RecordingTranscribe(should_fail=True)

    processor = TranscriptionProcessor(temp_db, temp_folder, transcribe_fn=transcribe)
    result = processor.process_file(str(audio_file))

    assert result.status == "failed"
    assert result.metrics is None

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

    assert result.status == "failed"
    assert result.error_message is not None

    # Verify status updated to failed
    status = temp_db.get_status(str(audio_file))
    assert status is not None
    assert status["status"] == "failed"
    assert "not found" in status["error_message"].lower()


def test_process_all_files(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test processing all files in provided list."""
    # Create test audio files
    audio1 = temp_folder / "audio1.wav"
    audio2 = temp_folder / "audio2.wav"
    audio1.touch()
    audio2.touch()

    processor = TranscriptionProcessor(temp_db, temp_folder, transcribe_fn=RecordingTranscribe())
    results = processor.process_all_files([str(audio1), str(audio2)])

    assert results["succeeded"] == 2
    assert results["failed"] == 0
    assert len(results["file_stats"]) == 2
    assert all(stat.status == "completed" for stat in results["file_stats"])


def test_process_folder(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test full folder processing workflow."""
    # Create test audio files
    audio1 = temp_folder / "audio1.wav"
    audio2 = temp_folder / "audio2.wav"
    audio1.touch()
    audio2.touch()

    processor = TranscriptionProcessor(temp_db, temp_folder, transcribe_fn=RecordingTranscribe())
    results = processor.process_folder()

    assert results["files_found"] == 2
    assert results["succeeded"] == 2
    assert results["failed"] == 0
    stats = results["run_statistics"]
    assert stats["files_found"] == 2
    assert len(stats["files"]) == 2


def test_process_file_move_failure_keeps_pending_status(
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
    transcribe = RecordingTranscribe()
    failing_move = RecordingMover(should_fail=True)

    processor = TranscriptionProcessor(temp_db, temp_folder, transcribe_fn=transcribe, move_fn=failing_move)
    result = processor.process_file(str(audio_file))

    # Processing should fail
    assert result.status == "failed"
    assert result.error_message

    # Verify status is 'failed', NOT 'completed'
    status = temp_db.get_status(str(audio_file))
    assert status is not None
    assert status["status"] == "failed"
    assert "Permission denied" in status["error_message"]

    # Verify file is still in original location (not moved)
    assert audio_file.exists()
