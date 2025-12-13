"""Unit tests for transcription processor."""

import logging
import shutil
from pathlib import Path

import pytest

from backend.database import TranscriptionDatabase
from backend.processor import TranscriptionProcessor
from backend.services.factory import ServiceFactory
from backend.services.interfaces import TranscriptionResult
from backend.transcribe import TranscriptionMetrics


def _create_test_processor(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
    preset: str = "et-large",
    language: str | None = None,
    output_format: str = "txt",
    variant=None,
    disable_file_moving: bool = False,
) -> TranscriptionProcessor:
    """Helper function to create a processor with services for testing."""
    transcription_service = ServiceFactory.create_transcription_service(
        variant=variant,
        preset=preset,
        language=language,
        output_format=output_format,
    )
    # Use the provided temp_db instead of creating a new one to avoid conflicts
    state_store = ServiceFactory.create_state_store(db_path=temp_db.db_path)
    file_mover = ServiceFactory.create_file_mover()
    output_writer = ServiceFactory.create_output_writer()

    return TranscriptionProcessor(
        transcription_service=transcription_service,
        state_store=state_store,
        file_mover=file_mover,
        output_writer=output_writer,
        input_folder=temp_folder,
        preset=preset,
        language=language,
        output_format=output_format,
        variant=variant,
        disable_file_moving=disable_file_moving,
    )


def test_processor_custom_preset(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test processor with custom preset."""
    processor = _create_test_processor(temp_db, temp_folder, preset="turbo")
    assert processor.preset == "turbo"


def test_scan_folder_empty(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test scanning an empty folder."""
    processor = _create_test_processor(temp_db, temp_folder)
    files = processor.scan_folder()
    assert len(files) == 0


def test_scan_folder_with_audio_files(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test scanning a folder with audio files."""
    # Create some test audio files
    (temp_folder / "audio1.wav").touch()
    (temp_folder / "audio2.mp3").touch()
    (temp_folder / "audio3.m4a").touch()
    (temp_folder / "not_audio.txt").touch()

    processor = _create_test_processor(temp_db, temp_folder)
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

    processor = _create_test_processor(temp_db, temp_folder)
    files = processor.scan_folder()

    assert len(files) == 1
    assert "audio1.wav" in files[0]


def test_get_files_to_process(temp_db: TranscriptionDatabase, temp_folder: Path) -> None:
    """Test getting list of files to process from folder."""
    # Create test audio files
    (temp_folder / "audio1.wav").touch()
    (temp_folder / "audio2.wav").touch()

    processor = _create_test_processor(temp_db, temp_folder)
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

    processor = _create_test_processor(temp_db, temp_folder)
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
        self, file_path: str, output_path: str, preset: str, language: str | None = None
    ) -> TranscriptionMetrics | None:
        self.calls.append((file_path, output_path, preset, language))
        if self.should_fail:
            raise RuntimeError("Test error")

        output_path_obj = Path(output_path)
        # Handle "both" format: if output_path is .txt, also create .json
        if output_path_obj.suffix == ".txt":
            # Create both .txt and .json files
            output_path_obj.write_text("Test transcription text\n")
            json_path = output_path_obj.with_suffix(".json")
            json_path.write_text("{}")
        elif output_path_obj.suffix == ".json":
            # Only JSON format
            output_path_obj.write_text("{}")
        else:
            # Default: write to the path as-is
            output_path_obj.write_text("{}")

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

    # Create mock transcription service
    from unittest.mock import Mock

    mock_transcription_service = Mock()
    mock_transcription_service.transcribe.return_value = TranscriptionResult(
        metrics=TranscriptionMetrics(
            audio_path=str(audio_file),
            preset="test-preset",
            requested_language=None,
            applied_language="en",
            detected_language="en",
            language_probability=0.95,
            audio_duration=60.0,
            total_processing_time=1.1,
            transcribe_duration=1.0,
            preprocess_duration=0.1,
            speed_ratio=120.0,
            preprocess_enabled=True,
            preprocess_profile="default",
            target_sample_rate=16000,
            target_channels=1,
            preprocess_snr_before=-10.0,
            preprocess_snr_after=-5.0,
            preprocess_steps=[],
        ),
        payload={"segments": [{"text": "test transcription"}]},
    )

    # Create processor with mock service
    processor = TranscriptionProcessor(
        transcription_service=mock_transcription_service,
        state_store=ServiceFactory.create_state_store(db_path=temp_db.db_path),
        file_mover=ServiceFactory.create_file_mover(),
        output_writer=ServiceFactory.create_output_writer(),
        input_folder=temp_folder,
    )
    result = processor.process_file(str(audio_file))

    assert result.status == "completed"
    assert result.metrics is not None
    mock_transcription_service.transcribe.assert_called_once()

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
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test handling of transcription failure.

    This test intentionally triggers a transcription error to verify error handling.
    Expected ERROR logs are captured and verified, not suppressed.
    """
    # Create test audio file
    audio_file = temp_folder / "audio1.wav"
    audio_file.touch()

    # Register file in database
    temp_db.add_file(str(audio_file), "pending")

    # Create mock transcription service that fails
    from unittest.mock import Mock

    mock_transcription_service = Mock()
    mock_transcription_service.transcribe.side_effect = RuntimeError("Mock transcription failure")

    # Capture logs at ERROR level to verify expected error logging
    with caplog.at_level(logging.ERROR, logger="backend.processor"):
        processor = TranscriptionProcessor(
            transcription_service=mock_transcription_service,
            state_store=ServiceFactory.create_state_store(db_path=temp_db.db_path),
            file_mover=ServiceFactory.create_file_mover(),
            output_writer=ServiceFactory.create_output_writer(),
            input_folder=temp_folder,
        )
        result = processor.process_file(str(audio_file))

    # Verify expected error was logged (this is intentional for this test)
    assert any("Failed to process" in record.message for record in caplog.records), (
        "Expected ERROR log for transcription failure (this is intentional)"
    )

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


def test_process_file_not_found(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test processing a non-existent file.

    This test intentionally processes a missing file to verify error handling.
    Expected ERROR logs are captured and verified, not suppressed.
    """
    audio_file = temp_folder / "nonexistent.wav"

    # Register file in database (but don't create it)
    temp_db.add_file(str(audio_file), "pending")

    # Capture logs at ERROR level to verify expected error logging
    with caplog.at_level(logging.ERROR, logger="backend.processor"):
        processor = _create_test_processor(temp_db, temp_folder)
        result = processor.process_file(str(audio_file))

    # Verify expected error was logged (this is intentional for this test)
    assert any("File not found" in record.message for record in caplog.records), (
        "Expected ERROR log for missing file (this is intentional)"
    )

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

    # Create mock transcription service
    from unittest.mock import Mock

    mock_transcription_service = Mock()
    mock_transcription_service.transcribe.return_value = TranscriptionResult(
        metrics=TranscriptionMetrics(
            audio_path="",
            preset="test-preset",
            requested_language=None,
            applied_language="en",
            detected_language="en",
            language_probability=0.95,
            audio_duration=60.0,
            total_processing_time=1.1,
            transcribe_duration=1.0,
            preprocess_duration=0.1,
            speed_ratio=120.0,
            preprocess_enabled=True,
            preprocess_profile="default",
            target_sample_rate=16000,
            target_channels=1,
            preprocess_snr_before=-10.0,
            preprocess_snr_after=-5.0,
            preprocess_steps=[],
        ),
        payload={"segments": [{"text": "test transcription"}]},
    )

    processor = TranscriptionProcessor(
        transcription_service=mock_transcription_service,
        state_store=ServiceFactory.create_state_store(db_path=temp_db.db_path),
        file_mover=ServiceFactory.create_file_mover(),
        output_writer=ServiceFactory.create_output_writer(),
        input_folder=temp_folder,
    )
    results = processor.process_all_files([str(audio1), str(audio2)])

    assert results["succeeded"] == 2
    assert results["failed"] == 0
    assert len(results["file_stats"]) == 2
    assert all(stat.status == "completed" for stat in results["file_stats"])
    assert mock_transcription_service.transcribe.call_count == 2


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

    # Create mock transcription service
    from unittest.mock import Mock

    mock_transcription_service = Mock()
    mock_transcription_service.transcribe.return_value = TranscriptionResult(
        metrics=TranscriptionMetrics(
            audio_path="",
            preset="test-preset",
            requested_language=None,
            applied_language="en",
            detected_language="en",
            language_probability=0.95,
            audio_duration=60.0,
            total_processing_time=1.1,
            transcribe_duration=0.5,
            preprocess_duration=0.1,
            speed_ratio=120.0,
            preprocess_enabled=True,
            preprocess_profile="default",
            target_sample_rate=16000,
            target_channels=1,
            preprocess_snr_before=-10.0,
            preprocess_snr_after=-5.0,
            preprocess_steps=[],
        ),
        payload={"segments": [{"text": "test transcription"}]},
    )

    processor = TranscriptionProcessor(
        transcription_service=mock_transcription_service,
        state_store=ServiceFactory.create_state_store(db_path=temp_db.db_path),
        file_mover=ServiceFactory.create_file_mover(),
        output_writer=ServiceFactory.create_output_writer(),
        input_folder=temp_folder,
    )
    results = processor.process_folder()

    assert results["files_found"] == 2
    assert results["succeeded"] == 2
    assert results["failed"] == 0
    stats = results["run_statistics"]
    assert stats["updated_db"] is True
    assert stats["run_id"] is not None
    assert mock_transcription_service.transcribe.call_count == 2
    # Verify statistics are correctly calculated and returned
    # For 2 files with default metrics: preprocess=0.1, transcribe=0.5 each
    assert stats["total_processing_time"] > 0
    assert stats["total_preprocess_time"] == 0.2  # 0.1 + 0.1
    assert stats["total_transcribe_time"] == 1.0  # 0.5 + 0.5
    assert stats["average_speed_ratio"] == 120.0  # (120.0 + 120.0) / 2


def test_process_file_move_failure_keeps_pending_status(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that if file move fails, database status is NOT updated to completed.

    This is a regression test for the bug where database status was updated
    before moving files, causing orphaned files that won't be reprocessed.

    This test intentionally triggers file move failures to verify error handling.
    Expected ERROR and WARNING logs are captured and verified, not suppressed.
    """
    # Create test audio file
    audio_file = temp_folder / "audio1.wav"
    audio_file.touch()

    # Register file
    temp_db.add_file(str(audio_file), "pending")

    # Create mock transcription service
    from unittest.mock import Mock

    mock_transcription_service = Mock()
    mock_transcription_service.transcribe.return_value = TranscriptionResult(
        metrics=TranscriptionMetrics(
            audio_path=str(audio_file),
            preset="test-preset",
            requested_language=None,
            applied_language="en",
            detected_language="en",
            language_probability=0.95,
            audio_duration=60.0,
            total_processing_time=1.1,
            transcribe_duration=1.0,
            preprocess_duration=0.1,
            speed_ratio=120.0,
            preprocess_enabled=True,
            preprocess_profile="default",
            target_sample_rate=16000,
            target_channels=1,
            preprocess_snr_before=-10.0,
            preprocess_snr_after=-5.0,
            preprocess_steps=[],
        ),
        payload={"segments": [{"text": "test transcription"}]},
    )

    # Create mock file mover that fails
    mock_file_mover = Mock()
    mock_file_mover.move.side_effect = Exception("Permission denied")

    # Capture logs at WARNING and ERROR levels to verify expected error logging
    with caplog.at_level(logging.WARNING, logger="backend.processor"):
        processor = TranscriptionProcessor(
            transcription_service=mock_transcription_service,
            state_store=ServiceFactory.create_state_store(db_path=temp_db.db_path),
            file_mover=mock_file_mover,
            output_writer=ServiceFactory.create_output_writer(),
            input_folder=temp_folder,
        )
        result = processor.process_file(str(audio_file))

    # Verify expected errors/warnings were logged (these are intentional for this test)
    assert any("Failed to process" in record.message for record in caplog.records), (
        "Expected ERROR log for move failure (this is intentional)"
    )
    assert any("Failed to move file" in record.message for record in caplog.records), (
        "Expected WARNING log for move failure (this is intentional)"
    )

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


def test_process_file_preserves_subdirectory_structure_with_output_base_dir(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test that files in subdirectories preserve directory structure when _output_base_dir is set.

    This prevents collisions when files with the same name exist in different subdirectories.
    """
    # Create subdirectories with files having the same name
    subdir1 = temp_folder / "subdir1"
    subdir2 = temp_folder / "subdir2"
    subdir1.mkdir()
    subdir2.mkdir()

    audio1 = subdir1 / "audio.wav"
    audio2 = subdir2 / "audio.wav"
    audio1.touch()
    audio2.touch()

    # Register files
    temp_db.add_file(str(audio1), "pending")
    temp_db.add_file(str(audio2), "pending")

    # Create output base directory (simulating multi-variant mode)
    output_base = temp_folder / "outputs"
    output_base.mkdir()

    # Create mock transcription service
    from unittest.mock import Mock

    mock_transcription_service = Mock()
    mock_transcription_service.transcribe.return_value = TranscriptionResult(
        metrics=TranscriptionMetrics(
            audio_path="",
            preset="test-preset",
            requested_language=None,
            applied_language="en",
            detected_language="en",
            language_probability=0.95,
            audio_duration=60.0,
            total_processing_time=1.1,
            transcribe_duration=1.0,
            preprocess_duration=0.1,
            speed_ratio=120.0,
            preprocess_enabled=True,
            preprocess_profile="default",
            target_sample_rate=16000,
            target_channels=1,
            preprocess_snr_before=-10.0,
            preprocess_snr_after=-5.0,
            preprocess_steps=[],
        ),
        payload={"segments": [{"text": "test transcription"}]},
    )

    processor = TranscriptionProcessor(
        transcription_service=mock_transcription_service,
        state_store=ServiceFactory.create_state_store(db_path=temp_db.db_path),
        file_mover=ServiceFactory.create_file_mover(),
        output_writer=ServiceFactory.create_output_writer(),
        input_folder=temp_folder,
        disable_file_moving=True,
        output_format="both",
    )
    processor._output_base_dir = output_base

    # Process both files
    result1 = processor.process_file(str(audio1))
    result2 = processor.process_file(str(audio2))

    assert result1.status == "completed"
    assert result2.status == "completed"

    # Verify outputs are in separate subdirectories, preventing collision
    output1_txt = output_base / "subdir1" / "audio.txt"
    output1_json = output_base / "subdir1" / "audio.json"
    output2_txt = output_base / "subdir2" / "audio.txt"
    output2_json = output_base / "subdir2" / "audio.json"

    assert output1_txt.exists(), f"Expected {output1_txt} to exist"
    assert output1_json.exists(), f"Expected {output1_json} to exist"
    assert output2_txt.exists(), f"Expected {output2_txt} to exist"
    assert output2_json.exists(), f"Expected {output2_json} to exist"

    # Verify they are different files (not overwritten)
    assert output1_txt != output2_txt
    assert output1_json != output2_json


def test_process_file_root_level_file_with_output_base_dir(
    temp_db: TranscriptionDatabase,
    temp_folder: Path,
) -> None:
    """Test that files at root level work correctly with _output_base_dir (no subdirectory created)."""
    # Create file at root level
    audio_file = temp_folder / "audio.wav"
    audio_file.touch()

    temp_db.add_file(str(audio_file), "pending")

    output_base = temp_folder / "outputs"
    output_base.mkdir()

    # Create mock transcription service
    from unittest.mock import Mock

    mock_transcription_service = Mock()
    mock_transcription_service.transcribe.return_value = TranscriptionResult(
        metrics=TranscriptionMetrics(
            audio_path=str(audio_file),
            preset="test-preset",
            requested_language=None,
            applied_language="en",
            detected_language="en",
            language_probability=0.95,
            audio_duration=60.0,
            total_processing_time=1.1,
            transcribe_duration=1.0,
            preprocess_duration=0.1,
            speed_ratio=120.0,
            preprocess_enabled=True,
            preprocess_profile="default",
            target_sample_rate=16000,
            target_channels=1,
            preprocess_snr_before=-10.0,
            preprocess_snr_after=-5.0,
            preprocess_steps=[],
        ),
        payload={"segments": [{"text": "test transcription"}]},
    )

    processor = TranscriptionProcessor(
        transcription_service=mock_transcription_service,
        state_store=ServiceFactory.create_state_store(db_path=temp_db.db_path),
        file_mover=ServiceFactory.create_file_mover(),
        output_writer=ServiceFactory.create_output_writer(),
        input_folder=temp_folder,
        disable_file_moving=True,
        output_format="both",
    )
    processor._output_base_dir = output_base

    result = processor.process_file(str(audio_file))

    assert result.status == "completed"

    # Verify output is directly in output_base, not in a subdirectory
    output_txt = output_base / "audio.txt"
    output_json = output_base / "audio.json"

    assert output_txt.exists(), f"Expected {output_txt} to exist"
    assert output_json.exists(), f"Expected {output_json} to exist"
    # Verify no unnecessary subdirectory was created (outputs should be directly in output_base)
    # Check that there are no subdirectories in output_base (only files)
    subdirs = [d for d in output_base.iterdir() if d.is_dir()]
    assert len(subdirs) == 0, f"Expected no subdirectories in {output_base}, found: {subdirs}"
