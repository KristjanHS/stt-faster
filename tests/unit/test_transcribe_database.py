"""Unit tests for transcription database operations."""

from datetime import datetime, timezone
from pathlib import Path

import duckdb

from backend.database import FileMetricRecord, RunRecord, TranscriptionDatabase


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


def test_record_run(temp_db: TranscriptionDatabase) -> None:
    """Test that run metadata can be recorded."""
    record = RunRecord(
        recorded_at=datetime(2025, 1, 1, 0, 0),
        input_folder="/tmp/input",
        preset="turbo",
        language="en",
        preprocess_enabled=True,
        preprocess_profile="cpu",
        target_sample_rate=16000,
        target_channels=1,
        files_found=2,
        succeeded=2,
        failed=0,
        total_processing_time=120.0,
        total_preprocess_time=30.0,
        total_transcribe_time=90.0,
        total_audio_duration=180.0,
        speed_ratio=2.0,
    )

    temp_db.record_run(record)
    history = temp_db.get_run_history()

    assert len(history) == 1
    entry = history[0]
    assert entry["input_folder"] == "/tmp/input"
    assert entry["preset"] == "turbo"
    assert entry["files_found"] == 2


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


def test_schema_migration_adds_rnnoise_model_column(tmp_path: Path) -> None:
    """Test that migration adds rnnoise_model column to existing databases."""
    db_path = tmp_path / "test_migration.db"

    # Create a database with old schema (without rnnoise_model column)
    # This simulates an existing database created before rnnoise_model was added
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_metrics (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            recorded_at TIMESTAMP NOT NULL,
            audio_path VARCHAR NOT NULL,
            preset VARCHAR NOT NULL,
            status VARCHAR NOT NULL,

            -- Language detection
            requested_language VARCHAR,
            applied_language VARCHAR,
            detected_language VARCHAR,
            language_probability DOUBLE,

            -- Timing metrics
            audio_duration DOUBLE,
            total_processing_time DOUBLE NOT NULL,
            transcribe_duration DOUBLE NOT NULL,
            preprocess_duration DOUBLE NOT NULL,
            speed_ratio DOUBLE,

            -- Preprocessing (Outcomes/Specifics)
            preprocess_enabled BOOLEAN NOT NULL,
            preprocess_profile VARCHAR NOT NULL,
            target_sample_rate INTEGER NOT NULL,
            target_channels INTEGER,
            preprocess_snr_before DOUBLE,
            preprocess_snr_after DOUBLE,
            -- rnnoise_model column is MISSING (old schema)
            rnnoise_mix DOUBLE,

            -- Audio inspection
            input_channels INTEGER,
            input_sample_rate INTEGER,
            input_bit_depth INTEGER,
            input_format VARCHAR,

            -- Downmix/resample parameters used
            volume_adjustment_db DOUBLE,
            resampler VARCHAR,
            sample_format VARCHAR,

            -- Loudness normalization parameters used
            loudnorm_preset VARCHAR,
            loudnorm_target_i DOUBLE,
            loudnorm_target_tp DOUBLE,
            loudnorm_target_lra DOUBLE,
            loudnorm_backend VARCHAR,

            -- Denoise parameters used
            denoise_method VARCHAR,
            denoise_library VARCHAR,

            -- SNR estimation
            snr_estimation_method VARCHAR,

            -- Transcription parameters used
            beam_size INTEGER,
            patience DOUBLE,
            word_timestamps BOOLEAN,
            task VARCHAR,
            chunk_length INTEGER,
            vad_filter BOOLEAN,
            vad_threshold DOUBLE,
            vad_min_speech_duration_ms INTEGER,
            vad_max_speech_duration_s DOUBLE,
            vad_min_silence_duration_ms INTEGER,
            vad_speech_pad_ms INTEGER,
            temperature VARCHAR,
            temperature_increment_on_fallback DOUBLE,
            best_of INTEGER,
            compression_ratio_threshold DOUBLE,
            logprob_threshold DOUBLE,
            no_speech_threshold DOUBLE,
            length_penalty DOUBLE,
            repetition_penalty DOUBLE,
            no_repeat_ngram_size INTEGER,
            suppress_tokens VARCHAR,
            condition_on_previous_text BOOLEAN,
            initial_prompt VARCHAR,

            -- Model parameters used
            model_id VARCHAR,
            device VARCHAR,
            compute_type VARCHAR,

            -- Output parameters
            output_format VARCHAR,
            float_precision INTEGER,

            -- Complex/Error
            preprocess_steps_json VARCHAR,
            error_message VARCHAR
        )
    """)
    conn.close()

    # Now open with TranscriptionDatabase - should trigger migration
    db = TranscriptionDatabase(db_path)

    # Verify the column was added
    columns_result = db.conn.execute("DESCRIBE file_metrics").fetchall()
    column_names = {row[0] for row in columns_result}
    assert "rnnoise_model" in column_names, "Migration should have added rnnoise_model column"

    # Verify we can insert a record with rnnoise_model
    record = FileMetricRecord(
        run_id=1,
        recorded_at=datetime.now(timezone.utc),
        audio_path="/test/audio.mp3",
        preset="turbo",
        status="completed",
        preprocess_enabled=True,
        preprocess_profile="cpu",
        target_sample_rate=16000,
        target_channels=1,
        rnnoise_model="models/sh.rnnn",
        rnnoise_mix=0.5,
        total_processing_time=10.0,
        transcribe_duration=8.0,
        preprocess_duration=2.0,
    )

    # This should not raise an error
    db.record_file_metric(record)

    # Verify the data was stored correctly
    result = db.conn.execute(
        "SELECT rnnoise_model FROM file_metrics WHERE audio_path = ?",
        ("/test/audio.mp3",),
    ).fetchone()
    assert result is not None
    assert result[0] == "models/sh.rnnn"

    db.close()
