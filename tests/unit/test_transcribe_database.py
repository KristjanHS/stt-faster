"""Unit tests for transcription database operations."""

import inspect
import re
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from backend.database import (
    FileMetricRecord,
    RunRecord,
    TranscriptionDatabase,
    get_default_db_path,
)


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
    """Test that summary statistics match actual file counts (business logic verification)."""
    # Setup: add files with different statuses
    temp_db.add_file("/path/to/audio1.wav", "pending")
    temp_db.add_file("/path/to/audio2.wav", "pending")
    temp_db.add_file("/path/to/audio3.wav", "completed")
    temp_db.add_file("/path/to/audio4.wav", "failed")

    # Test behavior: summary totals should match file counts
    summary = temp_db.get_summary()

    # Verify business logic: totals should match actual file counts
    all_files = temp_db.get_all_files()
    total_files = len(all_files)
    summary_total = summary.get("pending", 0) + summary.get("completed", 0) + summary.get("failed", 0)

    assert summary_total == total_files, f"Summary total ({summary_total}) should match file count ({total_files})"
    assert summary.get("pending", 0) == 2, "Pending count should match"
    assert summary.get("completed", 0) == 1, "Completed count should match"
    assert summary.get("failed", 0) == 1, "Failed count should match"

    # Verify business logic: counts match when queried by status
    pending_files = temp_db.get_files_by_status("pending")
    completed_files = temp_db.get_files_by_status("completed")
    failed_files = temp_db.get_files_by_status("failed")

    assert len(pending_files) == summary.get("pending", 0), "Pending count should match query result"
    assert len(completed_files) == summary.get("completed", 0), "Completed count should match query result"
    assert len(failed_files) == summary.get("failed", 0), "Failed count should match query result"


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
    db_path = tmp_path / "test_migration.duckdb"

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


def test_migration_on_production_database() -> None:
    """Test that migrations work correctly on a copy of the production database.

    This test copies the actual production database (if it exists) and verifies
    that all migrations run successfully and all required columns exist.

    Note: This test requires the production database to exist.
    The production database is created when the application runs in production mode.
    To create it, run the application at least once, which will create the database
    at the default location (typically ~/.local/share/stt-faster/transcribe_state.duckdb).
    """
    import shutil
    import tempfile

    # Get production database path using the same function the application uses
    # This respects XDG_DATA_HOME environment variable if set
    prod_db_path = get_default_db_path()

    # Fail test if production database doesn't exist (required for migration testing)
    assert prod_db_path.exists(), (
        f"Production database not found at {prod_db_path}. "
        "This test requires the production database to exist. "
        "Run the application at least once to create the database, or set XDG_DATA_HOME "
        "if using a custom data directory."
    )

    # Create a temporary copy for testing
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
        test_db_path = tmp.name

    try:
        # Copy production database
        shutil.copy2(prod_db_path, test_db_path)

        # Open the copied database - this should trigger migrations
        db = TranscriptionDatabase(test_db_path)

        # Get all columns from file_metrics table
        columns_result = db.conn.execute("DESCRIBE file_metrics").fetchall()
        db_columns = {row[0] for row in columns_result}

        # Extract INSERT columns from source code
        from backend.database import TranscriptionDatabase as DB

        source = inspect.getsource(DB.record_file_metric)
        match = re.search(r"INSERT INTO file_metrics\s*\((.*?)\)\s*VALUES", source, re.DOTALL)
        assert match is not None, "Could not parse INSERT statement"
        columns_text = match.group(1)
        insert_columns = {col.strip() for col in columns_text.split(",") if col.strip()}

        # Verify all INSERT columns exist in the migrated database
        missing_in_db = insert_columns - db_columns
        assert not missing_in_db, (
            f"Migration failed: Columns missing from production database: {missing_in_db}. "
            f"These columns are referenced in INSERT but don't exist in the database. "
            f"Add these columns to _migrate_schema() in database.py"
        )

        # Verify critical columns that were causing production errors
        critical_columns = {
            "snr_estimation_method",
            "patience",
            "loudnorm_target_lra",
            "rnnoise_model",
            "rnnoise_mix",
        }
        missing_critical = critical_columns - db_columns
        assert not missing_critical, (
            f"Critical columns missing after migration: {missing_critical}. "
            f"These columns were causing production errors."
        )

        # Test that we can actually insert a record (ultimate validation)
        test_record = FileMetricRecord(
            run_id=1,
            recorded_at=datetime.now(timezone.utc),
            audio_path="/test/migration_test.mp3",
            preset="turbo",
            status="completed",
            total_processing_time=10.0,
            transcribe_duration=8.0,
            preprocess_duration=2.0,
            preprocess_enabled=True,
            preprocess_profile="cpu",
            target_sample_rate=16000,
            target_channels=1,
            snr_estimation_method="estimate_snr_db",
            patience=1.0,
            loudnorm_target_lra=7.0,
        )

        # This should not raise any schema-related errors
        db.record_file_metric(test_record)

        db.close()
    finally:
        # Cleanup
        Path(test_db_path).unlink(missing_ok=True)


def test_schema_migration_adds_rnnoise_mix_column(tmp_path: Path) -> None:
    """Test that migration adds rnnoise_mix column to existing databases."""
    db_path = tmp_path / "test_migration_mix.duckdb"

    # Create a database with old schema (without rnnoise_mix column)
    # This simulates an existing database created before rnnoise_mix was added
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
            rnnoise_model VARCHAR,
            -- rnnoise_mix column is MISSING (old schema)

            -- Audio inspection
            input_channels INTEGER,
            input_sample_rate INTEGER,
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
    assert "rnnoise_mix" in column_names, "Migration should have added rnnoise_mix column"

    # Verify we can insert a record with rnnoise_mix
    record = FileMetricRecord(
        run_id=1,
        recorded_at=datetime.now(timezone.utc),
        audio_path="/test/audio2.mp3",
        preset="turbo",
        status="completed",
        preprocess_enabled=True,
        preprocess_profile="cpu",
        target_sample_rate=16000,
        target_channels=1,
        rnnoise_model="models/sh.rnnn",
        rnnoise_mix=0.75,
        total_processing_time=10.0,
        transcribe_duration=8.0,
        preprocess_duration=2.0,
    )

    # This should not raise an error
    db.record_file_metric(record)

    # Verify the data was stored correctly
    result = db.conn.execute(
        "SELECT rnnoise_mix FROM file_metrics WHERE audio_path = ?",
        ("/test/audio2.mp3",),
    ).fetchone()
    assert result is not None
    assert result[0] == 0.75

    db.close()


def test_schema_migration_adds_both_rnnoise_columns(tmp_path: Path) -> None:
    """Test that migration adds both rnnoise_model and rnnoise_mix columns when both are missing."""
    db_path = tmp_path / "test_migration_both.duckdb"

    # Create a database with very old schema (without both rnnoise columns)
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
            -- Both rnnoise_model and rnnoise_mix columns are MISSING (very old schema)

            -- Audio inspection
            input_channels INTEGER,
            input_sample_rate INTEGER,
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

    # Verify both columns were added
    columns_result = db.conn.execute("DESCRIBE file_metrics").fetchall()
    column_names = {row[0] for row in columns_result}
    assert "rnnoise_model" in column_names, "Migration should have added rnnoise_model column"
    assert "rnnoise_mix" in column_names, "Migration should have added rnnoise_mix column"

    # Verify we can insert a record with both fields
    record = FileMetricRecord(
        run_id=1,
        recorded_at=datetime.now(timezone.utc),
        audio_path="/test/audio3.mp3",
        preset="turbo",
        status="completed",
        preprocess_enabled=True,
        preprocess_profile="cpu",
        target_sample_rate=16000,
        target_channels=1,
        rnnoise_model="models/sh.rnnn",
        rnnoise_mix=0.6,
        total_processing_time=10.0,
        transcribe_duration=8.0,
        preprocess_duration=2.0,
    )

    # This should not raise an error
    db.record_file_metric(record)

    # Verify both values were stored correctly
    result = db.conn.execute(
        "SELECT rnnoise_model, rnnoise_mix FROM file_metrics WHERE audio_path = ?",
        ("/test/audio3.mp3",),
    ).fetchone()
    assert result is not None
    assert result[0] == "models/sh.rnnn"
    assert result[1] == 0.6

    db.close()


def test_file_metrics_schema_consistency(temp_db: TranscriptionDatabase) -> None:
    """Test that FileMetricRecord dataclass, INSERT statement, and database schema are consistent.

    This test catches schema mismatches early, preventing production errors like:
    "Table 'file_metrics' does not have a column with name 'loudnorm_target_lra'"

    It validates:
    1. All columns in the INSERT statement exist in the database
    2. All dataclass fields map to database columns (with special handling for transformations)
    3. The database schema matches what the code expects
    """
    # Get actual database columns
    columns_result = temp_db.conn.execute("DESCRIBE file_metrics").fetchall()
    db_columns = {row[0] for row in columns_result}

    # Extract column names from the INSERT statement in record_file_metric
    # Read the actual INSERT statement from the source code to ensure we test against reality
    from backend.database import TranscriptionDatabase as DB

    # Get the source code of record_file_metric method
    source = inspect.getsource(DB.record_file_metric)

    # Extract the INSERT statement from the source
    match = re.search(r"INSERT INTO file_metrics\s*\((.*?)\)\s*VALUES", source, re.DOTALL)
    assert match is not None, "Could not parse INSERT statement"
    columns_text = match.group(1)
    insert_columns = {col.strip() for col in columns_text.split(",") if col.strip()}

    # Special mappings: fields that are transformed before insertion
    # - preprocess_steps -> preprocess_steps_json (stored as JSON)
    # - recorded_at is handled specially (formatted to string)
    field_to_column_mapping = {
        "preprocess_steps": "preprocess_steps_json",
        "recorded_at": "recorded_at",  # Same name, but handled specially
    }

    # Expected database columns from dataclass (excluding id which is auto-generated)
    expected_columns_from_dataclass = set()
    for field in fields(FileMetricRecord):
        if field.name in field_to_column_mapping:
            expected_columns_from_dataclass.add(field_to_column_mapping[field.name])
        else:
            expected_columns_from_dataclass.add(field.name)

    # Remove id as it's auto-generated by the database
    expected_columns_from_dataclass.discard("id")

    # Validation 1: All INSERT columns must exist in database
    missing_in_db = insert_columns - db_columns
    assert not missing_in_db, (
        f"INSERT statement references columns that don't exist in database: {missing_in_db}. "
        f"This will cause production errors. Add these columns to the schema or migration."
    )

    # Validation 2: All dataclass fields (with transformations) should map to database columns
    missing_dataclass_fields = expected_columns_from_dataclass - db_columns
    assert not missing_dataclass_fields, (
        f"FileMetricRecord has fields that don't exist in database: {missing_dataclass_fields}. "
        f"This indicates a schema mismatch. Add these columns to the schema or migration."
    )

    # Validation 3: All INSERT columns should be in the dataclass (or be special like preprocess_steps_json)
    # preprocess_steps_json is derived from preprocess_steps, so it's expected
    special_columns = {"id", "preprocess_steps_json"}  # id is auto-generated, preprocess_steps_json is derived
    insert_only_columns = insert_columns - expected_columns_from_dataclass - special_columns
    assert not insert_only_columns, (
        f"INSERT statement references columns not in FileMetricRecord: {insert_only_columns}. "
        f"This indicates a code/schema mismatch."
    )

    # Validation 4: Test that we can actually insert a record with ALL fields populated
    # This is the ultimate test - if the schema matches, insertion should work
    # By populating ALL fields, we catch any missing columns that weren't caught by static analysis
    test_record = FileMetricRecord(
        # Required fields
        run_id=1,
        recorded_at=datetime.now(timezone.utc),
        audio_path="/test/schema_validation.mp3",
        preset="turbo",
        status="completed",
        # Language detection
        requested_language="en",
        applied_language="en",
        detected_language="en",
        language_probability=0.95,
        # Timing metrics
        audio_duration=120.5,
        total_processing_time=10.0,
        transcribe_duration=8.0,
        preprocess_duration=2.0,
        speed_ratio=2.5,
        # Preprocessing
        preprocess_enabled=True,
        preprocess_profile="cpu",
        target_sample_rate=16000,
        target_channels=1,
        preprocess_snr_before=15.0,
        preprocess_snr_after=20.0,
        preprocess_steps=[{"step": "test", "enabled": True}],
        rnnoise_model="test_model",
        rnnoise_mix=0.5,
        # Audio inspection
        input_channels=2,
        input_sample_rate=44100,
        input_format="mp3",
        # Downmix/resample parameters
        volume_adjustment_db=-3.0,
        resampler="soxr",
        sample_format="s16",
        # Loudness normalization parameters (including the one that caused production error)
        loudnorm_preset="ebu",
        loudnorm_target_i=-23.0,
        loudnorm_target_tp=-2.0,
        loudnorm_target_lra=7.0,  # This was the field that caused the production error
        loudnorm_backend="ffmpeg",
        # Denoise parameters
        denoise_method="rnnoise",
        denoise_library="rnnoise",
        # SNR estimation
        snr_estimation_method="ffmpeg",
        # Transcription parameters
        beam_size=5,
        patience=1.0,
        word_timestamps=True,
        task="transcribe",
        chunk_length=30,
        vad_filter=True,
        vad_threshold=0.5,
        vad_min_speech_duration_ms=250,
        vad_max_speech_duration_s=30.0,
        vad_min_silence_duration_ms=100,
        vad_speech_pad_ms=400,
        temperature="[0.0,0.2,0.4,0.6,0.8,1.0]",
        temperature_increment_on_fallback=0.2,
        best_of=5,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        length_penalty=1.0,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        suppress_tokens="-1",
        condition_on_previous_text=True,
        initial_prompt="Hello",
        # Model parameters
        model_id="base",
        device="cpu",
        compute_type="int8",
        # Output parameters
        output_format="json",
        float_precision=2,
        # Error
        error_message=None,
    )

    # This should not raise any schema-related errors
    # By testing with ALL fields, we catch any missing columns that static analysis might miss
    try:
        temp_db.record_file_metric(test_record)
    except Exception as e:
        error_msg = str(e)
        if "does not have a column" in error_msg or "no such column" in error_msg.lower():
            # Extract the missing column name from the error
            column_match = re.search(r"column.*?['\"]([^'\"]+)['\"]", error_msg, re.IGNORECASE)
            missing_column = column_match.group(1) if column_match else "unknown"
            raise AssertionError(
                f"Schema mismatch detected: Column '{missing_column}' is missing from database. "
                f"This test would have caught the production error. "
                f"Add '{missing_column}' to the schema or migration. "
                f"Original error: {e}"
            ) from e
        raise  # Re-raise if it's a different error


def test_file_metrics_schema_consistency_catches_missing_column(tmp_path: Path) -> None:
    """Test that schema consistency check catches missing columns like loudnorm_target_lra and snr_estimation_method.

    This simulates the production error scenario where an old database exists
    without columns that the code expects, and verifies our test would catch it.
    """
    db_path = tmp_path / "test_missing_column.duckdb"

    # Create a database with old schema missing loudnorm_target_lra and snr_estimation_method
    # (simulating the production error)
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

            -- Preprocessing
            preprocess_enabled BOOLEAN NOT NULL,
            preprocess_profile VARCHAR NOT NULL,
            target_sample_rate INTEGER NOT NULL,
            target_channels INTEGER,
            preprocess_snr_before DOUBLE,
            preprocess_snr_after DOUBLE,
            rnnoise_model VARCHAR,
            rnnoise_mix DOUBLE,

            -- Audio inspection
            input_channels INTEGER,
            input_sample_rate INTEGER,
            input_format VARCHAR,

            -- Downmix/resample parameters
            volume_adjustment_db DOUBLE,
            resampler VARCHAR,
            sample_format VARCHAR,

            -- Loudness normalization parameters
            loudnorm_preset VARCHAR,
            loudnorm_target_i DOUBLE,
            loudnorm_target_tp DOUBLE,
            -- loudnorm_target_lra DOUBLE,  <-- MISSING! This causes the production error
            loudnorm_backend VARCHAR,

            -- Denoise parameters
            denoise_method VARCHAR,
            denoise_library VARCHAR,

            -- SNR estimation
            -- snr_estimation_method VARCHAR,  <-- MISSING! This causes the production error

            -- Transcription parameters
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

            -- Model parameters
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

    # Now open with TranscriptionDatabase - migration should add missing columns
    # But if migration doesn't handle loudnorm_target_lra or snr_estimation_method, our test should catch it
    db = TranscriptionDatabase(db_path)

    # Get database columns
    columns_result = db.conn.execute("DESCRIBE file_metrics").fetchall()
    db_columns = {row[0] for row in columns_result}

    # Extract INSERT columns from source code (same logic as main test)
    from backend.database import TranscriptionDatabase as DB

    source = inspect.getsource(DB.record_file_metric)
    match = re.search(r"INSERT INTO file_metrics\s*\((.*?)\)\s*VALUES", source, re.DOTALL)
    assert match is not None, "Could not parse INSERT statement"
    columns_text = match.group(1)
    insert_columns = {col.strip() for col in columns_text.split(",") if col.strip()}

    # After migration, all INSERT columns should exist in the database
    # This test enforces that migrations actually add missing columns
    missing_in_db = insert_columns - db_columns

    # Verify that loudnorm_target_lra was added by migration
    assert "loudnorm_target_lra" in db_columns, (
        "Column loudnorm_target_lra should exist after migration. "
        "The migration must add this column to prevent production errors. "
        "If this test fails, add loudnorm_target_lra to _migrate_schema() in database.py"
    )

    # Verify that snr_estimation_method was added by migration
    assert "snr_estimation_method" in db_columns, (
        "Column snr_estimation_method should exist after migration. "
        "The migration must add this column to prevent production errors. "
        "If this test fails, add snr_estimation_method to _migrate_schema() in database.py"
    )

    # If there are any other missing columns, that's also a problem
    # (though the main schema consistency test should catch most of these)
    assert not missing_in_db, (
        f"Migration did not add all required columns. Missing: {missing_in_db}. "
        f"This will cause production errors. Add these columns to _migrate_schema() in database.py"
    )

    db.close()
