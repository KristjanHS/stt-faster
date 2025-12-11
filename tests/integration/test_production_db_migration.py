"""Integration test for production database migration and data insertion.

This test validates that:
1. Production database can be migrated successfully
2. New run data can be added to migrated database
3. New file metric data can be added to migrated database
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.database import FileMetricRecord, RunRecord, TranscriptionDatabase


def test_production_db_migration_and_data_insertion() -> None:
    """Test that production database migration works and new data can be inserted.

    This integration test:
    1. Copies the production database
    2. Verifies migrations run successfully
    3. Adds a new run record
    4. Adds new file metric records linked to that run
    5. Verifies the data was inserted correctly
    """
    # Get production database path
    prod_db_path = Path.home() / ".local" / "share" / "stt-faster" / "transcribe_state.duckdb"

    # Skip test if production database doesn't exist
    if not prod_db_path.exists():
        pytest.skip("Production database not found - skipping migration integration test")

    # Create a temporary copy for testing
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as tmp:
        test_db_path = tmp.name

    try:
        # Step 1: Copy production database
        shutil.copy2(prod_db_path, test_db_path)

        # Step 2: Open the copied database - this should trigger migrations
        db = TranscriptionDatabase(test_db_path)

        # Step 3: Verify migrations ran successfully by checking critical columns
        columns_result = db.conn.execute("DESCRIBE file_metrics").fetchall()
        db_columns = {row[0] for row in columns_result}

        # Verify critical columns that were causing production errors exist
        critical_columns = {
            "snr_estimation_method",
            "patience",
            "loudnorm_target_lra",
            "rnnoise_model",
            "rnnoise_mix",
            "task",
            "chunk_length",
            "vad_filter",
        }
        missing_critical = critical_columns - db_columns
        assert not missing_critical, (
            f"Critical columns missing after migration: {missing_critical}. Migration did not complete successfully."
        )

        # Step 4: Get initial run count to verify we're adding new data
        initial_runs = db.get_run_history()
        initial_run_count = len(initial_runs)

        # Step 5: Create and insert a new run record
        new_run = RunRecord(
            recorded_at=datetime.now(timezone.utc),
            input_folder="/test/integration/audio",
            preset="turbo",
            language="en",
            preprocess_enabled=True,
            preprocess_profile="cpu",
            target_sample_rate=16000,
            target_channels=1,
            loudnorm_preset="ebu",
            model_id="base",
            device="cpu",
            compute_type="int8",
            beam_size=5,
            patience=1.0,
            word_timestamps=True,
            task="transcribe",
            chunk_length=30,
            vad_filter=True,
            vad_threshold=0.5,
            files_found=2,
            succeeded=2,
            failed=0,
            total_processing_time=45.5,
            total_preprocess_time=10.2,
            total_transcribe_time=35.3,
            total_audio_duration=120.0,
            speed_ratio=2.5,
        )

        run_id = db.record_run(new_run)

        # Verify run was inserted
        runs_after = db.get_run_history()
        assert len(runs_after) == initial_run_count + 1, "New run should be added to database"

        # Verify the inserted run data
        inserted_run = runs_after[0]  # Most recent run is first
        assert inserted_run["id"] == run_id
        assert inserted_run["preset"] == "turbo"
        assert inserted_run["input_folder"] == "/test/integration/audio"
        assert inserted_run["files_found"] == 2
        assert inserted_run["succeeded"] == 2
        assert inserted_run["failed"] == 0
        assert inserted_run["patience"] == 1.0
        assert inserted_run["task"] == "transcribe"

        # Step 6: Create and insert file metric records linked to the new run
        file_metrics = [
            FileMetricRecord(
                run_id=run_id,
                recorded_at=datetime.now(timezone.utc),
                audio_path="/test/integration/audio/file1.wav",
                preset="turbo",
                status="completed",
                requested_language="en",
                applied_language="en",
                detected_language="en",
                language_probability=0.98,
                audio_duration=60.0,
                total_processing_time=22.5,
                transcribe_duration=18.0,
                preprocess_duration=4.5,
                speed_ratio=2.5,
                preprocess_enabled=True,
                preprocess_profile="cpu",
                target_sample_rate=16000,
                target_channels=1,
                preprocess_snr_before=15.5,
                preprocess_snr_after=20.3,
                rnnoise_model="models/sh.rnnn",
                rnnoise_mix=0.6,
                input_channels=2,
                input_sample_rate=44100,
                input_bit_depth=16,
                input_format="wav",
                volume_adjustment_db=-2.0,
                resampler="soxr",
                sample_format="s16",
                loudnorm_preset="ebu",
                loudnorm_target_i=-23.0,
                loudnorm_target_tp=-2.0,
                loudnorm_target_lra=7.0,
                loudnorm_backend="ffmpeg",
                denoise_method="rnnoise",
                denoise_library="rnnoise",
                snr_estimation_method="estimate_snr_db",
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
                model_id="base",
                device="cpu",
                compute_type="int8",
                output_format="json",
                float_precision=2,
            ),
            FileMetricRecord(
                run_id=run_id,
                recorded_at=datetime.now(timezone.utc),
                audio_path="/test/integration/audio/file2.mp3",
                preset="turbo",
                status="completed",
                requested_language="en",
                applied_language="en",
                detected_language="en",
                language_probability=0.95,
                audio_duration=60.0,
                total_processing_time=23.0,
                transcribe_duration=17.3,
                preprocess_duration=5.7,
                speed_ratio=2.6,
                preprocess_enabled=True,
                preprocess_profile="cpu",
                target_sample_rate=16000,
                target_channels=1,
                preprocess_snr_before=14.2,
                preprocess_snr_after=19.8,
                rnnoise_model="models/sh.rnnn",
                rnnoise_mix=0.6,
                input_channels=2,
                input_sample_rate=48000,
                input_bit_depth=24,
                input_format="mp3",
                volume_adjustment_db=-1.5,
                resampler="soxr",
                sample_format="s16",
                loudnorm_preset="ebu",
                loudnorm_target_i=-23.0,
                loudnorm_target_tp=-2.0,
                loudnorm_target_lra=7.0,
                loudnorm_backend="ffmpeg",
                denoise_method="rnnoise",
                denoise_library="rnnoise",
                snr_estimation_method="estimate_snr_db",
                beam_size=5,
                patience=1.0,
                word_timestamps=True,
                task="transcribe",
                chunk_length=30,
                vad_filter=True,
                vad_threshold=0.5,
                model_id="base",
                device="cpu",
                compute_type="int8",
                output_format="json",
                float_precision=2,
            ),
        ]

        # Insert file metrics
        for file_metric in file_metrics:
            db.record_file_metric(file_metric)

        # Step 7: Verify file metrics were inserted correctly
        # Query file metrics for the new run
        file_metrics_result = db.conn.execute(
            "SELECT * FROM file_metrics WHERE run_id = ? ORDER BY audio_path",
            (run_id,),
        ).fetchall()

        assert len(file_metrics_result) == 2, "Both file metrics should be inserted"

        # Verify first file metric
        first_file = file_metrics_result[0]
        assert first_file[3] == "/test/integration/audio/file1.wav"  # audio_path column
        assert first_file[4] == "turbo"  # preset column
        assert first_file[5] == "completed"  # status column
        # Verify critical columns that were causing production errors
        # Find column indices by checking the schema
        columns_list = [row[0] for row in columns_result]
        snr_idx = columns_list.index("snr_estimation_method")
        patience_idx = columns_list.index("patience")
        lra_idx = columns_list.index("loudnorm_target_lra")

        assert first_file[snr_idx] == "estimate_snr_db"
        assert first_file[patience_idx] == 1.0
        assert first_file[lra_idx] == 7.0

        # Verify second file metric
        second_file = file_metrics_result[1]
        assert second_file[3] == "/test/integration/audio/file2.mp3"  # audio_path column
        assert second_file[4] == "turbo"  # preset column
        assert second_file[5] == "completed"  # status column

        # Step 8: Verify we can query the data back using database methods
        # Get run history and verify our new run is there
        final_runs = db.get_run_history()
        assert len(final_runs) == initial_run_count + 1

        # Verify the run summary matches
        summary = db.get_summary()
        # Summary should reflect all files in the database, including our new ones
        all_files = db.get_all_files()
        summary_total = sum(summary.values())
        assert summary_total == len(all_files), "Summary total should match file count"

        db.close()

    finally:
        # Cleanup: remove the test database copy
        Path(test_db_path).unlink(missing_ok=True)
