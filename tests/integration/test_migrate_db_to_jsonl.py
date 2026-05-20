"""Integration test for ``scripts/migrate_db_to_jsonl.py``.

Builds a small synthetic DuckDB with the production schema (runs + run_configs
+ run_metrics + run_parameters + file_metrics + transcriptions), runs the
migration script as a subprocess, and verifies the JSONL output preserves
historical ids, embeds per-file metrics, doesn't null-stomp wide-row data
when the narrow tables are absent (plan §5 R-1), and is byte-deterministic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "migrate_db_to_jsonl.py"


def _init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Recreate the minimum tables the migration script reads."""
    # Mirror the schema in backend/database/schema.py:create_initial_tables —
    # only the columns the migration script actually reads matter, but we keep
    # the wide-row shape so the SELECT * works.
    conn.execute("""
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY,
            recorded_at TIMESTAMP NOT NULL,
            input_folder VARCHAR,
            preset VARCHAR NOT NULL,
            language VARCHAR,
            preprocess_enabled BOOLEAN NOT NULL,
            files_found INTEGER NOT NULL,
            succeeded INTEGER NOT NULL,
            failed INTEGER NOT NULL,
            total_processing_time DOUBLE,
            total_audio_duration DOUBLE,
            speed_ratio DOUBLE,
            -- Wide columns the migration also looks at (older shape):
            preprocess_profile VARCHAR,
            target_sample_rate INTEGER,
            target_channels INTEGER,
            loudnorm_preset VARCHAR,
            volume_adjustment_db DOUBLE,
            resampler VARCHAR,
            sample_format VARCHAR,
            loudnorm_target_i DOUBLE,
            loudnorm_target_tp DOUBLE,
            loudnorm_target_lra DOUBLE,
            loudnorm_backend VARCHAR,
            denoise_method VARCHAR,
            denoise_library VARCHAR,
            rnnoise_model VARCHAR,
            rnnoise_mix DOUBLE,
            snr_estimation_method VARCHAR,
            model_id VARCHAR,
            device VARCHAR,
            compute_type VARCHAR,
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
            total_preprocess_time DOUBLE,
            total_transcribe_time DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE run_configs (
            run_id INTEGER PRIMARY KEY,
            model_id VARCHAR,
            device VARCHAR,
            compute_type VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE run_metrics (
            run_id INTEGER PRIMARY KEY,
            total_preprocess_time DOUBLE,
            total_transcribe_time DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE run_parameters (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            category VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            value VARCHAR,
            value_type VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE file_metrics (
            id INTEGER PRIMARY KEY,
            run_id INTEGER,
            recorded_at TIMESTAMP NOT NULL,
            audio_path VARCHAR NOT NULL,
            preset VARCHAR NOT NULL,
            status VARCHAR NOT NULL,
            requested_language VARCHAR,
            applied_language VARCHAR,
            detected_language VARCHAR,
            language_probability DOUBLE,
            audio_duration DOUBLE,
            total_processing_time DOUBLE NOT NULL,
            transcribe_duration DOUBLE NOT NULL,
            preprocess_duration DOUBLE NOT NULL,
            speed_ratio DOUBLE,
            preprocess_enabled BOOLEAN NOT NULL,
            preprocess_profile VARCHAR,
            target_sample_rate INTEGER,
            target_channels INTEGER,
            preprocess_snr_before DOUBLE,
            preprocess_snr_after DOUBLE,
            rnnoise_model VARCHAR,
            rnnoise_mix DOUBLE,
            input_channels INTEGER,
            input_sample_rate INTEGER,
            input_format VARCHAR,
            volume_adjustment_db DOUBLE,
            resampler VARCHAR,
            sample_format VARCHAR,
            loudnorm_preset VARCHAR,
            loudnorm_target_i DOUBLE,
            loudnorm_target_tp DOUBLE,
            loudnorm_target_lra DOUBLE,
            loudnorm_backend VARCHAR,
            denoise_method VARCHAR,
            denoise_library VARCHAR,
            snr_estimation_method VARCHAR,
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
            model_id VARCHAR,
            device VARCHAR,
            compute_type VARCHAR,
            output_format VARCHAR,
            float_precision INTEGER,
            preprocess_steps_json VARCHAR,
            error_message VARCHAR
        )
    """)


def _seed(conn: duckdb.DuckDBPyConnection) -> None:
    """Populate two runs:

    - run 1: NEW shape — has run_configs + run_metrics rows.
    - run 2: OLD shape — only the wide ``runs`` row, recovers model info via
      ``run_parameters``. R-1: the wide-row ``preset``/``language`` must NOT
      be null-stomped just because run_configs is absent.
    """
    ts1 = datetime(2025, 12, 8, 10, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 5, 18, 15, 6, 38, tzinfo=timezone.utc)

    # Run 1 — newer shape.
    conn.execute(
        """
        INSERT INTO runs (
            id, recorded_at, input_folder, preset, language, preprocess_enabled,
            files_found, succeeded, failed, total_processing_time, total_audio_duration, speed_ratio,
            preprocess_profile, target_sample_rate, target_channels,
            beam_size, vad_filter, vad_threshold
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            1,
            ts1,
            "/in/folder1",
            "turbo",
            "en",
            True,
            2,
            2,
            0,
            301.8,
            6930.77,
            23.31,
            "cpu",
            16000,
            1,
            5,
            True,
            0.5,
        ],
    )
    conn.execute(
        "INSERT INTO run_configs (run_id, model_id, device, compute_type) VALUES (?, ?, ?, ?)",
        [1, "Systran/faster-distil-whisper-large-v3", "cuda", "float16"],
    )
    conn.execute(
        "INSERT INTO run_metrics (run_id, total_preprocess_time, total_transcribe_time) VALUES (?, ?, ?)",
        [1, 1.42, 297.28],
    )

    # Run 2 — older shape: no run_configs, no run_metrics. Wide-row carries
    # preset/language/model_id but run_parameters also holds device/compute_type
    # that we want to recover.
    conn.execute(
        """
        INSERT INTO runs (
            id, recorded_at, input_folder, preset, language, preprocess_enabled,
            files_found, succeeded, failed, total_processing_time, total_audio_duration, speed_ratio,
            model_id, beam_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            2,
            ts2,
            "/in/folder2",
            "et-large",
            "et",
            True,
            1,
            0,
            1,
            500.0,
            6000.0,
            12.0,
            "old-wide-model-id",
            3,
        ],
    )
    # Backfill the missing fields via run_parameters.
    conn.execute(
        "INSERT INTO run_parameters (id, run_id, category, name, value, value_type) VALUES (?, ?, ?, ?, ?, ?)",
        [1, 2, "model", "device", "cpu", "str"],
    )
    conn.execute(
        "INSERT INTO run_parameters (id, run_id, category, name, value, value_type) VALUES (?, ?, ?, ?, ?, ?)",
        [2, 2, "model", "compute_type", "int8", "str"],
    )
    conn.execute(
        "INSERT INTO run_parameters (id, run_id, category, name, value, value_type) VALUES (?, ?, ?, ?, ?, ?)",
        [3, 2, "transcription", "no_speech_threshold", "0.6", "float"],
    )

    # File metrics — 2 rows for run 1, 1 row for run 2.
    conn.execute(
        """
        INSERT INTO file_metrics (
            id, run_id, recorded_at, audio_path, preset, status,
            audio_duration, total_processing_time, transcribe_duration, preprocess_duration,
            preprocess_enabled, beam_size, detected_language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [1, 1, ts1, "/in/folder1/a.mp3", "turbo", "completed", 3500.0, 150.0, 148.0, 1.0, True, 5, "en"],
    )
    conn.execute(
        """
        INSERT INTO file_metrics (
            id, run_id, recorded_at, audio_path, preset, status,
            audio_duration, total_processing_time, transcribe_duration, preprocess_duration,
            preprocess_enabled, beam_size, detected_language
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [2, 1, ts1, "/in/folder1/b.mp3", "turbo", "completed", 3430.77, 151.8, 149.28, 0.42, True, 5, "en"],
    )
    conn.execute(
        """
        INSERT INTO file_metrics (
            id, run_id, recorded_at, audio_path, preset, status,
            audio_duration, total_processing_time, transcribe_duration, preprocess_duration,
            preprocess_enabled, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [3, 2, ts2, "/in/folder2/c.mp3", "et-large", "failed", 6000.0, 500.0, 498.0, 2.0, True, "transcribe failed"],
    )


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        _init_schema(conn)
        _seed(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _run_script(db_path: Path, output: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--db-path",
            str(db_path),
            "--output",
            str(output),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return result


def test_migration_produces_two_lines_with_preserved_ids(synthetic_db: Path, tmp_path: Path) -> None:
    output = tmp_path / "runs.jsonl"
    result = _run_script(synthetic_db, output)
    assert result.returncode == 0, f"script failed: {result.stderr}"

    raw = output.read_text(encoding="utf-8")
    lines = [line for line in raw.split("\n") if line]
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    assert [r["id"] for r in records] == [1, 2]
    # Every line parses as JSON (json.loads above already proves this).
    # Sanity: file ends with a single newline.
    assert raw.endswith("\n")


def test_file_metrics_embedded_in_files_array(synthetic_db: Path, tmp_path: Path) -> None:
    output = tmp_path / "runs.jsonl"
    result = _run_script(synthetic_db, output)
    assert result.returncode == 0, result.stderr

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    run1, run2 = records
    assert len(run1["files"]) == 2
    assert len(run2["files"]) == 1
    assert {f["path"] for f in run1["files"]} == {"/in/folder1/a.mp3", "/in/folder1/b.mp3"}
    assert run2["files"][0]["path"] == "/in/folder2/c.mp3"
    assert run2["files"][0]["status"] == "failed"
    assert run2["files"][0]["error_message"] == "transcribe failed"


def test_r1_missing_run_configs_does_not_null_stomp_wide_row(synthetic_db: Path, tmp_path: Path) -> None:
    """Run 2 has NO run_configs/run_metrics row. Its wide-row ``preset``,
    ``language``, and ``model_id`` must still appear in the JSONL — backfill
    from run_parameters fills in ``device`` and ``compute_type`` too."""
    output = tmp_path / "runs.jsonl"
    result = _run_script(synthetic_db, output)
    assert result.returncode == 0, result.stderr

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    run2 = records[1]
    assert run2["preset"] == "et-large"
    assert run2["language"] == "et"
    assert run2["model"]["id"] == "old-wide-model-id"
    # Backfilled via run_parameters:
    assert run2["model"]["device"] == "cpu"
    assert run2["model"]["compute_type"] == "int8"
    assert run2["params"]["no_speech_threshold"] == 0.6


def test_run_configs_wins_when_both_populated(synthetic_db: Path, tmp_path: Path) -> None:
    output = tmp_path / "runs.jsonl"
    result = _run_script(synthetic_db, output)
    assert result.returncode == 0, result.stderr

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    run1 = records[0]
    # run_configs values surface in the JSONL.
    assert run1["model"]["id"] == "Systran/faster-distil-whisper-large-v3"
    assert run1["model"]["device"] == "cuda"
    assert run1["model"]["compute_type"] == "float16"
    # run_metrics totals also propagated.
    assert run1["totals"]["preprocess_time_s"] == 1.42
    assert run1["totals"]["transcribe_time_s"] == 297.28


def test_recorded_at_normalised_to_iso_utc_with_z_suffix(synthetic_db: Path, tmp_path: Path) -> None:
    output = tmp_path / "runs.jsonl"
    result = _run_script(synthetic_db, output)
    assert result.returncode == 0, result.stderr

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    for r in records:
        assert isinstance(r["recorded_at"], str)
        assert r["recorded_at"].endswith("Z")


def test_idempotent_byte_identical_output(synthetic_db: Path, tmp_path: Path) -> None:
    out1 = tmp_path / "runs_1.jsonl"
    out2 = tmp_path / "runs_2.jsonl"
    r1 = _run_script(synthetic_db, out1)
    r2 = _run_script(synthetic_db, out2)
    assert r1.returncode == 0 and r2.returncode == 0
    assert out1.read_bytes() == out2.read_bytes()
