"""Schema records, value coercion helpers, and DDL initialization for the database package."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RunRecord:
    """Structured configuration and summary for a transcription batch."""

    recorded_at: str | datetime
    input_folder: str | None

    # Configuration (Static for all files in run)
    preset: str
    language: str | None

    # Preprocessing Config
    preprocess_enabled: bool
    preprocess_profile: str | None
    target_sample_rate: int | None
    target_channels: int | None
    loudnorm_preset: str | None = None
    # Preprocessing parameters used (from actual processing)
    volume_adjustment_db: float | None = None
    resampler: str | None = None
    sample_format: str | None = None
    loudnorm_target_i: float | None = None
    loudnorm_target_tp: float | None = None
    loudnorm_target_lra: float | None = None
    loudnorm_backend: str | None = None
    denoise_method: str | None = None
    denoise_library: str | None = None
    rnnoise_model: str | None = None
    rnnoise_mix: float | None = None
    snr_estimation_method: str | None = None

    # Model Config
    model_id: str | None = None
    device: str | None = None
    compute_type: str | None = None

    # Transcription Config
    # Note: For baseline/minimal variants, some parameters may be None because they weren't
    # explicitly passed to model.transcribe(). In those cases, faster-whisper uses its own
    # internal defaults, which we don't know. Setting these to None is more honest than
    # using TranscriptionConfig defaults that weren't actually used.
    beam_size: int | None = None
    patience: float | None = None
    word_timestamps: bool | None = None
    task: str | None = None
    chunk_length: int | None = None
    vad_filter: bool | None = None
    vad_threshold: float | None = None
    vad_min_speech_duration_ms: int | None = None
    vad_max_speech_duration_s: float | None = None
    vad_min_silence_duration_ms: int | None = None
    vad_speech_pad_ms: int | None = None
    temperature: str | None = None  # Stored as JSON string for list[float] support
    temperature_increment_on_fallback: float | None = None
    best_of: int | None = None
    compression_ratio_threshold: float | None = None
    logprob_threshold: float | None = None
    no_speech_threshold: float | None = None
    length_penalty: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    suppress_tokens: str | None = None
    condition_on_previous_text: bool | None = None
    initial_prompt: str | None = None

    # Batch Aggregates (Outcomes)
    files_found: int = 0
    succeeded: int = 0
    failed: int = 0
    total_processing_time: float | None = None
    total_preprocess_time: float | None = None
    total_transcribe_time: float | None = None
    total_audio_duration: float | None = None
    speed_ratio: float | None = None


@dataclass(slots=True)
class FileMetricRecord:
    """Detailed metrics for a single file transcription."""

    run_id: int
    recorded_at: str | datetime

    # Identifier
    audio_path: str
    preset: str
    status: str

    # Language detection
    requested_language: str | None = None
    applied_language: str | None = None
    detected_language: str | None = None
    language_probability: float | None = None

    # Timing metrics
    audio_duration: float | None = None
    total_processing_time: float = 0.0
    transcribe_duration: float = 0.0
    preprocess_duration: float = 0.0
    speed_ratio: float | None = None

    # Preprocessing
    preprocess_enabled: bool = False
    preprocess_profile: str | None = None
    target_sample_rate: int | None = None
    target_channels: int | None = None
    preprocess_snr_before: float | None = None
    preprocess_snr_after: float | None = None
    preprocess_steps: list[dict[str, Any]] | None = None
    rnnoise_model: str | None = None
    rnnoise_mix: float | None = None

    # Audio inspection
    input_channels: int | None = None
    input_sample_rate: int | None = None
    input_format: str | None = None

    # Downmix/resample parameters used
    volume_adjustment_db: float | None = None
    resampler: str | None = None
    sample_format: str | None = None

    # Loudness normalization parameters used
    loudnorm_preset: str | None = None
    loudnorm_target_i: float | None = None
    loudnorm_target_tp: float | None = None
    loudnorm_target_lra: float | None = None
    loudnorm_backend: str | None = None

    # Denoise parameters used
    denoise_method: str | None = None
    denoise_library: str | None = None

    # SNR estimation
    snr_estimation_method: str | None = None

    # Transcription parameters used
    # Note: For baseline/minimal variants, some parameters may be None because they weren't
    # explicitly passed to model.transcribe(). In those cases, faster-whisper uses its own
    # internal defaults, which we don't know. Setting these to None is more honest than
    # using TranscriptionConfig defaults that weren't actually used.
    beam_size: int | None = None
    patience: float | None = None
    word_timestamps: bool | None = None
    task: str | None = None
    chunk_length: int | None = None
    vad_filter: bool | None = None
    vad_threshold: float | None = None
    vad_min_speech_duration_ms: int | None = None
    vad_max_speech_duration_s: float | None = None
    vad_min_silence_duration_ms: int | None = None
    vad_speech_pad_ms: int | None = None
    temperature: str | None = None  # Stored as JSON string for list[float] support
    temperature_increment_on_fallback: float | None = None
    best_of: int | None = None
    compression_ratio_threshold: float | None = None
    logprob_threshold: float | None = None
    no_speech_threshold: float | None = None
    length_penalty: float | None = None
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None
    suppress_tokens: str | None = None
    condition_on_previous_text: bool | None = None
    initial_prompt: str | None = None

    # Model parameters used
    model_id: str | None = None
    device: str | None = None
    compute_type: str | None = None

    # Output parameters
    output_format: str | None = None
    float_precision: int | None = None

    # Error
    error_message: str | None = None


@dataclass(slots=True)
class Migration:
    """Represents a database schema migration."""

    version: int
    name: str
    description: str
    migrate: Callable[[duckdb.DuckDBPyConnection], None]


def format_timestamp(value: str | datetime | None) -> str:
    """Return a consistent ISO timestamp string."""
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def convert_parameter_value(value: str, value_type: str) -> Any:
    """Convert parameter value string back to original type.

    Args:
        value: String representation of the value
        value_type: Type name (e.g., 'float', 'int', 'bool', 'str')

    Returns:
        Converted value with appropriate type. Returns original string
        if conversion fails (with warning logged).
    """
    try:
        if value_type == "float":
            return float(value)
        elif value_type == "int":
            return int(value)
        elif value_type == "bool":
            # Handle boolean conversion explicitly
            value_lower = str(value).lower()
            if value_lower in ("true", "1", "yes", "on"):
                return True
            elif value_lower in ("false", "0", "no", "off", ""):
                return False
            else:
                # Invalid boolean string - log warning and default to False
                LOGGER.warning(
                    "Invalid boolean value '%s' (type: %s), defaulting to False",
                    value,
                    value_type,
                )
                return False
        elif value_type == "str":
            return value
        else:
            # Fallback: try to infer type
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
    except (ValueError, TypeError) as e:
        LOGGER.warning(
            "Failed to convert parameter value '%s' (type: %s): %s. Returning original string.",
            value,
            value_type,
            e,
        )
        return value  # Return original string as fallback


def get_default_db_path() -> Path:
    """Get XDG-compliant default database path.

    Uses XDG Base Directory specification:
    - XDG_DATA_HOME if set (e.g., ~/.local/share)
    - Falls back to ~/.local/share if not set

    Returns:
        Path to database file in XDG data directory
    """
    # Get XDG_DATA_HOME or use default
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        data_dir = Path(xdg_data_home)
    else:
        data_dir = Path.home() / ".local" / "share"

    # Create app-specific directory
    app_data_dir = data_dir / "stt-faster"
    app_data_dir.mkdir(parents=True, exist_ok=True)

    return app_data_dir / "transcribe_state.duckdb"


def create_initial_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the baseline tables for a fresh database.

    Runs on every `_init_db` call; `CREATE TABLE IF NOT EXISTS` makes it
    idempotent. Existing databases stay untouched; the migration registry
    handles upgrades.
    """
    # Legacy table for status tracking (kept for file status persistence)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcriptions (
            -- DuckDB uses SEQUENCE for auto-increment usually, but INTEGER PRIMARY KEY implies it
            id INTEGER PRIMARY KEY,
            file_path VARCHAR UNIQUE NOT NULL,
            status VARCHAR NOT NULL,
            error_message VARCHAR
        );
        CREATE SEQUENCE IF NOT EXISTS seq_transcriptions_id START 1;
        ALTER TABLE transcriptions ALTER COLUMN id SET DEFAULT nextval('seq_transcriptions_id');
    """)

    # Runs table - Stores BATCH CONFIGURATION and AGGREGATES (normalized schema)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
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
            speed_ratio DOUBLE
        );
        CREATE SEQUENCE IF NOT EXISTS seq_runs_id START 1;
    """)
    # Set default only if table was just created (no foreign key dependencies yet)
    # If normalized tables exist, skip ALTER to avoid foreign key constraint issues
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('run_configs', 'run_metrics')"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}
        if len(existing_tables) < 2:
            conn.execute("ALTER TABLE runs ALTER COLUMN id SET DEFAULT nextval('seq_runs_id')")
    except Exception as exc:
        LOGGER.debug(
            "_init_db runs.id sequence default skipped (probe or ALTER failed; sequence still works): %s",
            exc,
        )

    # Run configs table - Stores configuration parameters
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_configs (
            run_id INTEGER PRIMARY KEY REFERENCES runs(id),
            model_id VARCHAR,
            device VARCHAR,
            compute_type VARCHAR
        );
    """)

    # Run metrics table - Stores timing and performance metrics
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_metrics (
            run_id INTEGER PRIMARY KEY REFERENCES runs(id),
            total_preprocess_time DOUBLE,
            total_transcribe_time DOUBLE,
            additional_metrics JSON
        );
    """)

    # Run parameters table - Flexible parameter storage (replaces JSON configs)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_parameters (
            id INTEGER PRIMARY KEY,
            run_id INTEGER REFERENCES runs(id),
            category VARCHAR NOT NULL,
            name VARCHAR NOT NULL,
            value VARCHAR,
            value_type VARCHAR
        );
        CREATE SEQUENCE IF NOT EXISTS seq_run_parameters_id START 1;
        ALTER TABLE run_parameters ALTER COLUMN id SET DEFAULT nextval('seq_run_parameters_id');
    """)

    # File Metrics table - Stores PER-FILE OUTCOMES
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
        );
        CREATE SEQUENCE IF NOT EXISTS seq_file_metrics_id START 1;
        ALTER TABLE file_metrics ALTER COLUMN id SET DEFAULT nextval('seq_file_metrics_id');
    """)

    # Schema version table for tracking migrations
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
