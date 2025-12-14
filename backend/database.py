"""Database operations for transcription state tracking."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb

from backend.exceptions import DatabaseError

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


def _format_timestamp(value: str | datetime | None) -> str:
    """Return a consistent ISO timestamp string."""
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    return value


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


@dataclass(slots=True)
class Migration:
    """Represents a database schema migration."""

    version: int
    name: str
    description: str
    migrate: Callable[[duckdb.DuckDBPyConnection], None]


# Migration Functions
def _get_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    """Get existing column names for a table."""
    try:
        columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        return {row[0] for row in columns_result}
    except Exception:
        # Table doesn't exist yet
        return set()


def _migration_001_add_rnnoise_columns(conn: duckdb.DuckDBPyConnection) -> None:
    """Add rnnoise_model and rnnoise_mix columns to file_metrics."""
    existing_columns = _get_columns(conn, "file_metrics")
    if "rnnoise_model" not in existing_columns:
        LOGGER.debug("Migrating schema: adding rnnoise_model column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_model VARCHAR")
    if "rnnoise_mix" not in existing_columns:
        LOGGER.debug("Migrating schema: adding rnnoise_mix column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_mix DOUBLE")


def _migration_002_add_file_metrics_columns(conn: duckdb.DuckDBPyConnection) -> None:
    """Add loudnorm_target_lra and snr_estimation_method columns to file_metrics."""
    existing_columns = _get_columns(conn, "file_metrics")
    if "loudnorm_target_lra" not in existing_columns:
        LOGGER.debug("Migrating schema: adding loudnorm_target_lra column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN loudnorm_target_lra DOUBLE")
    if "snr_estimation_method" not in existing_columns:
        LOGGER.debug("Migrating schema: adding snr_estimation_method column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN snr_estimation_method VARCHAR")


def _migration_003_add_transcription_params_to_file_metrics(conn: duckdb.DuckDBPyConnection) -> None:
    """Add transcription parameter columns to file_metrics table."""
    existing_columns = _get_columns(conn, "file_metrics")
    transcription_columns = {
        "patience": "DOUBLE",
        "task": "VARCHAR",
        "chunk_length": "INTEGER",
        "vad_filter": "BOOLEAN",
        "vad_threshold": "DOUBLE",
        "vad_min_speech_duration_ms": "INTEGER",
        "vad_max_speech_duration_s": "DOUBLE",
        "vad_min_silence_duration_ms": "INTEGER",
        "vad_speech_pad_ms": "INTEGER",
        "temperature": "VARCHAR",
        "temperature_increment_on_fallback": "DOUBLE",
        "best_of": "INTEGER",
        "compression_ratio_threshold": "DOUBLE",
        "logprob_threshold": "DOUBLE",
        "no_speech_threshold": "DOUBLE",
        "length_penalty": "DOUBLE",
        "repetition_penalty": "DOUBLE",
        "no_repeat_ngram_size": "INTEGER",
        "suppress_tokens": "VARCHAR",
        "condition_on_previous_text": "BOOLEAN",
        "initial_prompt": "VARCHAR",
    }

    for column_name, column_type in transcription_columns.items():
        if column_name not in existing_columns:
            LOGGER.debug("Migrating schema: adding %s column to file_metrics", column_name)
            if column_type == "DOUBLE":
                conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} DOUBLE")
            elif column_type == "VARCHAR":
                conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} VARCHAR")
            elif column_type == "INTEGER":
                conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} INTEGER")
            elif column_type == "BOOLEAN":
                conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} BOOLEAN")
            else:
                LOGGER.warning("Unknown column type %s for %s, skipping migration", column_type, column_name)


def _migration_004_add_transcription_params_to_runs(conn: duckdb.DuckDBPyConnection) -> None:
    """Add transcription parameter columns to runs table."""
    # Skip if normalized tables exist (schema already migrated to normalized form)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('run_configs', 'run_metrics')"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}
        if len(existing_tables) >= 2:
            LOGGER.debug("Normalized tables exist, skipping migration 4 (transcription params in run_configs)")
            return
    except Exception:
        pass  # nosec B110 - safe pass for migration check

    existing_columns = _get_columns(conn, "runs")
    transcription_columns = {
        "patience": "DOUBLE",
        "task": "VARCHAR",
        "chunk_length": "INTEGER",
        "vad_filter": "BOOLEAN",
        "vad_threshold": "DOUBLE",
        "vad_min_speech_duration_ms": "INTEGER",
        "vad_max_speech_duration_s": "DOUBLE",
        "vad_min_silence_duration_ms": "INTEGER",
        "vad_speech_pad_ms": "INTEGER",
        "temperature": "VARCHAR",
        "temperature_increment_on_fallback": "DOUBLE",
        "best_of": "INTEGER",
        "compression_ratio_threshold": "DOUBLE",
        "logprob_threshold": "DOUBLE",
        "no_speech_threshold": "DOUBLE",
        "length_penalty": "DOUBLE",
        "repetition_penalty": "DOUBLE",
        "no_repeat_ngram_size": "INTEGER",
        "suppress_tokens": "VARCHAR",
        "condition_on_previous_text": "BOOLEAN",
        "initial_prompt": "VARCHAR",
    }

    for column_name, column_type in transcription_columns.items():
        if column_name not in existing_columns:
            LOGGER.debug("Migrating schema: adding %s column to runs", column_name)
            if column_type == "DOUBLE":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} DOUBLE")
            elif column_type == "VARCHAR":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} VARCHAR")
            elif column_type == "INTEGER":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} INTEGER")
            elif column_type == "BOOLEAN":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} BOOLEAN")
            else:
                LOGGER.warning("Unknown column type %s for %s, skipping", column_type, column_name)


def _migration_005_add_preprocessing_params_to_runs(conn: duckdb.DuckDBPyConnection) -> None:
    """Add preprocessing parameter columns to runs table."""
    # Skip if normalized tables exist (schema already migrated to normalized form)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('run_configs', 'run_metrics')"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}
        if len(existing_tables) >= 2:
            LOGGER.debug("Normalized tables exist, skipping migration 5 (preprocessing params in run_configs)")
            return
    except Exception:
        pass  # nosec B110 - safe pass for migration check

    existing_columns = _get_columns(conn, "runs")
    preprocessing_columns = {
        "volume_adjustment_db": "DOUBLE",
        "resampler": "VARCHAR",
        "sample_format": "VARCHAR",
        "loudnorm_target_i": "DOUBLE",
        "loudnorm_target_tp": "DOUBLE",
        "loudnorm_target_lra": "DOUBLE",
        "loudnorm_backend": "VARCHAR",
        "denoise_method": "VARCHAR",
        "denoise_library": "VARCHAR",
        "rnnoise_model": "VARCHAR",
        "rnnoise_mix": "DOUBLE",
        "snr_estimation_method": "VARCHAR",
    }

    for column_name, column_type in preprocessing_columns.items():
        if column_name not in existing_columns:
            LOGGER.debug("Migrating schema: adding %s column to runs", column_name)
            if column_type == "DOUBLE":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} DOUBLE")
            elif column_type == "VARCHAR":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} VARCHAR")
            elif column_type == "INTEGER":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} INTEGER")
            elif column_type == "BOOLEAN":
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} BOOLEAN")
            else:
                LOGGER.warning("Unknown column type %s for %s, skipping", column_type, column_name)


def _migration_006_normalize_runs_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Normalize runs table schema for Alternative 3 - split into multiple tables for flexibility."""
    import json

    # First check if normalized tables already exist (for new databases created with new schema)
    # If they exist, this migration should skip entirely to avoid foreign key constraint issues
    # The wide columns in runs from migrations 4-5 are harmless if normalized tables handle the data
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('run_configs', 'run_metrics', 'run_parameters')"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}
        if len(existing_tables) >= 2:  # At least run_configs and run_metrics exist
            LOGGER.debug("Normalized tables already exist, skipping migration 6 (schema already normalized)")
            return
    except Exception:
        pass  # nosec B110 - safe pass for migration check  # Continue to check old columns

    # Check if migration is needed by looking for old wide columns
    try:
        cursor = conn.execute("DESCRIBE runs")
        columns = {row[0] for row in cursor.fetchall()}

        # Check for old wide columns that should be removed
        old_wide_columns = {
            "volume_adjustment_db",
            "resampler",
            "sample_format",
            "loudnorm_target_i",
            "loudnorm_target_tp",
            "loudnorm_target_lra",
            "loudnorm_backend",
            "denoise_method",
            "denoise_library",
            "rnnoise_model",
            "rnnoise_mix",
            "snr_estimation_method",
            "beam_size",
            "patience",
            "word_timestamps",
            "task",
            "chunk_length",
            "vad_filter",
            "vad_threshold",
            "vad_min_speech_duration_ms",
            "vad_max_speech_duration_s",
            "vad_min_silence_duration_ms",
            "vad_speech_pad_ms",
            "temperature",
            "temperature_increment_on_fallback",
            "best_of",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "length_penalty",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "suppress_tokens",
            "condition_on_previous_text",
            "initial_prompt",
        }

        has_old_columns = bool(old_wide_columns & columns)

        if not has_old_columns:
            LOGGER.debug("Runs table already normalized (no old wide columns found), skipping migration")
            return
        else:
            LOGGER.info(
                "Detected old wide runs schema with %d old columns, proceeding with migration",
                len(old_wide_columns & columns),
            )

    except Exception as e:
        LOGGER.info("Could not check migration status (%s), proceeding with migration", e)

    LOGGER.info("Starting runs table normalization migration")

    # Wrap migration in transaction for atomicity
    transaction_started = False
    try:
        conn.execute("BEGIN TRANSACTION")
        transaction_started = True
    except Exception:
        # DuckDB may not support explicit transactions for DDL, continue anyway
        pass  # nosec B110 - safe pass for transaction initialization

    try:
        # Check if normalized tables already exist (from _init_db for new databases)
        # This is a second check in case the first one didn't catch it
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('run_configs', 'run_metrics', 'run_parameters')"
            )
            existing_normalized: set[str] = {row[0] for row in cursor.fetchall()}
            if len(existing_normalized) >= 2:
                LOGGER.debug("Normalized tables found in second check, aborting migration")
                # Clean up any intermediate tables
                try:
                    conn.execute("DROP TABLE IF EXISTS runs_new")
                except Exception:
                    pass  # nosec B110 - safe pass for cleanup
                if transaction_started:
                    try:
                        conn.execute("COMMIT")
                    except Exception:
                        pass  # nosec B110 - safe pass for transaction commit
                return
        except Exception:
            existing_normalized = set()

        # Create new normalized tables (only if they don't exist)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS runs_new (
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
        """)

        # Only create normalized tables if they don't exist
        if "run_configs" not in existing_normalized:
            conn.execute("""
                CREATE TABLE run_configs (
                    run_id INTEGER PRIMARY KEY REFERENCES runs_new(id),
                    model_id VARCHAR,
                    device VARCHAR,
                    compute_type VARCHAR
                );
            """)

        if "run_metrics" not in existing_normalized:
            conn.execute("""
                CREATE TABLE run_metrics (
                    run_id INTEGER PRIMARY KEY REFERENCES runs_new(id),
                    total_preprocess_time DOUBLE,
                    total_transcribe_time DOUBLE,
                    additional_metrics JSON
                );
            """)

        if "run_parameters" not in existing_normalized:
            conn.execute("""
                CREATE TABLE run_parameters (
                    id INTEGER PRIMARY KEY,
                    run_id INTEGER REFERENCES runs_new(id),
                    category VARCHAR,
                    name VARCHAR,
                    value VARCHAR,
                    value_type VARCHAR
                );
            """)

        # Migrate existing data
        LOGGER.debug("Migrating existing runs data to normalized schema")

        # Check if there's any data to migrate
        try:
            count_result = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            has_data = count_result and count_result[0] > 0
        except Exception:
            has_data = False

        if not has_data and existing_normalized:
            LOGGER.debug("No runs data to migrate and normalized tables already exist, skipping migration")
            # Drop intermediate tables if they exist
            try:
                conn.execute("DROP TABLE IF EXISTS runs_new")
            except Exception:
                pass  # nosec B110 - safe pass for cleanup
            try:
                conn.execute("DROP TABLE IF EXISTS file_metrics_new")
            except Exception:
                pass  # nosec B110 - safe pass for cleanup
            if transaction_started:
                try:
                    conn.execute("COMMIT")
                except Exception:
                    pass  # nosec B110 - safe pass for transaction commit
            return

        # Get all existing runs
        cursor = conn.execute("SELECT * FROM runs ORDER BY id")
        existing_runs = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        for row in existing_runs:
            run_data = dict(zip(column_names, row))
            run_id = run_data["id"]

            # Insert core run data
            conn.execute(
                """
            INSERT INTO runs_new (
                id, recorded_at, input_folder, preset, language, preprocess_enabled,
                files_found, succeeded, failed, total_processing_time,
                total_audio_duration, speed_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
                (
                    run_id,
                    run_data["recorded_at"],
                    run_data["input_folder"],
                    run_data["preset"],
                    run_data["language"],
                    run_data["preprocess_enabled"],
                    run_data["files_found"],
                    run_data["succeeded"],
                    run_data["failed"],
                    run_data["total_processing_time"],
                    run_data["total_audio_duration"],
                    run_data["speed_ratio"],
                ),
            )

            # Extract and insert config data
            preprocess_config = {
                "profile": run_data.get("preprocess_profile"),
                "target_sample_rate": run_data.get("target_sample_rate"),
                "target_channels": run_data.get("target_channels"),
                "loudnorm_preset": run_data.get("loudnorm_preset"),
                "volume_adjustment_db": run_data.get("volume_adjustment_db"),
                "resampler": run_data.get("resampler"),
                "sample_format": run_data.get("sample_format"),
                "loudnorm_target_i": run_data.get("loudnorm_target_i"),
                "loudnorm_target_tp": run_data.get("loudnorm_target_tp"),
                "loudnorm_target_lra": run_data.get("loudnorm_target_lra"),
                "loudnorm_backend": run_data.get("loudnorm_backend"),
                "denoise_method": run_data.get("denoise_method"),
                "denoise_library": run_data.get("denoise_library"),
                "rnnoise_model": run_data.get("rnnoise_model"),
                "rnnoise_mix": run_data.get("rnnoise_mix"),
                "snr_estimation_method": run_data.get("snr_estimation_method"),
            }

            transcription_config = {
                "beam_size": run_data.get("beam_size"),
                "patience": run_data.get("patience"),
                "word_timestamps": run_data.get("word_timestamps"),
                "task": run_data.get("task"),
                "chunk_length": run_data.get("chunk_length"),
                "vad_filter": run_data.get("vad_filter"),
                "vad_threshold": run_data.get("vad_threshold"),
                "vad_min_speech_duration_ms": run_data.get("vad_min_speech_duration_ms"),
                "vad_max_speech_duration_s": run_data.get("vad_max_speech_duration_s"),
                "vad_min_silence_duration_ms": run_data.get("vad_min_silence_duration_ms"),
                "vad_speech_pad_ms": run_data.get("vad_speech_pad_ms"),
                "temperature": run_data.get("temperature"),
                "temperature_increment_on_fallback": run_data.get("temperature_increment_on_fallback"),
                "best_of": run_data.get("best_of"),
                "compression_ratio_threshold": run_data.get("compression_ratio_threshold"),
                "logprob_threshold": run_data.get("logprob_threshold"),
                "no_speech_threshold": run_data.get("no_speech_threshold"),
                "length_penalty": run_data.get("length_penalty"),
                "repetition_penalty": run_data.get("repetition_penalty"),
                "no_repeat_ngram_size": run_data.get("no_repeat_ngram_size"),
                "suppress_tokens": run_data.get("suppress_tokens"),
                "condition_on_previous_text": run_data.get("condition_on_previous_text"),
                "initial_prompt": run_data.get("initial_prompt"),
            }

            conn.execute(
                """
                INSERT INTO run_configs (
                    run_id, model_id, device, compute_type
                ) VALUES (?, ?, ?, ?)
            """,
                (
                    run_id,
                    run_data.get("model_id"),
                    run_data.get("device"),
                    run_data.get("compute_type"),
                ),
            )

            # Insert metrics
            conn.execute(
                """
                INSERT INTO run_metrics (
                    run_id, total_preprocess_time, total_transcribe_time, additional_metrics
                ) VALUES (?, ?, ?, ?)
            """,
                (
                    run_id,
                    run_data.get("total_preprocess_time"),
                    run_data.get("total_transcribe_time"),
                    json.dumps({}, sort_keys=True),  # Empty for now, extensible
                ),
            )

            # Insert individual parameters for flexibility
            params_to_insert: list[tuple[int, str, str, str, str]] = []

            # Preprocessing parameters
            for key, value in preprocess_config.items():
                if value is not None:
                    params_to_insert.append((run_id, "preprocess", key, str(value), type(value).__name__))

            # Transcription parameters
            for key, value in transcription_config.items():
                if value is not None:
                    params_to_insert.append((run_id, "transcription", key, str(value), type(value).__name__))

            # Model parameters
            if run_data.get("model_id"):
                params_to_insert.append((run_id, "model", "model_id", run_data["model_id"], "str"))
            if run_data.get("device"):
                params_to_insert.append((run_id, "model", "device", run_data["device"], "str"))
            if run_data.get("compute_type"):
                params_to_insert.append((run_id, "model", "compute_type", run_data["compute_type"], "str"))

            for param_data in params_to_insert:
                conn.execute(
                    """
                    INSERT INTO run_parameters (run_id, category, name, value, value_type)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    param_data,
                )

        # Since we can't rename tables with foreign key dependencies in DuckDB,
        # we need to drop dependent tables first, then recreate everything

        # First, drop all dependent tables that reference runs
        # This allows us to drop and recreate runs table
        try:
            conn.execute("DROP TABLE IF EXISTS run_parameters")
        except Exception:
            pass  # nosec B110 - safe pass for migration cleanup
        try:
            conn.execute("DROP TABLE IF EXISTS run_metrics")
        except Exception:
            pass  # nosec B110 - safe pass for migration cleanup
        try:
            conn.execute("DROP TABLE IF EXISTS run_configs")
        except Exception:
            pass  # nosec B110 - safe pass for migration cleanup
        # Note: We don't touch file_metrics - it doesn't have a foreign key constraint
        # and its structure doesn't change in this migration

        # Now we can safely drop runs table
        try:
            conn.execute("DROP TABLE IF EXISTS runs")
        except Exception as e:
            LOGGER.warning("Could not drop old runs table: %s", e)

        # Check if runs table already exists before trying to rename
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
            runs_exists = cursor.fetchone() is not None
            if runs_exists:
                # Check if runs is already normalized
                cursor = conn.execute("DESCRIBE runs")
                columns = {row[0] for row in cursor.fetchall()}
                if "volume_adjustment_db" not in columns:
                    # Runs table is already normalized, no need to rename
                    LOGGER.debug("Runs table already exists and is normalized, dropping runs_new")
                    try:
                        conn.execute("DROP TABLE IF EXISTS runs_new")
                    except Exception:
                        pass  # nosec B110 - safe pass for cleanup
                else:
                    # Runs exists but is not normalized - we can't drop it due to foreign keys
                    # This shouldn't happen if the early checks worked, but handle it gracefully
                    LOGGER.warning("Runs table exists with old schema but can't be dropped due to dependencies")
                    try:
                        conn.execute("DROP TABLE IF EXISTS runs_new")
                    except Exception:
                        pass  # nosec B110 - safe pass for cleanup
            else:
                # Runs doesn't exist, safe to rename
                try:
                    conn.execute("ALTER TABLE runs_new RENAME TO runs")
                except Exception as e:
                    LOGGER.warning("Could not rename runs_new to runs: %s", e)
        except Exception as e:
            LOGGER.warning("Error checking runs table status: %s", e)

        # Recreate normalized tables that reference runs
        # (They were dropped earlier to allow runs table to be dropped)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_configs (
                run_id INTEGER PRIMARY KEY REFERENCES runs(id),
                model_id VARCHAR,
                device VARCHAR,
                compute_type VARCHAR
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_metrics (
                run_id INTEGER PRIMARY KEY REFERENCES runs(id),
                total_preprocess_time DOUBLE,
                total_transcribe_time DOUBLE,
                additional_metrics JSON
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_parameters (
                id INTEGER PRIMARY KEY,
                run_id INTEGER REFERENCES runs(id),
                category VARCHAR,
                name VARCHAR,
                value VARCHAR,
                value_type VARCHAR
            );
        """)

        # Update sequences (skip ALTER TABLE if normalized tables exist to avoid foreign key issues)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('run_configs', 'run_metrics')"
            )
            existing_tables = {row[0] for row in cursor.fetchall()}
            has_normalized = len(existing_tables) >= 2
        except Exception:
            has_normalized = False

        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_runs_id START 1")
        if not has_normalized:
            try:
                conn.execute("ALTER TABLE runs ALTER COLUMN id SET DEFAULT nextval('seq_runs_id')")
            except Exception:
                pass  # nosec B110 - Skip if table has foreign key dependencies
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_run_parameters_id START 1")
        conn.execute("ALTER TABLE run_parameters ALTER COLUMN id SET DEFAULT nextval('seq_run_parameters_id')")

        # Commit transaction if we started one
        if transaction_started:
            try:
                conn.execute("COMMIT")
            except Exception:
                # DuckDB may auto-commit, that's fine
                pass  # nosec B110 - safe pass for transaction commit

        LOGGER.info("Completed runs table normalization migration")
    except Exception:
        # Rollback on error if transaction was started
        if transaction_started:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                # DuckDB may not support rollback for DDL, log and continue
                pass  # nosec B110 - safe pass for transaction rollback
        raise


def _validate_migration_ordering() -> None:
    """Validate that migrations are properly ordered and unique."""
    versions = [m.version for m in MIGRATIONS]
    if versions != sorted(versions):
        raise ValueError("Migrations must be ordered by version number")
    if len(versions) != len(set(versions)):
        raise ValueError("Migration versions must be unique")
    if versions and versions[0] != 1:
        raise ValueError("Migrations must start with version 1")


# Migration Registry
MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        name="add_rnnoise_columns",
        description="Add rnnoise_model and rnnoise_mix columns to file_metrics table",
        migrate=_migration_001_add_rnnoise_columns,
    ),
    Migration(
        version=2,
        name="add_file_metrics_columns",
        description="Add loudnorm_target_lra and snr_estimation_method columns to file_metrics table",
        migrate=_migration_002_add_file_metrics_columns,
    ),
    Migration(
        version=3,
        name="add_transcription_params_file_metrics",
        description="Add transcription parameter columns to file_metrics table",
        migrate=_migration_003_add_transcription_params_to_file_metrics,
    ),
    Migration(
        version=4,
        name="add_transcription_params_runs",
        description="Add transcription parameter columns to runs table",
        migrate=_migration_004_add_transcription_params_to_runs,
    ),
    Migration(
        version=5,
        name="add_preprocessing_params_runs",
        description="Add preprocessing parameter columns to runs table",
        migrate=_migration_005_add_preprocessing_params_to_runs,
    ),
    Migration(
        version=6,
        name="normalize_runs_schema",
        description="Normalize runs table schema with separate config, metrics, and parameters tables",
        migrate=_migration_006_normalize_runs_schema,
    ),
]


class TranscriptionDatabase:
    """Simple SQLite database for tracking transcription state."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
                     If None, uses XDG-compliant default path.
                     For backward compatibility, can still pass custom path.
        """
        if db_path is None:
            self.db_path = str(get_default_db_path())
        else:
            self.db_path = str(db_path)

        self.conn: duckdb.DuckDBPyConnection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema if it doesn't exist."""
        try:
            self.conn = duckdb.connect(self.db_path)

            # DuckDB doesn't use row_factory like sqlite3, we'll handle dict conversion manually

            # Legacy table for status tracking (kept for file status persistence)
            self.conn.execute("""
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
            self.conn.execute("""
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
                cursor = self.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('run_configs', 'run_metrics')"
                )
                existing_tables = {row[0] for row in cursor.fetchall()}
                if len(existing_tables) < 2:
                    self.conn.execute("ALTER TABLE runs ALTER COLUMN id SET DEFAULT nextval('seq_runs_id')")
            except Exception:
                # If we can't check or alter, continue - sequence will work anyway
                pass  # nosec B110 - safe pass for initialization

            # Run configs table - Stores configuration parameters
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS run_configs (
                    run_id INTEGER PRIMARY KEY REFERENCES runs(id),
                    model_id VARCHAR,
                    device VARCHAR,
                    compute_type VARCHAR
                );
            """)

            # Run metrics table - Stores timing and performance metrics
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_id INTEGER PRIMARY KEY REFERENCES runs(id),
                    total_preprocess_time DOUBLE,
                    total_transcribe_time DOUBLE,
                    additional_metrics JSON
                );
            """)

            # Run parameters table - Flexible parameter storage (replaces JSON configs)
            self.conn.execute("""
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
            self.conn.execute("""
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
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Migrate existing tables: add missing columns if they don't exist
            self._migrate_schema()

            LOGGER.debug("DuckDB initialized at %s", self.db_path)
        except Exception as e:
            msg = f"Failed to initialize database at {self.db_path}: {e}"
            raise DatabaseError(msg) from e

    def _migrate_schema(self) -> None:
        """Migrate existing database schema using version-based ordered migrations."""
        if self.conn is None:
            return

        try:
            # Validate migration registry
            _validate_migration_ordering()

            # Get current schema version
            current_version = self._get_current_schema_version()

            # Find pending migrations
            pending = [m for m in MIGRATIONS if m.version > current_version]

            if not pending:
                LOGGER.debug("Database schema is up to date (version %d)", current_version)
                return

            LOGGER.info("Running %d pending migration(s)", len(pending))

            # Apply each pending migration
            for migration in pending:
                try:
                    LOGGER.info(
                        "Applying migration %d: %s - %s", migration.version, migration.name, migration.description
                    )
                    migration.migrate(self.conn)
                    self._record_migration(migration.version)
                    self.conn.commit()
                except Exception as e:
                    LOGGER.error("Migration %d failed: %s", migration.version, e)
                    raise DatabaseError(f"Migration {migration.version} failed") from e

        except Exception as e:
            # Log but don't fail - migration errors shouldn't break initialization
            LOGGER.warning("Schema migration encountered an issue: %s", e)

    def _get_current_schema_version(self) -> int:
        """Get current schema version from database."""
        if self.conn is None:
            return 0

        try:
            result = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            return result[0] if result is not None and result[0] is not None else 0
        except Exception:
            # Table doesn't exist yet, no migrations applied
            return 0

    def _record_migration(self, version: int) -> None:
        """Record that a migration was applied."""
        if self.conn is None:
            raise DatabaseError("Database connection not available")

        self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))

    def get_migration_history(self) -> list[tuple[Any, ...]]:
        """Get migration history for debugging/auditing."""
        if self.conn is None:
            return []

        try:
            return self.conn.execute("""
                SELECT version, applied_at
                FROM schema_version
                ORDER BY version
            """).fetchall()
        except Exception:
            # Table doesn't exist yet
            return []

    def add_file(self, file_path: str, status: str = "pending") -> None:
        """Add a file to track or update its status if it exists.

        Args:
            file_path: Path to the audio file
            status: Initial status (default: 'pending')
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            self.conn.execute(
                "INSERT INTO transcriptions (file_path, status) VALUES (?, ?)",
                (file_path, status),
            )
            self.conn.commit()
            LOGGER.debug("Added file: %s with status: %s", file_path, status)
        except duckdb.ConstraintException:  # DuckDB equivalent of IntegrityError for UNIQUE constraint
            LOGGER.debug("File already exists in database: %s", file_path)
        except Exception as e:
            msg = f"Failed to add file {file_path}: {e}"
            raise DatabaseError(msg) from e

    def update_status(
        self,
        file_path: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update the status of a file.

        Args:
            file_path: Path to the audio file
            status: New status ('pending', 'completed', 'failed')
            error_message: Optional error message if status is 'failed'
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            self.conn.execute(
                "UPDATE transcriptions SET status = ?, error_message = ? WHERE file_path = ?",
                (status, error_message, file_path),
            )
            self.conn.commit()
            LOGGER.debug("Updated %s to status: %s", file_path, status)
        except Exception as e:
            msg = f"Failed to update status for {file_path}: {e}"
            raise DatabaseError(msg) from e

    def get_status(self, file_path: str) -> dict[str, Any] | None:
        """Get the status of a specific file.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with file info or None if not found
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            # DuckDB execute returns self, fetchone returns tuple
            # We need column names to make a dict
            cursor = self.conn.execute(
                "SELECT * FROM transcriptions WHERE file_path = ?",
                [file_path],
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Map tuple to dict using column descriptions
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        except Exception as e:
            msg = f"Failed to get status for {file_path}: {e}"
            raise DatabaseError(msg) from e

    def get_files_by_status(self, status: str) -> list[dict[str, Any]]:
        """Get all files with a specific status.

        Args:
            status: Status to filter by ('pending', 'completed', 'failed')

        Returns:
            List of dictionaries with file info
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.execute(
                "SELECT * FROM transcriptions WHERE status = ?",
                [status],
            )
            rows = cursor.fetchall()

            if not rows:
                return []

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            msg = f"Failed to get files by status {status}: {e}"
            raise DatabaseError(msg) from e

    def get_all_files(self) -> list[dict[str, Any]]:
        """Get all files in the database.

        Returns:
            List of dictionaries with file info
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.execute("SELECT * FROM transcriptions")
            rows = cursor.fetchall()

            if not rows:
                return []

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            msg = f"Failed to get all files: {e}"
            raise DatabaseError(msg) from e

    def record_file_metric(self, record: FileMetricRecord) -> None:
        """Persist detailed metrics for a single file."""
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        preprocess_steps_json = json.dumps(record.preprocess_steps or [], sort_keys=True)
        recorded_at = _format_timestamp(record.recorded_at)

        try:
            self.conn.execute(
                """
                INSERT INTO file_metrics (
                    run_id, recorded_at, audio_path, preset, status,
                    requested_language, applied_language, detected_language, language_probability,
                    audio_duration, total_processing_time, transcribe_duration, preprocess_duration, speed_ratio,
                    preprocess_enabled, preprocess_profile, target_sample_rate, target_channels,
                    preprocess_snr_before, preprocess_snr_after, rnnoise_model, rnnoise_mix,
                    input_channels, input_sample_rate, input_format,
                    volume_adjustment_db, resampler, sample_format,
                    loudnorm_preset, loudnorm_target_i, loudnorm_target_tp, loudnorm_target_lra, loudnorm_backend,
                    denoise_method, denoise_library,
                    snr_estimation_method,
                    beam_size, patience, word_timestamps, task, chunk_length,
                    vad_filter, vad_threshold, vad_min_speech_duration_ms, vad_max_speech_duration_s,
                    vad_min_silence_duration_ms, vad_speech_pad_ms,
                    temperature, temperature_increment_on_fallback, best_of,
                    compression_ratio_threshold, logprob_threshold, no_speech_threshold,
                    length_penalty, repetition_penalty, no_repeat_ngram_size,
                    suppress_tokens, condition_on_previous_text, initial_prompt,
                    model_id, device, compute_type,
                    output_format, float_precision,
                    preprocess_steps_json, error_message
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?,
                    ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?
                )
                """,
                (
                    record.run_id,
                    recorded_at,
                    record.audio_path,
                    record.preset,
                    record.status,
                    record.requested_language,
                    record.applied_language,
                    record.detected_language,
                    record.language_probability,
                    record.audio_duration,
                    record.total_processing_time,
                    record.transcribe_duration,
                    record.preprocess_duration,
                    record.speed_ratio,
                    int(record.preprocess_enabled),
                    record.preprocess_profile,
                    record.target_sample_rate,
                    record.target_channels,
                    record.preprocess_snr_before,
                    record.preprocess_snr_after,
                    record.rnnoise_model,
                    record.rnnoise_mix,
                    record.input_channels,
                    record.input_sample_rate,
                    record.input_format,
                    record.volume_adjustment_db,
                    record.resampler,
                    record.sample_format,
                    record.loudnorm_preset,
                    record.loudnorm_target_i,
                    record.loudnorm_target_tp,
                    record.loudnorm_target_lra,
                    record.loudnorm_backend,
                    record.denoise_method,
                    record.denoise_library,
                    record.snr_estimation_method,
                    record.beam_size,
                    record.patience,
                    int(record.word_timestamps) if record.word_timestamps is not None else None,
                    record.task,
                    record.chunk_length,
                    int(record.vad_filter) if record.vad_filter is not None else None,
                    record.vad_threshold,
                    record.vad_min_speech_duration_ms,
                    record.vad_max_speech_duration_s,
                    record.vad_min_silence_duration_ms,
                    record.vad_speech_pad_ms,
                    record.temperature,
                    record.temperature_increment_on_fallback,
                    record.best_of,
                    record.compression_ratio_threshold,
                    record.logprob_threshold,
                    record.no_speech_threshold,
                    record.length_penalty,
                    record.repetition_penalty,
                    record.no_repeat_ngram_size,
                    record.suppress_tokens,
                    int(record.condition_on_previous_text) if record.condition_on_previous_text is not None else None,
                    record.initial_prompt,
                    record.model_id,
                    record.device,
                    record.compute_type,
                    record.output_format,
                    record.float_precision,
                    preprocess_steps_json,
                    record.error_message,
                ),
            )
            self.conn.commit()
            LOGGER.debug("Recorded file metric for %s", record.audio_path)
        except Exception as e:
            msg = f"Failed to record file metrics for {record.audio_path}: {e}"
            raise DatabaseError(msg) from e

    def record_run(self, record: RunRecord) -> int:
        """Persist a new run record and return its ID using normalized schema."""
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        recorded_at = _format_timestamp(record.recorded_at)

        # Begin transaction for atomicity
        transaction_started = False
        try:
            self.conn.execute("BEGIN TRANSACTION")
            transaction_started = True
        except Exception:
            # DuckDB may auto-commit, continue anyway
            pass  # nosec B110 - safe pass for transaction initialization

        try:
            # Insert core run data
            cursor = self.conn.execute(
                """
                INSERT INTO runs (
                    recorded_at, input_folder, preset, language, preprocess_enabled,
                    files_found, succeeded, failed, total_processing_time,
                    total_audio_duration, speed_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id;
                """,
                (
                    recorded_at,
                    record.input_folder,
                    record.preset,
                    record.language,
                    int(record.preprocess_enabled),
                    record.files_found,
                    record.succeeded,
                    record.failed,
                    record.total_processing_time,
                    record.total_audio_duration,
                    record.speed_ratio,
                ),
            )

            # Fetch the returned ID
            row = cursor.fetchone()
            if not row:
                raise DatabaseError("Failed to retrieve inserted run ID")
            run_id = row[0]

            # Prepare config data
            preprocess_config = {
                "profile": record.preprocess_profile,
                "target_sample_rate": record.target_sample_rate,
                "target_channels": record.target_channels,
                "loudnorm_preset": record.loudnorm_preset,
                "volume_adjustment_db": record.volume_adjustment_db,
                "resampler": record.resampler,
                "sample_format": record.sample_format,
                "loudnorm_target_i": record.loudnorm_target_i,
                "loudnorm_target_tp": record.loudnorm_target_tp,
                "loudnorm_target_lra": record.loudnorm_target_lra,
                "loudnorm_backend": record.loudnorm_backend,
                "denoise_method": record.denoise_method,
                "denoise_library": record.denoise_library,
                "rnnoise_model": record.rnnoise_model,
                "rnnoise_mix": record.rnnoise_mix,
                "snr_estimation_method": record.snr_estimation_method,
            }

            transcription_config = {
                "beam_size": record.beam_size,
                "patience": record.patience,
                "word_timestamps": record.word_timestamps,
                "task": record.task,
                "chunk_length": record.chunk_length,
                "vad_filter": record.vad_filter,
                "vad_threshold": record.vad_threshold,
                "vad_min_speech_duration_ms": record.vad_min_speech_duration_ms,
                "vad_max_speech_duration_s": record.vad_max_speech_duration_s,
                "vad_min_silence_duration_ms": record.vad_min_silence_duration_ms,
                "vad_speech_pad_ms": record.vad_speech_pad_ms,
                "temperature": record.temperature,
                "temperature_increment_on_fallback": record.temperature_increment_on_fallback,
                "best_of": record.best_of,
                "compression_ratio_threshold": record.compression_ratio_threshold,
                "logprob_threshold": record.logprob_threshold,
                "no_speech_threshold": record.no_speech_threshold,
                "length_penalty": record.length_penalty,
                "repetition_penalty": record.repetition_penalty,
                "no_repeat_ngram_size": record.no_repeat_ngram_size,
                "suppress_tokens": record.suppress_tokens,
                "condition_on_previous_text": record.condition_on_previous_text,
                "initial_prompt": record.initial_prompt,
            }

            # Insert config data
            self.conn.execute(
                """
                INSERT INTO run_configs (
                    run_id, model_id, device, compute_type
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    record.model_id,
                    record.device,
                    record.compute_type,
                ),
            )

            # Insert metrics data
            self.conn.execute(
                """
                INSERT INTO run_metrics (
                    run_id, total_preprocess_time, total_transcribe_time, additional_metrics
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    record.total_preprocess_time,
                    record.total_transcribe_time,
                    json.dumps({}, sort_keys=True),  # Empty for now, extensible
                ),
            )

            # Insert individual parameters for flexibility
            params_to_insert: list[tuple[int, str, str, str, str]] = []

            # Preprocessing parameters
            for key, value in preprocess_config.items():
                if value is not None:
                    params_to_insert.append((run_id, "preprocess", key, str(value), type(value).__name__))

            # Transcription parameters
            for key, value in transcription_config.items():
                if value is not None:
                    params_to_insert.append((run_id, "transcription", key, str(value), type(value).__name__))

            # Model parameters
            if record.model_id:
                params_to_insert.append((run_id, "model", "model_id", record.model_id, "str"))
            if record.device:
                params_to_insert.append((run_id, "model", "device", record.device, "str"))
            if record.compute_type:
                params_to_insert.append((run_id, "model", "compute_type", record.compute_type, "str"))

            for param_data in params_to_insert:
                self.conn.execute(
                    """
                    INSERT INTO run_parameters (run_id, category, name, value, value_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    param_data,
                )

            # Commit transaction if we started one
            if transaction_started:
                self.conn.commit()
            else:
                # DuckDB may auto-commit, but ensure commit anyway
                try:
                    self.conn.commit()
                except Exception:
                    pass  # nosec B110 - safe pass for transaction commit

            LOGGER.debug("Recorded run with ID: %s", run_id)
            return run_id

        except Exception as e:
            # Rollback on error if transaction was started
            if transaction_started:
                try:
                    self.conn.rollback()
                except Exception:
                    # DuckDB may not support rollback, continue anyway
                    pass  # nosec B110 - safe pass for transaction rollback
            msg = f"Failed to record run: {e}"
            raise DatabaseError(msg) from e

    def get_run_by_id(self, run_id: int) -> dict[str, Any] | None:
        """Get a specific run by ID, reconstructing the full record from normalized tables.

        Args:
            run_id: The run ID to retrieve

        Returns:
            Dictionary with run info or None if not found
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            # Query to get core run data and configs
            cursor = self.conn.execute(
                """
                SELECT
                    r.id, r.recorded_at, r.input_folder, r.preset, r.language, r.preprocess_enabled,
                    r.files_found, r.succeeded, r.failed, r.total_processing_time,
                    r.total_audio_duration, r.speed_ratio,
                    c.model_id, c.device, c.compute_type,
                    m.total_preprocess_time, m.total_transcribe_time
                FROM runs r
                LEFT JOIN run_configs c ON r.id = c.run_id
                LEFT JOIN run_metrics m ON r.id = m.run_id
                WHERE r.id = ?
                """,
                [run_id],
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Get column names for safe access
            column_names = [desc[0] for desc in cursor.description]
            row_dict = dict(zip(column_names, row))

            # Reconstruct the flattened structure expected by callers
            result = {
                "id": row_dict["id"],
                "recorded_at": row_dict["recorded_at"],
                "input_folder": row_dict["input_folder"],
                "preset": row_dict["preset"],
                "language": row_dict["language"],
                "preprocess_enabled": bool(row_dict["preprocess_enabled"]),
                "files_found": row_dict["files_found"],
                "succeeded": row_dict["succeeded"],
                "failed": row_dict["failed"],
                "total_processing_time": row_dict["total_processing_time"],
                "total_audio_duration": row_dict["total_audio_duration"],
                "speed_ratio": row_dict["speed_ratio"],
                "model_id": row_dict.get("model_id"),
                "device": row_dict.get("device"),
                "compute_type": row_dict.get("compute_type"),
            }

            # Add metrics
            if row_dict.get("total_preprocess_time") is not None:
                result["total_preprocess_time"] = row_dict["total_preprocess_time"]
            if row_dict.get("total_transcribe_time") is not None:
                result["total_transcribe_time"] = row_dict["total_transcribe_time"]

            # Fetch parameters from run_parameters table
            params_cursor = self.conn.execute(
                """
                SELECT category, name, value, value_type
                FROM run_parameters
                WHERE run_id = ?
                ORDER BY category, name
                """,
                [run_id],
            )
            for param_row in params_cursor.fetchall():
                category, name, value, _value_type = param_row
                # Map category-specific names to result keys
                if category == "preprocess" and name == "profile":
                    result["preprocess_profile"] = value
                elif category == "preprocess":
                    result[name] = value
                elif category == "transcription":
                    result[name] = value
                elif category == "model":
                    # model_id, device, compute_type already in result from run_configs
                    # Only add if not already present
                    if name not in result or result[name] is None:
                        result[name] = value

            return result

        except Exception as e:
            msg = f"Failed to fetch run {run_id}: {e}"
            raise DatabaseError(msg) from e

    def get_run_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return stored run metadata entries ordered by timestamp descending."""
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            query = """
                SELECT
                    r.id, r.recorded_at, r.input_folder, r.preset, r.language, r.preprocess_enabled,
                    r.files_found, r.succeeded, r.failed, r.total_processing_time,
                    r.total_audio_duration, r.speed_ratio,
                    c.model_id, c.device, c.compute_type,
                    m.total_preprocess_time, m.total_transcribe_time
                FROM runs r
                LEFT JOIN run_configs c ON r.id = c.run_id
                LEFT JOIN run_metrics m ON r.id = m.run_id
                ORDER BY r.recorded_at DESC
            """
            params: tuple[int, ...] = ()
            if limit is not None:
                query += " LIMIT ?"
                params = (limit,)

            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return []

            # Get column names for safe access
            column_names = [desc[0] for desc in cursor.description]

            # Get all run IDs to fetch parameters in batch
            run_ids = [row[0] for row in rows]

            # Fetch all parameters for these runs
            if run_ids:
                # Build parameterized query safely
                placeholders = ",".join("?" * len(run_ids))
                params_query = (
                    "SELECT run_id, category, name, value, value_type "
                    "FROM run_parameters "
                    f"WHERE run_id IN ({placeholders}) "  # nosec B608 - placeholders safe (from len), values parameterized
                    "ORDER BY run_id, category, name"
                )
                params_cursor = self.conn.execute(params_query, run_ids)
                params_by_run: dict[int, list[tuple[str, str, str, str]]] = {}
                for param_row in params_cursor.fetchall():
                    run_id, category, name, value, value_type = param_row
                    if run_id not in params_by_run:
                        params_by_run[run_id] = []
                    params_by_run[run_id].append((category, name, value, value_type))
            else:
                params_by_run = {}

            results: list[dict[str, Any]] = []
            for row in rows:
                row_dict = dict(zip(column_names, row))
                run_id = row_dict["id"]

                # Reconstruct the flattened structure expected by callers
                result = {
                    "id": row_dict["id"],
                    "recorded_at": row_dict["recorded_at"],
                    "input_folder": row_dict["input_folder"],
                    "preset": row_dict["preset"],
                    "language": row_dict["language"],
                    "preprocess_enabled": bool(row_dict["preprocess_enabled"]),
                    "files_found": row_dict["files_found"],
                    "succeeded": row_dict["succeeded"],
                    "failed": row_dict["failed"],
                    "total_processing_time": row_dict["total_processing_time"],
                    "total_audio_duration": row_dict["total_audio_duration"],
                    "speed_ratio": row_dict["speed_ratio"],
                    "model_id": row_dict.get("model_id"),
                    "device": row_dict.get("device"),
                    "compute_type": row_dict.get("compute_type"),
                }

                # Add metrics
                if row_dict.get("total_preprocess_time") is not None:
                    result["total_preprocess_time"] = row_dict["total_preprocess_time"]
                if row_dict.get("total_transcribe_time") is not None:
                    result["total_transcribe_time"] = row_dict["total_transcribe_time"]

                # Add parameters from run_parameters table
                if run_id in params_by_run:
                    for category, name, value, value_type in params_by_run[run_id]:
                        if category == "preprocess" and name == "profile":
                            result["preprocess_profile"] = value
                        elif category == "preprocess":
                            result[name] = value
                        elif category == "transcription":
                            result[name] = value
                        elif category == "model":
                            # Only add if not already present from run_configs
                            if name not in result or result[name] is None:
                                result[name] = value

                results.append(result)

            return results

        except Exception as e:
            msg = f"Failed to fetch run history: {e}"
            raise DatabaseError(msg) from e

    def get_summary(self) -> dict[str, int]:
        """Get summary statistics of all files.

        Returns:
            Dictionary with counts by status
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            rows = self.conn.execute("SELECT status, COUNT(*) as count FROM transcriptions GROUP BY status").fetchall()
            return {row[0]: row[1] for row in rows}
        except Exception as e:
            msg = f"Failed to get summary: {e}"
            raise DatabaseError(msg) from e

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            LOGGER.debug("Database connection closed")

    def __enter__(self) -> "TranscriptionDatabase":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
