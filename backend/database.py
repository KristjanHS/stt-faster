"""Database operations for transcription state tracking."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

            # Runs table - Stores BATCH CONFIGURATION and AGGREGATES
            # Removed JSON blobs, expanded config columns
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    recorded_at TIMESTAMP NOT NULL,
                    input_folder VARCHAR,

                    -- Configuration
                    preset VARCHAR NOT NULL,
                    language VARCHAR,
                    preprocess_enabled BOOLEAN NOT NULL,
                    preprocess_profile VARCHAR,
                    target_sample_rate INTEGER,
                    target_channels INTEGER,
                    loudnorm_preset VARCHAR,
                    -- Preprocessing parameters used
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

                    -- Aggregates
                    files_found INTEGER NOT NULL,
                    succeeded INTEGER NOT NULL,
                    failed INTEGER NOT NULL,
                    total_processing_time DOUBLE,
                    total_preprocess_time DOUBLE,
                    total_transcribe_time DOUBLE,
                    total_audio_duration DOUBLE,
                    speed_ratio DOUBLE
                );
                CREATE SEQUENCE IF NOT EXISTS seq_runs_id START 1;
                ALTER TABLE runs ALTER COLUMN id SET DEFAULT nextval('seq_runs_id');
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

            # Migrate existing tables: add missing columns if they don't exist
            self._migrate_schema()

            LOGGER.debug("DuckDB initialized at %s", self.db_path)
        except Exception as e:
            msg = f"Failed to initialize database at {self.db_path}: {e}"
            raise DatabaseError(msg) from e

    def _migrate_schema(self) -> None:
        """Migrate existing database schema by adding missing columns."""
        if self.conn is None:
            return

        try:
            # Migrate file_metrics table
            try:
                columns_result = self.conn.execute("DESCRIBE file_metrics").fetchall()
                existing_columns: set[str] = {row[0] for row in columns_result}
            except Exception:
                # Table doesn't exist yet, no migration needed
                existing_columns = set()

            # Add rnnoise_model column if it doesn't exist
            if "rnnoise_model" not in existing_columns:
                LOGGER.debug("Migrating schema: adding rnnoise_model column to file_metrics")
                self.conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_model VARCHAR")

            # Add rnnoise_mix column if it doesn't exist
            if "rnnoise_mix" not in existing_columns:
                LOGGER.debug("Migrating schema: adding rnnoise_mix column to file_metrics")
                self.conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_mix DOUBLE")

            # Add loudnorm_target_lra column if it doesn't exist
            if "loudnorm_target_lra" not in existing_columns:
                LOGGER.debug("Migrating schema: adding loudnorm_target_lra column to file_metrics")
                self.conn.execute("ALTER TABLE file_metrics ADD COLUMN loudnorm_target_lra DOUBLE")

            # Add snr_estimation_method column if it doesn't exist
            if "snr_estimation_method" not in existing_columns:
                LOGGER.debug("Migrating schema: adding snr_estimation_method column to file_metrics")
                self.conn.execute("ALTER TABLE file_metrics ADD COLUMN snr_estimation_method VARCHAR")

            # Add transcription parameter columns to file_metrics if they don't exist
            # List of transcription parameter columns that might be missing from old databases
            transcription_columns_to_add = {
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

            for column_name, column_type in transcription_columns_to_add.items():
                if column_name not in existing_columns:
                    LOGGER.debug("Migrating schema: adding %s column to file_metrics", column_name)
                    # Use hardcoded SQL statements matching existing migration pattern
                    # This avoids any potential SQL injection risks
                    if column_type == "DOUBLE":
                        self.conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} DOUBLE")
                    elif column_type == "VARCHAR":
                        self.conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} VARCHAR")
                    elif column_type == "INTEGER":
                        self.conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} INTEGER")
                    elif column_type == "BOOLEAN":
                        self.conn.execute(f"ALTER TABLE file_metrics ADD COLUMN {column_name} BOOLEAN")
                    else:
                        LOGGER.warning("Unknown column type %s for %s, skipping migration", column_type, column_name)

            # Migrate runs table - add missing transcription parameters
            try:
                runs_columns_result = self.conn.execute("DESCRIBE runs").fetchall()
                runs_existing_columns: set[str] = {row[0] for row in runs_columns_result}
            except Exception:
                # Table doesn't exist yet, no migration needed
                runs_existing_columns = set()

            # List of new transcription parameter columns to add
            # Using hardcoded strings for each column to match existing migration pattern
            # and avoid any potential SQL injection risks
            new_columns_to_add = {
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

            for column_name, column_type in new_columns_to_add.items():
                if column_name not in runs_existing_columns:
                    LOGGER.debug("Migrating schema: adding %s column to runs", column_name)
                    # Use hardcoded SQL statements matching existing migration pattern
                    # This avoids any potential SQL injection risks
                    if column_type == "DOUBLE":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} DOUBLE")
                    elif column_type == "VARCHAR":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} VARCHAR")
                    elif column_type == "INTEGER":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} INTEGER")
                    elif column_type == "BOOLEAN":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} BOOLEAN")
                    else:
                        LOGGER.warning("Unknown column type %s for %s, skipping", column_type, column_name)

            # Add preprocessing parameter columns to runs table
            preprocessing_columns_to_add = {
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

            for column_name, column_type in preprocessing_columns_to_add.items():
                if column_name not in runs_existing_columns:
                    LOGGER.debug("Migrating schema: adding %s column to runs", column_name)
                    if column_type == "DOUBLE":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} DOUBLE")
                    elif column_type == "VARCHAR":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} VARCHAR")
                    elif column_type == "INTEGER":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} INTEGER")
                    elif column_type == "BOOLEAN":
                        self.conn.execute(f"ALTER TABLE runs ADD COLUMN {column_name} BOOLEAN")
                    else:
                        LOGGER.warning("Unknown column type %s for %s, skipping", column_type, column_name)

        except Exception as e:
            # Log but don't fail - migration errors shouldn't break initialization
            LOGGER.warning("Schema migration encountered an issue: %s", e)

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
        """Persist a new run record and return its ID."""
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        recorded_at = _format_timestamp(record.recorded_at)

        try:
            # DuckDB execute with parameters and RETURNING clause
            cursor = self.conn.execute(
                """
                INSERT INTO runs (
                    recorded_at, input_folder,
                    preset, language,
                    preprocess_enabled, preprocess_profile, target_sample_rate, target_channels, loudnorm_preset,
                    volume_adjustment_db, resampler, sample_format,
                    loudnorm_target_i, loudnorm_target_tp, loudnorm_target_lra, loudnorm_backend,
                    denoise_method, denoise_library, rnnoise_model, rnnoise_mix, snr_estimation_method,
                    model_id, device, compute_type,
                    beam_size, patience, word_timestamps, task, chunk_length,
                    vad_filter, vad_threshold, vad_min_speech_duration_ms, vad_max_speech_duration_s,
                    vad_min_silence_duration_ms, vad_speech_pad_ms,
                    temperature, temperature_increment_on_fallback, best_of,
                    compression_ratio_threshold, logprob_threshold, no_speech_threshold,
                    length_penalty, repetition_penalty, no_repeat_ngram_size,
                    suppress_tokens, condition_on_previous_text, initial_prompt,
                    files_found, succeeded, failed,
                    total_processing_time,
                    total_preprocess_time,
                    total_transcribe_time,
                    total_audio_duration,
                    speed_ratio
                ) VALUES (
                    ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                RETURNING id;
                """,
                (
                    recorded_at,
                    record.input_folder,
                    record.preset,
                    record.language,
                    int(record.preprocess_enabled),
                    record.preprocess_profile,
                    record.target_sample_rate,
                    record.target_channels,
                    record.loudnorm_preset,
                    record.volume_adjustment_db,
                    record.resampler,
                    record.sample_format,
                    record.loudnorm_target_i,
                    record.loudnorm_target_tp,
                    record.loudnorm_target_lra,
                    record.loudnorm_backend,
                    record.denoise_method,
                    record.denoise_library,
                    record.rnnoise_model,
                    record.rnnoise_mix,
                    record.snr_estimation_method,
                    record.model_id,
                    record.device,
                    record.compute_type,
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
                    record.files_found,
                    record.succeeded,
                    record.failed,
                    record.total_processing_time,
                    record.total_preprocess_time,
                    record.total_transcribe_time,
                    record.total_audio_duration,
                    record.speed_ratio,
                ),
            )

            # Fetch the returned ID
            row = cursor.fetchone()
            if row:
                run_id = row[0]
                self.conn.commit()
                LOGGER.debug("Recorded run with ID: %s", run_id)
                return run_id

            raise DatabaseError("Failed to retrieve inserted run ID")

        except Exception as e:
            msg = f"Failed to record run: {e}"
            raise DatabaseError(msg) from e

    def get_run_by_id(self, run_id: int) -> dict[str, Any] | None:
        """Get a specific run by ID.

        Args:
            run_id: The run ID to retrieve

        Returns:
            Dictionary with run info or None if not found
        """
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.execute("SELECT * FROM runs WHERE id = ?", [run_id])
            row = cursor.fetchone()

            if not row:
                return None

            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        except Exception as e:
            msg = f"Failed to fetch run {run_id}: {e}"
            raise DatabaseError(msg) from e

    def get_run_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return stored run metadata entries ordered by timestamp descending."""
        if not self.conn:
            msg = "Database not initialized"
            raise DatabaseError(msg)

        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM runs ORDER BY recorded_at DESC"
            params: tuple[int, ...] = ()
            if limit is not None:
                query += " LIMIT ?"
                params = (limit,)
            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return []

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
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
