"""Migration 6: normalize runs table schema into runs / run_configs / run_metrics / run_parameters.

Moved verbatim from `backend/database.py` per Stage D.2's "move untouched"
guardrail; the only edits are import-path adjustments to the new helper
locations. The latent data-migration bug documented in the function body is
explicitly out of scope for this PR.
"""

from __future__ import annotations

import json
import logging

import duckdb

from backend.database._txn import begin_transaction, commit_transaction, rollback_transaction

LOGGER = logging.getLogger(__name__)


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    """Normalize runs table schema for Alternative 3 - split into multiple tables for flexibility.

    KNOWN-LATENT BUG (out of scope for the Stage D rollout): the wide-row
    data-migration loop further down this function creates `run_parameters`
    without a sequence default before its INSERT, so a direct invocation against
    a synthetic pre-006 wide-shape fixture fails with a NOT NULL constraint on
    `run_parameters.id`. In production this never fires — the first guard below
    short-circuits whenever the normalized tables already exist, which is true
    for every database on every restart since the normalized schema was
    introduced — so the data-migration loop is effectively dead code. Fixing it
    requires a synthetic-wide-fixture test + a sequence default before the
    INSERT, both deferred to a separate PR.
    """
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
    except Exception as exc:
        LOGGER.debug("migration 6 normalized-table probe failed; will fall through to old-column check: %s", exc)

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
    transaction_started = begin_transaction(conn, "migration 6")

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
                except Exception as exc:
                    LOGGER.debug("migration 6 abort-path: runs_new drop failed: %s", exc)
                if transaction_started:
                    commit_transaction(conn, "migration 6 abort-path")
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
            except Exception as exc:
                LOGGER.debug("migration 6 no-data-path: runs_new drop failed: %s", exc)
            try:
                conn.execute("DROP TABLE IF EXISTS file_metrics_new")
            except Exception as exc:
                LOGGER.debug("migration 6 no-data-path: file_metrics_new drop failed: %s", exc)
            if transaction_started:
                commit_transaction(conn, "migration 6 no-data-path")
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
        except Exception as exc:
            LOGGER.debug("migration 6 drop run_parameters failed: %s", exc)
        try:
            conn.execute("DROP TABLE IF EXISTS run_metrics")
        except Exception as exc:
            LOGGER.debug("migration 6 drop run_metrics failed: %s", exc)
        try:
            conn.execute("DROP TABLE IF EXISTS run_configs")
        except Exception as exc:
            LOGGER.debug("migration 6 drop run_configs failed: %s", exc)
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
                    except Exception as exc:
                        LOGGER.debug("migration 6 already-normalized branch: runs_new drop failed: %s", exc)
                else:
                    # Runs exists but is not normalized - we can't drop it due to foreign keys
                    # This shouldn't happen if the early checks worked, but handle it gracefully
                    LOGGER.warning("Runs table exists with old schema but can't be dropped due to dependencies")
                    try:
                        conn.execute("DROP TABLE IF EXISTS runs_new")
                    except Exception as exc:
                        LOGGER.debug("migration 6 stuck-old-schema branch: runs_new drop failed: %s", exc)
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
            except Exception as exc:
                LOGGER.debug("migration 6 runs.id sequence default failed (likely foreign-key blocked): %s", exc)
        conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_run_parameters_id START 1")
        conn.execute("ALTER TABLE run_parameters ALTER COLUMN id SET DEFAULT nextval('seq_run_parameters_id')")

        # Commit transaction if we started one
        if transaction_started:
            commit_transaction(conn, "migration 6 final")

        LOGGER.info("Completed runs table normalization migration")
    except Exception:
        # Rollback on error if transaction was started
        if transaction_started:
            rollback_transaction(conn, "migration 6 rollback")
        raise
