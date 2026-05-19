"""Run-record persistence and reconstruction against the normalized schema."""

from __future__ import annotations

import json
import logging
from typing import Any

import duckdb

from backend.database._txn import begin_transaction, commit_transaction, rollback_transaction
from backend.database.schema import RunRecord, convert_parameter_value, format_timestamp
from backend.exceptions import DatabaseError

LOGGER = logging.getLogger(__name__)


def record_run(conn: duckdb.DuckDBPyConnection, record: RunRecord) -> int:
    """Persist a new run record and return its ID using normalized schema."""
    recorded_at = format_timestamp(record.recorded_at)

    # Begin transaction for atomicity
    transaction_started = begin_transaction(conn, "record_run")

    try:
        # Insert core run data
        cursor = conn.execute(
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
        conn.execute(
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
        conn.execute(
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
            conn.execute(
                """
                INSERT INTO run_parameters (run_id, category, name, value, value_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                param_data,
            )

        # Commit transaction if we started one. On the started path we let
        # failure propagate to the outer except so callers see DatabaseError;
        # on the auto-commit fallback we swallow because there's nothing to
        # roll back to.
        if transaction_started:
            conn.commit()
        else:
            commit_transaction(conn, "record_run")

        LOGGER.debug("Recorded run with ID: %s", run_id)
        return run_id

    except Exception as e:
        if transaction_started:
            rollback_transaction(conn, "record_run")
        msg = f"Failed to record run: {e}"
        raise DatabaseError(msg) from e


def get_run_by_id(conn: duckdb.DuckDBPyConnection, run_id: int) -> dict[str, Any] | None:
    """Get a specific run by ID, reconstructing the full record from normalized tables.

    Args:
        conn: open DuckDB connection
        run_id: The run ID to retrieve

    Returns:
        Dictionary with run info or None if not found
    """
    try:
        # Query to get core run data and configs
        cursor = conn.execute(
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
        params_cursor = conn.execute(
            """
            SELECT category, name, value, value_type
            FROM run_parameters
            WHERE run_id = ?
            ORDER BY category, name
            """,
            [run_id],
        )
        for param_row in params_cursor.fetchall():
            category, name, value, value_type = param_row
            # Convert value back to original type
            converted_value = convert_parameter_value(value, value_type)
            # Map category-specific names to result keys
            if category == "preprocess" and name == "profile":
                result["preprocess_profile"] = converted_value
            elif category == "preprocess":
                result[name] = converted_value
            elif category == "transcription":
                result[name] = converted_value
            elif category == "model":
                # model_id, device, compute_type already in result from run_configs
                # Only add if not already present
                if name not in result or result[name] is None:
                    result[name] = converted_value

        return result

    except Exception as e:
        msg = f"Failed to fetch run {run_id}: {e}"
        raise DatabaseError(msg) from e


def get_run_history(conn: duckdb.DuckDBPyConnection, limit: int | None = None) -> list[dict[str, Any]]:
    """Return stored run metadata entries ordered by timestamp descending."""
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

        cursor = conn.execute(query, params)
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
            params_cursor = conn.execute(params_query, run_ids)
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
                    # Convert value back to original type
                    converted_value = convert_parameter_value(value, value_type)
                    if category == "preprocess" and name == "profile":
                        result["preprocess_profile"] = converted_value
                    elif category == "preprocess":
                        result[name] = converted_value
                    elif category == "transcription":
                        result[name] = converted_value
                    elif category == "model":
                        # Only add if not already present from run_configs
                        if name not in result or result[name] is None:
                            result[name] = converted_value

            results.append(result)

        return results

    except Exception as e:
        msg = f"Failed to fetch run history: {e}"
        raise DatabaseError(msg) from e
