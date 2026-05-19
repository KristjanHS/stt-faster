"""Per-file metric persistence."""

from __future__ import annotations

import json
import logging

import duckdb

from backend.database.schema import FileMetricRecord, format_timestamp
from backend.exceptions import DatabaseError

LOGGER = logging.getLogger(__name__)


def record_file_metric(conn: duckdb.DuckDBPyConnection, record: FileMetricRecord) -> None:
    """Persist detailed metrics for a single file."""
    preprocess_steps_json = json.dumps(record.preprocess_steps or [], sort_keys=True)
    recorded_at = format_timestamp(record.recorded_at)

    try:
        conn.execute(
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
        conn.commit()
        LOGGER.debug("Recorded file metric for %s", record.audio_path)
    except Exception as e:
        msg = f"Failed to record file metrics for {record.audio_path}: {e}"
        raise DatabaseError(msg) from e
