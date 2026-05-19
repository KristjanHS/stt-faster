"""Migration 3: add transcription parameter columns to file_metrics."""

from __future__ import annotations

import duckdb

from backend.database._columns import add_columns_if_missing


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    """Add transcription parameter columns to file_metrics table."""
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
    add_columns_if_missing(conn, "file_metrics", transcription_columns)
