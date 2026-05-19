"""Migration 4: add transcription parameter columns to runs table."""

from __future__ import annotations

import logging

import duckdb

from backend.database._columns import add_columns_if_missing

LOGGER = logging.getLogger(__name__)


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
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
    except Exception as exc:
        LOGGER.debug("migration 4 normalized-table probe failed: %s", exc)

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
    add_columns_if_missing(conn, "runs", transcription_columns)
