"""Migration 5: add preprocessing parameter columns to runs table."""

from __future__ import annotations

import logging

import duckdb

from backend.database._columns import add_columns_if_missing

LOGGER = logging.getLogger(__name__)


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
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
    except Exception as exc:
        LOGGER.debug("migration 5 normalized-table probe failed: %s", exc)

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
    add_columns_if_missing(conn, "runs", preprocessing_columns)
