"""Migration 1: add rnnoise_model and rnnoise_mix columns to file_metrics."""

from __future__ import annotations

import logging

import duckdb

from backend.database._columns import get_columns

LOGGER = logging.getLogger(__name__)


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    """Add rnnoise_model and rnnoise_mix columns to file_metrics."""
    existing_columns = get_columns(conn, "file_metrics")
    if "rnnoise_model" not in existing_columns:
        LOGGER.debug("Migrating schema: adding rnnoise_model column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_model VARCHAR")
    if "rnnoise_mix" not in existing_columns:
        LOGGER.debug("Migrating schema: adding rnnoise_mix column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_mix DOUBLE")
