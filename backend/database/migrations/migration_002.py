"""Migration 2: add loudnorm_target_lra and snr_estimation_method to file_metrics."""

from __future__ import annotations

import logging

import duckdb

from backend.database._columns import get_columns

LOGGER = logging.getLogger(__name__)


def migrate(conn: duckdb.DuckDBPyConnection) -> None:
    """Add loudnorm_target_lra and snr_estimation_method columns to file_metrics."""
    existing_columns = get_columns(conn, "file_metrics")
    if "loudnorm_target_lra" not in existing_columns:
        LOGGER.debug("Migrating schema: adding loudnorm_target_lra column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN loudnorm_target_lra DOUBLE")
    if "snr_estimation_method" not in existing_columns:
        LOGGER.debug("Migrating schema: adding snr_estimation_method column to file_metrics")
        conn.execute("ALTER TABLE file_metrics ADD COLUMN snr_estimation_method VARCHAR")
