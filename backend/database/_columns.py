"""Column discovery + idempotent ALTER TABLE ADD COLUMN helpers."""

from __future__ import annotations

import logging

import duckdb

LOGGER = logging.getLogger(__name__)


def get_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    """Get existing column names for a table."""
    try:
        columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
        return {row[0] for row in columns_result}
    except Exception:
        # Table doesn't exist yet
        return set()


def add_columns_if_missing(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    """ALTER TABLE ADD COLUMN for each column in `columns` not already on `table_name`.

    `columns` maps column name → SQL type (e.g. {"patience": "DOUBLE"}). Existing
    columns are skipped silently; each newly added column emits a DEBUG log.
    """
    existing_columns = get_columns(conn, table_name)
    for column_name, column_type in columns.items():
        if column_name in existing_columns:
            continue
        LOGGER.debug("Migrating schema: adding %s column to %s", column_name, table_name)
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
