"""DuckDB transaction helpers with swallowed-on-failure semantics.

DuckDB may auto-commit DDL or refuse explicit transactions in some modes; the
helpers log and treat failures as "no transaction" so callers can short-circuit
matching COMMIT/ROLLBACK calls without raising to the caller.
"""

from __future__ import annotations

import logging

import duckdb

LOGGER = logging.getLogger(__name__)


def begin_transaction(conn: duckdb.DuckDBPyConnection, label: str) -> bool:
    """Issue BEGIN TRANSACTION; return whether it succeeded.

    DuckDB may auto-commit DDL or not support explicit transactions in some modes;
    we log and treat that as "no transaction started" so callers can short-circuit
    matching COMMIT/ROLLBACK calls.
    """
    try:
        conn.execute("BEGIN TRANSACTION")
        return True
    except Exception as exc:
        LOGGER.debug(
            "%s: BEGIN TRANSACTION failed (DuckDB may not support explicit transactions): %s",
            label,
            exc,
        )
        return False


def commit_transaction(conn: duckdb.DuckDBPyConnection, label: str) -> None:
    """Issue COMMIT; log on failure rather than raising.

    Used on cleanup/auto-commit paths where the caller has already decided that a
    swallowed commit error is acceptable (the alternative is leaking the failure
    to a caller who has no meaningful recovery). Callers that need failure to
    propagate should issue `conn.commit()` directly.
    """
    try:
        conn.execute("COMMIT")
    except Exception as exc:
        LOGGER.debug("%s: commit failed: %s", label, exc)


def rollback_transaction(conn: duckdb.DuckDBPyConnection, label: str) -> None:
    """Issue ROLLBACK; log on failure rather than raising."""
    try:
        conn.execute("ROLLBACK")
    except Exception as exc:
        LOGGER.debug("%s: rollback failed (DuckDB may not support rollback): %s", label, exc)
