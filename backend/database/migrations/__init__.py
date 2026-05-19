"""Ordered registry of schema migrations.

Add a new migration: drop a `migration_NNN.py` module next to this file
exposing a `migrate(conn)` callable, then append a `Migration(...)` entry to
`MIGRATIONS` in version order. `_init_db` invokes them via the registry.
"""

from __future__ import annotations

from backend.database.migrations import (
    migration_001,
    migration_002,
    migration_003,
    migration_004,
    migration_005,
    migration_006,
)
from backend.database.schema import Migration

MIGRATIONS: list[Migration] = [
    Migration(
        version=1,
        name="add_rnnoise_columns",
        description="Add rnnoise_model and rnnoise_mix columns to file_metrics table",
        migrate=migration_001.migrate,
    ),
    Migration(
        version=2,
        name="add_file_metrics_columns",
        description="Add loudnorm_target_lra and snr_estimation_method columns to file_metrics table",
        migrate=migration_002.migrate,
    ),
    Migration(
        version=3,
        name="add_transcription_params_file_metrics",
        description="Add transcription parameter columns to file_metrics table",
        migrate=migration_003.migrate,
    ),
    Migration(
        version=4,
        name="add_transcription_params_runs",
        description="Add transcription parameter columns to runs table",
        migrate=migration_004.migrate,
    ),
    Migration(
        version=5,
        name="add_preprocessing_params_runs",
        description="Add preprocessing parameter columns to runs table",
        migrate=migration_005.migrate,
    ),
    Migration(
        version=6,
        name="normalize_runs_schema",
        description="Normalize runs table schema with separate config, metrics, and parameters tables",
        migrate=migration_006.migrate,
    ),
]


def validate_migration_ordering() -> None:
    """Validate that migrations are properly ordered and unique."""
    versions = [m.version for m in MIGRATIONS]
    if versions != sorted(versions):
        raise ValueError("Migrations must be ordered by version number")
    if len(versions) != len(set(versions)):
        raise ValueError("Migration versions must be unique")
    if versions and versions[0] != 1:
        raise ValueError("Migrations must start with version 1")
