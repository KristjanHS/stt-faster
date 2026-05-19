# Problem 7 Solution: Schema Versioning & Migration Alternatives

**Problem**: Schema drift risk - migrations are reactive and column-based, leading to "works on my machine" DB breakages and production emergency patching.

**Current State**: 
- `_migrate_schema()` checks for missing columns and adds them reactively
- No version tracking
- No migration ordering
- Tests simulate missing columns to validate migrations

---

## Alternative 1: Simple Schema Version Table (Cleanest)

**Philosophy**: Minimal, straightforward approach - just enough to solve the problem.

### Implementation

1. **Schema Version Table**
   ```sql
   CREATE TABLE IF NOT EXISTS schema_version (
       version INTEGER PRIMARY KEY,
       applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
   );
   ```

2. **Ordered Migration List**
   ```python
   # In database.py
   MIGRATIONS = [
       (1, _migration_001_add_rnnoise_columns),
       (2, _migration_002_add_transcription_params),
       (3, _migration_003_add_preprocessing_params),
       # ... future migrations
   ]
   ```

3. **Migration Runner**
   ```python
   def _migrate_schema(self) -> None:
       """Run ordered migrations based on schema version."""
       # Initialize schema_version table
       self.conn.execute("""
           CREATE TABLE IF NOT EXISTS schema_version (
               version INTEGER PRIMARY KEY,
               applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
           )
       """)
       
       # Get current version
       result = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
       current_version = result[0] if result[0] is not None else 0
       
       # Run pending migrations
       for version, migration_func in MIGRATIONS:
           if version > current_version:
               LOGGER.info("Running migration %d", version)
               migration_func(self.conn)
               self.conn.execute(
                   "INSERT INTO schema_version (version) VALUES (?)",
                   (version,)
               )
               self.conn.commit()
   ```

4. **Individual Migration Functions**
   ```python
   def _migration_001_add_rnnoise_columns(conn: duckdb.DuckDBPyConnection) -> None:
       """Add rnnoise_model and rnnoise_mix columns."""
       # Check and add to file_metrics
       existing = _get_columns(conn, "file_metrics")
       if "rnnoise_model" not in existing:
           conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_model VARCHAR")
       if "rnnoise_mix" not in existing:
           conn.execute("ALTER TABLE file_metrics ADD COLUMN rnnoise_mix DOUBLE")
   
   def _migration_002_add_transcription_params(conn: duckdb.DuckDBPyConnection) -> None:
       """Add transcription parameter columns."""
       # ... similar pattern
   ```

### Pros
- ✅ Simple to understand and maintain
- ✅ Minimal code changes
- ✅ Clear migration ordering
- ✅ Easy to test (can test each migration independently)
- ✅ Version tracking prevents re-running migrations

### Cons
- ⚠️ No migration history (only latest version stored)
- ⚠️ No rollback capability
- ⚠️ No migration metadata (who/what/when details)

### Migration from Current Code
- Replace `_migrate_schema()` with version-based approach
- Extract existing column additions into numbered migration functions
- Set initial version to match current schema state

---

## Alternative 2: Schema Version + Migration Registry (Structured)

**Philosophy**: More organized, with metadata and better separation of concerns.

### Implementation

1. **Schema Version Table** (same as Alternative 1)

2. **Migration Registry with Metadata**
   ```python
   from dataclasses import dataclass
   from typing import Callable
   
   @dataclass
   class Migration:
       version: int
       name: str
       description: str
       migrate: Callable[[duckdb.DuckDBPyConnection], None]
   
   MIGRATIONS: list[Migration] = [
       Migration(
           version=1,
           name="add_rnnoise_columns",
           description="Add rnnoise_model and rnnoise_mix columns to file_metrics",
           migrate=_migration_001_add_rnnoise_columns,
       ),
       Migration(
           version=2,
           name="add_transcription_params",
           description="Add transcription parameter columns to file_metrics and runs",
           migrate=_migration_002_add_transcription_params,
       ),
       # ... more migrations
   ]
   ```

3. **Enhanced Migration Runner**
   ```python
   def _migrate_schema(self) -> None:
       """Run ordered migrations based on schema version."""
       self._ensure_schema_version_table()
       
       current_version = self._get_current_schema_version()
       pending = [m for m in MIGRATIONS if m.version > current_version]
       
       if not pending:
           LOGGER.debug("Database schema is up to date (version %d)", current_version)
           return
       
       LOGGER.info("Running %d pending migration(s)", len(pending))
       for migration in pending:
           try:
               LOGGER.info("Applying migration %d: %s - %s", 
                          migration.version, migration.name, migration.description)
               migration.migrate(self.conn)
               self._record_migration(migration.version)
               self.conn.commit()
           except Exception as e:
               LOGGER.error("Migration %d failed: %s", migration.version, e)
               raise DatabaseError(f"Migration {migration.version} failed") from e
   
   def _get_current_schema_version(self) -> int:
       """Get current schema version from database."""
       result = self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
       return result[0] if result[0] is not None else 0
   
   def _record_migration(self, version: int) -> None:
       """Record that a migration was applied."""
       self.conn.execute(
           "INSERT INTO schema_version (version) VALUES (?)",
           (version,)
       )
   ```

4. **Migration Validation Helper**
   ```python
   def _validate_migration_ordering() -> None:
       """Validate that migrations are properly ordered."""
       versions = [m.version for m in MIGRATIONS]
       if versions != sorted(versions):
           raise ValueError("Migrations must be ordered by version")
       if len(versions) != len(set(versions)):
           raise ValueError("Migration versions must be unique")
   ```

### Pros
- ✅ Better organization with metadata
- ✅ Self-documenting (name/description in code)
- ✅ Easier to debug (know which migration is running)
- ✅ Can validate migration ordering at startup
- ✅ Migration functions are testable independently

### Cons
- ⚠️ Slightly more complex than Alternative 1
- ⚠️ Still no rollback or detailed history

### Migration from Current Code
- Same as Alternative 1, but organize migrations into `Migration` dataclass instances
- Add validation to ensure migrations are ordered

---

## Alternative 3: Full Migration System with History (Best Practices)

**Philosophy**: Production-ready, industry-standard approach with full audit trail and safety features.

### Implementation

1. **Schema Version + Migration History Tables**
   ```sql
   CREATE TABLE IF NOT EXISTS schema_version (
       version INTEGER PRIMARY KEY,
       applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
   );
   
   CREATE TABLE IF NOT EXISTS migration_history (
       id INTEGER PRIMARY KEY,
       version INTEGER NOT NULL,
       name VARCHAR NOT NULL,
       description VARCHAR,
       applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       applied_by VARCHAR,  -- Could be user/system identifier
       success BOOLEAN NOT NULL,
       error_message VARCHAR,
       checksum VARCHAR,  -- Optional: hash of migration code for validation
       UNIQUE(version)
   );
   ```

2. **Migration Registry with Full Metadata**
   ```python
   @dataclass
   class Migration:
       version: int
       name: str
       description: str
       migrate: Callable[[duckdb.DuckDBPyConnection], None]
       rollback: Callable[[duckdb.DuckDBPyConnection], None] | None = None
       checksum: str | None = None  # Optional: hash of migration code
   
   MIGRATIONS: list[Migration] = [
       Migration(
           version=1,
           name="add_rnnoise_columns",
           description="Add rnnoise_model and rnnoise_mix columns",
           migrate=_migration_001_add_rnnoise_columns,
           rollback=_rollback_001_remove_rnnoise_columns,  # Optional
       ),
       # ... more migrations
   ]
   ```

3. **Robust Migration Runner**
   ```python
   def _migrate_schema(self) -> None:
       """Run ordered migrations with full history tracking."""
       self._ensure_migration_tables()
       
       current_version = self._get_current_schema_version()
       pending = [m for m in MIGRATIONS if m.version > current_version]
       
       if not pending:
           LOGGER.debug("Database schema is up to date (version %d)", current_version)
           return
       
       LOGGER.info("Running %d pending migration(s)", len(pending))
       for migration in pending:
           migration_id = self._start_migration(migration)
           try:
               LOGGER.info("Applying migration %d: %s", migration.version, migration.name)
               migration.migrate(self.conn)
               self._complete_migration(migration_id, success=True)
               self._record_schema_version(migration.version)
               self.conn.commit()
           except Exception as e:
               self._complete_migration(migration_id, success=False, error=str(e))
               self.conn.rollback()
               LOGGER.error("Migration %d failed, rolled back: %s", migration.version, e)
               raise DatabaseError(f"Migration {migration.version} failed") from e
   
   def _start_migration(self, migration: Migration) -> int:
       """Record migration start in history."""
       result = self.conn.execute("""
           INSERT INTO migration_history (version, name, description, success, checksum)
           VALUES (?, ?, ?, FALSE, ?)
           RETURNING id
       """, (migration.version, migration.name, migration.description, migration.checksum))
       return result.fetchone()[0]
   
   def _complete_migration(self, migration_id: int, success: bool, error: str | None = None) -> None:
       """Update migration history with result."""
       self.conn.execute("""
           UPDATE migration_history 
           SET success = ?, error_message = ?
           WHERE id = ?
       """, (success, error, migration_id))
   
   def get_migration_history(self) -> list[dict]:
       """Get full migration history for debugging/auditing."""
       return self.conn.execute("""
           SELECT version, name, description, applied_at, success, error_message
           FROM migration_history
           ORDER BY version
       """).fetchall()
   ```

4. **Migration Validation & Safety Checks**
   ```python
   def _validate_migrations(self) -> None:
       """Validate migration registry integrity."""
       # Check ordering
       versions = [m.version for m in MIGRATIONS]
       if versions != sorted(versions):
           raise ValueError("Migrations must be ordered by version")
       if len(versions) != len(set(versions)):
           raise ValueError("Migration versions must be unique")
       
       # Check for gaps (optional - might be intentional)
       expected = set(range(1, len(MIGRATIONS) + 1))
       actual = set(versions)
       gaps = expected - actual
       if gaps:
           LOGGER.warning("Migration version gaps detected: %s", gaps)
       
       # Validate checksums if provided (optional)
       for migration in MIGRATIONS:
           if migration.checksum:
               # Could validate that migration code hasn't changed
               pass
   ```

5. **Migration Status Query**
   ```python
   def get_schema_status(self) -> dict:
       """Get current schema status for debugging."""
       current_version = self._get_current_schema_version()
       history = self.get_migration_history()
       return {
           "current_version": current_version,
           "latest_available": max(m.version for m in MIGRATIONS) if MIGRATIONS else 0,
           "pending_count": len([m for m in MIGRATIONS if m.version > current_version]),
           "migration_history": history,
       }
   ```

### Pros
- ✅ Full audit trail (who/what/when)
- ✅ Error tracking (failed migrations recorded)
- ✅ Production-ready safety features
- ✅ Can query migration status
- ✅ Optional rollback support
- ✅ Optional checksum validation
- ✅ Industry-standard approach (similar to Alembic, Django migrations, etc.)

### Cons
- ⚠️ More complex implementation
- ⚠️ More code to maintain
- ⚠️ Might be overkill for simple use case
- ⚠️ Requires more testing

### Migration from Current Code
- Same extraction of migrations, but add history tracking
- Add migration history table
- Implement status query methods
- Consider adding rollback functions for critical migrations

---

## Recommendation

**For this project**: Start with **Alternative 1** (Simple Schema Version Table)

**Rationale**:
- Solves the core problem (schema drift, unordered migrations)
- Minimal complexity - easy to understand and maintain
- Can evolve to Alternative 2 or 3 if needed later
- Matches the project's preference for simplicity (see other refactorings)
- Quick to implement and test

**When to upgrade**:
- **Alternative 2**: If you need better organization and debugging info
- **Alternative 3**: If you need production-grade audit trails, multiple environments, or compliance requirements

---

## Implementation Steps (for Alternative 1)

1. Create `schema_version` table in `_init_db()`
2. Extract existing column additions into numbered migration functions
3. Create `MIGRATIONS` list with ordered migrations
4. Replace `_migrate_schema()` with version-based runner
5. Update tests to verify version tracking
6. Set initial migration version to match current schema state

**Estimated effort**: 2-3 hours
**Risk**: Low (backward compatible, can test with existing databases)
