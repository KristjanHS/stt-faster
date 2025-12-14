# Problem 9: State DB Naming/Extension Inconsistency - Solution Alternatives

## Current State

- **Production code**: Uses `transcribe_state.duckdb` (correct - matches DuckDB format)
- **Tests**: Mix of `.db` and `.duckdb` suffixes in temp files
- **Documentation**: References `transcribe_state.db` (incorrect - outdated)
- **Scripts**: Reference `transcribe_state.db` (incorrect - outdated)

**Impact**: Confusing operations/debugging; risk of creating new DBs instead of migrating existing ones; harder cross-platform support.

---

## Alternative 1: Cleanest - Standardize Extension and Update References

**Approach**: Pick `.duckdb` as canonical (already used in production), update all references, no migration logic.

### Implementation Steps

1. **Update documentation references**:
   - `README.md`: Change `transcribe_state.db` → `transcribe_state.duckdb`
   - `scripts/windows/README.md`: Update references
   - All `.cursor/plans/*.md`: Update examples

2. **Update test fixtures**:
   - `tests/unit/conftest.py`: Change `.db` → `.duckdb` in temp file suffix
   - `tests/integration/conftest.py`: Change `.db` → `.duckdb`
   - `tests/e2e/conftest.py`: Change `.db` → `.duckdb`

3. **Update scripts**:
   - `scripts/new_project_cleanup.sh`: Change `transcribe_state.db` → `transcribe_state.duckdb`
   - `scripts/new_project_bootstrap.sh`: Change `transcribe_state.db` → `transcribe_state.duckdb`

4. **Update legacy documentation** (optional cleanup):
   - `REFACTORING_SUMMARY.md`: Update historical references (or mark as historical)

### Pros
- ✅ Minimal code changes (no migration logic)
- ✅ Fast to implement (~30 minutes)
- ✅ Clear, single source of truth
- ✅ No runtime overhead

### Cons
- ❌ Users with existing `transcribe_state.db` files won't be migrated automatically
- ❌ May create duplicate databases if old `.db` file exists
- ❌ No backward compatibility

### Risk Level
**Low** - Only affects users who have old `.db` files (likely few, since XDG migration was recent)

---

## Alternative 2: Migration-Aware Standardization

**Approach**: Standardize on `.duckdb`, add one-time migration logic to detect and migrate existing `.db` files.

### Implementation Steps

1. **Add migration detection in `get_default_db_path()` or `TranscriptionDatabase.__init__()`**:
   ```python
   def get_default_db_path() -> Path:
       """Get XDG-compliant default database path with migration support."""
       app_data_dir = _get_app_data_dir()
       canonical_path = app_data_dir / "transcribe_state.duckdb"
       legacy_path = app_data_dir / "transcribe_state.db"
       
       # One-time migration: if legacy exists and canonical doesn't, migrate
       if legacy_path.exists() and not canonical_path.exists():
           _migrate_legacy_db(legacy_path, canonical_path)
       
       return canonical_path
   
   def _migrate_legacy_db(legacy_path: Path, canonical_path: Path) -> None:
       """Migrate legacy .db file to canonical .duckdb location."""
       import shutil
       import logging
       
       logger = logging.getLogger(__name__)
       logger.info(
           f"Migrating legacy database {legacy_path} to {canonical_path}"
       )
       
       # DuckDB can read its own format regardless of extension
       # Copy file and update extension
       shutil.copy2(legacy_path, canonical_path)
       
       # Optionally: backup legacy file before removal
       backup_path = legacy_path.with_suffix('.db.backup')
       shutil.move(legacy_path, backup_path)
       logger.info(f"Legacy database backed up to {backup_path}")
   ```

2. **Update all references** (same as Alternative 1):
   - Documentation
   - Test fixtures
   - Scripts

3. **Add migration test**:
   ```python
   def test_legacy_db_migration(tmp_path: Path):
       """Test that legacy .db files are migrated to .duckdb."""
       app_dir = tmp_path / "stt-faster"
       app_dir.mkdir()
       
       legacy_db = app_dir / "transcribe_state.db"
       canonical_db = app_dir / "transcribe_state.duckdb"
       
       # Create legacy DB
       db = TranscriptionDatabase(str(legacy_db))
       db.close()
       
       # Trigger migration
       migrated_path = get_default_db_path()
       assert migrated_path == canonical_db
       assert canonical_db.exists()
       assert legacy_db.with_suffix('.db.backup').exists()
   ```

### Pros
- ✅ Backward compatible - handles existing `.db` files
- ✅ One-time automatic migration
- ✅ Prevents duplicate database creation
- ✅ Still relatively simple (~1-2 hours)

### Cons
- ❌ Adds migration logic complexity
- ❌ Requires testing migration scenarios
- ❌ Backup file management (optional cleanup needed)

### Risk Level
**Low-Medium** - Migration logic needs careful testing, but DuckDB format is extension-agnostic

---

## Alternative 3: Best Practices Aligned - Comprehensive Solution

**Approach**: Full migration system with validation, error handling, CLI migration command, and comprehensive documentation updates.

### Implementation Steps

1. **Create migration module** (`backend/database_migration.py`):
   ```python
   """Database migration utilities for extension standardization."""
   
   from pathlib import Path
   import logging
   import shutil
   from typing import Optional
   
   logger = logging.getLogger(__name__)
   
   LEGACY_EXTENSIONS = [".db", ".sqlite", ".sqlite3"]
   CANONICAL_EXTENSION = ".duckdb"
   CANONICAL_NAME = "transcribe_state.duckdb"
   
   def find_legacy_databases(app_data_dir: Path) -> list[Path]:
       """Find all legacy database files in app data directory."""
       legacy_files = []
       for ext in LEGACY_EXTENSIONS:
           legacy_path = app_data_dir / f"transcribe_state{ext}"
           if legacy_path.exists():
               legacy_files.append(legacy_path)
       return legacy_files
   
   def validate_duckdb_file(db_path: Path) -> bool:
       """Validate that file is a valid DuckDB database."""
       try:
           import duckdb
           conn = duckdb.connect(str(db_path), read_only=True)
           # Try to query schema to verify it's a valid DuckDB file
           conn.execute("SELECT 1")
           conn.close()
           return True
       except Exception:
           return False
   
   def migrate_legacy_database(
       legacy_path: Path,
       canonical_path: Path,
       create_backup: bool = True,
   ) -> bool:
       """Migrate legacy database to canonical path.
       
       Returns:
           True if migration succeeded, False otherwise.
       """
       if canonical_path.exists():
           logger.warning(
               f"Canonical database already exists at {canonical_path}, "
               f"skipping migration from {legacy_path}"
           )
           return False
       
       if not validate_duckdb_file(legacy_path):
           logger.error(
               f"Legacy file {legacy_path} is not a valid DuckDB database, "
               f"skipping migration"
           )
           return False
       
       try:
           # Create backup
           if create_backup:
               backup_path = legacy_path.with_suffix(
                   f"{legacy_path.suffix}.backup"
               )
               shutil.copy2(legacy_path, backup_path)
               logger.info(f"Created backup: {backup_path}")
           
           # Migrate to canonical path
           shutil.copy2(legacy_path, canonical_path)
           logger.info(
               f"Migrated database from {legacy_path} to {canonical_path}"
           )
           
           # Remove legacy file after successful migration
           legacy_path.unlink()
           logger.info(f"Removed legacy database: {legacy_path}")
           
           return True
       except Exception as e:
           logger.error(f"Migration failed: {e}", exc_info=True)
           return False
   
   def auto_migrate_on_startup(app_data_dir: Path) -> bool:
       """Automatically migrate legacy databases on application startup.
       
       Returns:
           True if any migration occurred, False otherwise.
       """
       canonical_path = app_data_dir / CANONICAL_NAME
       if canonical_path.exists():
           return False  # Already using canonical path
       
       legacy_files = find_legacy_databases(app_data_dir)
       if not legacy_files:
           return False  # No legacy files to migrate
       
       # If multiple legacy files exist, prefer .db
       legacy_path = next(
           (f for f in legacy_files if f.suffix == ".db"),
           legacy_files[0]
       )
       
       return migrate_legacy_database(legacy_path, canonical_path)
   ```

2. **Integrate into `get_default_db_path()`**:
   ```python
   def get_default_db_path() -> Path:
       """Get XDG-compliant default database path with automatic migration."""
       app_data_dir = _get_app_data_dir()
       canonical_path = app_data_dir / "transcribe_state.duckdb"
       
       # Auto-migrate on first access
       from backend.database_migration import auto_migrate_on_startup
       auto_migrate_on_startup(app_data_dir)
       
       return canonical_path
   ```

3. **Add CLI migration command** (`backend/cli/migration_commands.py`):
   ```python
   """CLI commands for database migration."""
   
   import argparse
   from pathlib import Path
   import logging
   
   from backend.database_migration import (
       find_legacy_databases,
       migrate_legacy_database,
       auto_migrate_on_startup,
   )
   from backend.database import get_default_db_path
   
   logger = logging.getLogger(__name__)
   
   def migrate_db_command(args: argparse.Namespace) -> int:
       """Migrate legacy database files to canonical format."""
       app_data_dir = get_default_db_path().parent
       canonical_path = get_default_db_path()
       
       legacy_files = find_legacy_databases(app_data_dir)
       
       if not legacy_files:
           print("No legacy database files found.")
           return 0
       
       print(f"Found {len(legacy_files)} legacy database file(s):")
       for f in legacy_files:
           print(f"  - {f}")
       
       if canonical_path.exists():
           print(f"\nCanonical database already exists: {canonical_path}")
           print("Migration skipped to prevent data loss.")
           return 1
       
       # Migrate first legacy file found
       legacy_path = legacy_files[0]
       success = migrate_legacy_database(
           legacy_path,
           canonical_path,
           create_backup=not args.no_backup,
       )
       
       if success:
           print(f"\n✓ Migration successful: {canonical_path}")
           return 0
       else:
           print("\n✗ Migration failed. Check logs for details.")
           return 1
   
   def add_migration_parser(subparsers: argparse._SubParsersAction) -> None:
       """Add migration command to CLI."""
       migrate_parser = subparsers.add_parser(
           "migrate-db",
           help="Migrate legacy database files to canonical format",
       )
       migrate_parser.add_argument(
           "--no-backup",
           action="store_true",
           help="Skip creating backup of legacy database",
       )
       migrate_parser.set_defaults(func=migrate_db_command)
   ```

4. **Update all references** (same as Alternatives 1 & 2):
   - Documentation
   - Test fixtures  
   - Scripts

5. **Add comprehensive tests**:
   - Test migration from `.db` to `.duckdb`
   - Test migration with existing canonical file (should skip)
   - Test migration with invalid DuckDB file (should fail gracefully)
   - Test CLI migration command
   - Test auto-migration on startup

6. **Update documentation**:
   - Add migration section to `README.md`
   - Document CLI migration command
   - Update troubleshooting section

### Pros
- ✅ Full backward compatibility with automatic migration
- ✅ CLI command for manual migration control
- ✅ Validation prevents corrupt database migration
- ✅ Comprehensive error handling and logging
- ✅ Backup creation for safety
- ✅ Well-tested and documented
- ✅ Handles edge cases (multiple legacy files, invalid files)

### Cons
- ❌ Most complex implementation (~4-6 hours)
- ❌ More code to maintain
- ❌ Requires extensive testing

### Risk Level
**Low** - Comprehensive validation and error handling minimize risks

---

## Recommendation

**Choose Alternative 2 (Migration-Aware Standardization)** for the best balance of:
- Backward compatibility
- Implementation simplicity
- Risk mitigation
- Maintenance burden

**Upgrade to Alternative 3** if:
- You expect many users with legacy `.db` files
- You want explicit migration control via CLI
- You need comprehensive validation and error handling

**Use Alternative 1** only if:
- You're certain no users have legacy `.db` files
- You want the absolute simplest solution
- You can accept potential duplicate database creation

---

## Implementation Checklist (for chosen alternative)

- [ ] Update `get_default_db_path()` with migration logic (if Alternative 2 or 3)
- [ ] Update all documentation references (`.db` → `.duckdb`)
- [ ] Update test fixtures (`.db` → `.duckdb`)
- [ ] Update scripts (`.db` → `.duckdb`)
- [ ] Add migration tests
- [ ] Add CLI migration command (if Alternative 3)
- [ ] Update README with migration notes
- [ ] Run full test suite
- [ ] Verify no `.db` references remain in codebase (grep check)
