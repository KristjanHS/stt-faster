# Problem 8 Implementation Fix Plan

## Goal
Fix critical issues in the normalized schema implementation while keeping changes minimal and safe.

---

## Issues to Fix

### 1. **Remove Data Duplication** (HIGHEST PRIORITY)
**Problem**: Data stored in both JSON configs AND run_parameters table  
**Fix**: Remove `run_parameters` table usage - JSON configs are sufficient

### 2. **Fix Hard-coded Column Indices** (HIGH PRIORITY)  
**Problem**: Fragile `row[15]`, `row[16]` indices in get methods  
**Fix**: Use column names from cursor.description

### 3. **Add Transaction to Migration 6** (MEDIUM PRIORITY)
**Problem**: Migration not atomic - could leave DB inconsistent  
**Fix**: Wrap in explicit transaction (though DuckDB may auto-commit DDL)

### 4. **Add Transaction to record_run()** (MEDIUM PRIORITY)
**Problem**: Multiple INSERTs without explicit transaction  
**Fix**: Begin transaction before first INSERT, rollback on failure

---

## Implementation Steps

### Step 1: Remove run_parameters Table Usage

**Files to modify**: `backend/database.py`

1. **Remove run_parameters creation from `_init_db()`**
   - Delete the CREATE TABLE statement for run_parameters
   - Remove sequence creation for run_parameters

2. **Remove run_parameters from migration 6**
   - Remove CREATE TABLE for run_parameters
   - Remove all INSERT INTO run_parameters statements
   - Remove DROP TABLE for run_parameters

3. **Remove run_parameters from `record_run()`**
   - Delete the entire `params_to_insert` section
   - Remove all INSERT INTO run_parameters code

**Rationale**: JSON configs in `run_configs` table provide all needed flexibility. The `run_parameters` table adds complexity without benefit.

---

### Step 2: Fix Hard-coded Column Indices

**Files to modify**: `backend/database.py`

1. **Create helper function**:
```python
def _get_column_index(cursor, column_name: str) -> int:
    """Get column index by name from cursor description."""
    for idx, desc in enumerate(cursor.description):
        if desc[0] == column_name:
            return idx
    raise ValueError(f"Column {column_name} not found")
```

2. **Update `get_run_by_id()`**:
   - Replace `row[15]` with `row[_get_column_index(cursor, "preprocess_config")]`
   - Replace `row[16]` with `row[_get_column_index(cursor, "transcription_config")]`
   - Replace `row[17]` with `row[_get_column_index(cursor, "total_preprocess_time")]`
   - Replace `row[18]` with `row[_get_column_index(cursor, "total_transcribe_time")]`

3. **Update `get_run_history()`**:
   - Apply same changes as `get_run_by_id()`

**Alternative (simpler)**: Use column names directly in SELECT and access by name from dict.

---

### Step 3: Add Transaction to Migration 6

**Files to modify**: `backend/database.py`

1. **Wrap migration in transaction**:
```python
def _migration_006_normalize_runs_schema(conn: duckdb.DuckDBPyConnection) -> None:
    # ... early return checks ...
    
    try:
        conn.execute("BEGIN TRANSACTION")
        # ... all migration steps ...
        conn.execute("COMMIT")
    except Exception as e:
        conn.execute("ROLLBACK")
        raise
```

**Note**: DuckDB may auto-commit DDL, but explicit transaction is still best practice.

---

### Step 4: Add Transaction to record_run()

**Files to modify**: `backend/database.py`

1. **Wrap INSERTs in transaction**:
```python
def record_run(self, record: RunRecord) -> int:
    # ... validation ...
    
    try:
        self.conn.execute("BEGIN TRANSACTION")
        # ... all INSERT statements ...
        self.conn.commit()
        return run_id
    except Exception as e:
        self.conn.rollback()
        raise DatabaseError(...) from e
```

---

## Execution Order

1. **Step 1** (Remove run_parameters) - Simplest, reduces complexity
2. **Step 2** (Fix indices) - Prevents future bugs
3. **Step 3** (Migration transaction) - Safety improvement
4. **Step 4** (record_run transaction) - Data integrity

---

## Testing After Fixes

1. Run unit tests: `make unit`
2. Run integration tests: `make integration`
3. Test migration on fresh database
4. Test migration on existing database with data
5. Verify no data loss in migration

---

## Estimated Effort

- **Step 1**: 30 minutes (remove code)
- **Step 2**: 45 minutes (refactor get methods)
- **Step 3**: 15 minutes (add transaction wrapper)
- **Step 4**: 15 minutes (add transaction wrapper)

**Total**: ~2 hours

---

## Risk Assessment

- **Step 1**: LOW RISK - Removing unused table, no data loss
- **Step 2**: LOW RISK - Refactoring access pattern, same functionality
- **Step 3**: LOW RISK - Adding safety wrapper, no behavior change
- **Step 4**: LOW RISK - Adding safety wrapper, no behavior change

**Overall Risk**: LOW - All changes are improvements without changing core behavior.
