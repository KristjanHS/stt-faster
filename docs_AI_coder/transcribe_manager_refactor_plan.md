# Architectural Refactoring Plan: transcribe_manager.py

## Problem
`transcribe_manager.py` exceeds the 300-line limit (currently 397 lines) due to multi-variant runner functionality. The added code is CLI orchestration, not business logic.

## Current Structure Analysis
- **Total lines**: 397
- **Helper functions**: ~70 lines (`_get_git_commit_hash`, `_get_variant_overrides`, `_configure_logging`)
- **Single-variant processing**: ~80 lines (lines 174-253)
- **Multi-variant processing**: ~97 lines (lines 254-351)
- **UI/Display functions**: ~60 lines (Rich table creation)
- **Parser/CLI setup**: ~100 lines
- **Main/status commands**: ~50 lines

---

## Alternative 1: Extract Multi-Variant Logic (Simplest)
**Effort**: Low | **Risk**: Low | **Lines saved**: ~97

### Approach
Extract the multi-variant runner logic into a separate module in `backend/cli/` or `scripts/cli_helpers.py`.

### Changes
1. Create `backend/cli/multi_variant_runner.py`:
   - Move multi-variant processing loop (lines 254-351)
   - Move `_get_git_commit_hash()` helper
   - Move `_get_variant_overrides()` helper
   - Move `_create_run_meta()` function

2. Update `transcribe_manager.py`:
   - Import `run_multi_variant_sweep()` from new module
   - Replace multi-variant block with function call (~10 lines)

### Pros
- Minimal changes
- Keeps single-variant logic in script (backward compatible)
- Clear separation of concerns
- Easy to test multi-variant logic independently

### Cons
- Still close to 300-line limit
- Doesn't address future growth
- Some duplication between single/multi-variant paths

### Estimated Result
- `transcribe_manager.py`: ~300 lines
- New file: ~120 lines

---

## Alternative 2: Extract All Processing Logic (Moderate)
**Effort**: Medium | **Risk**: Medium | **Lines saved**: ~180

### Approach
Extract both single and multi-variant processing into a dedicated CLI module, keeping only argument parsing and command dispatch in the script.

### Changes
1. Create `backend/cli/transcription_commands.py`:
   - Move `cmd_process()` function (entire processing logic)
   - Move `_get_git_commit_hash()`, `_get_variant_overrides()`, `_configure_logging()`
   - Move UI display helpers (Rich table creation functions)
   - Move variant number parsing logic

2. Create `backend/cli/ui.py` (optional):
   - Rich table/formatter functions
   - Display helpers for summaries and stats

3. Update `transcribe_manager.py`:
   - Keep only: `create_parser()`, `cmd_status()`, `main()`
   - Import `cmd_process` from `backend/cli/transcription_commands.py`

### Pros
- Script becomes thin wrapper (~150 lines)
- All processing logic in backend (testable)
- Clear separation: CLI parsing vs. business logic
- Easier to add new commands
- Better testability

### Cons
- More files to maintain
- Requires moving some Rich UI code to backend
- Slightly more complex import structure

### Estimated Result
- `transcribe_manager.py`: ~150 lines
- `backend/cli/transcription_commands.py`: ~200 lines
- `backend/cli/ui.py`: ~60 lines (optional)

---

## Alternative 3: Full CLI Framework (Most Sustainable)
**Effort**: High | **Risk**: Medium-High | **Lines saved**: ~250+

### Approach
Create a proper CLI framework with command classes, following patterns from modern CLI libraries (like Click or Typer structure).

### Changes
1. Create `backend/cli/base.py`:
   - Base `Command` class with `execute()` method
   - Shared helpers: logging, git hash, variant parsing
   - Common UI utilities

2. Create `backend/cli/commands/process.py`:
   - `ProcessCommand` class
   - Single-variant processing method
   - Multi-variant processing method
   - Variant selection/validation logic

3. Create `backend/cli/commands/status.py`:
   - `StatusCommand` class
   - Status display logic

4. Create `backend/cli/ui.py`:
   - Rich formatting utilities
   - Table builders
   - Summary generators

5. Update `transcribe_manager.py`:
   - Minimal: parser setup, command registry, dispatch
   - ~100 lines total

### Pros
- Highly maintainable and extensible
- Easy to add new commands
- Clear separation of concerns
- Testable command classes
- Follows established patterns
- Future-proof for growth

### Cons
- Most refactoring effort
- More files to create
- Learning curve for contributors
- May be overkill for current needs

### Estimated Result
- `transcribe_manager.py`: ~100 lines
- `backend/cli/base.py`: ~80 lines
- `backend/cli/commands/process.py`: ~180 lines
- `backend/cli/commands/status.py`: ~50 lines
- `backend/cli/ui.py`: ~60 lines

---

## Recommendation Matrix

| Criteria | Alt 1 | Alt 2 | Alt 3 |
|----------|-------|-------|-------|
| **Effort** | ⭐ Low | ⭐⭐ Medium | ⭐⭐⭐ High |
| **Risk** | ⭐ Low | ⭐⭐ Medium | ⭐⭐ Medium |
| **Maintainability** | ⭐⭐ Good | ⭐⭐⭐ Better | ⭐⭐⭐⭐ Best |
| **Testability** | ⭐⭐ Good | ⭐⭐⭐ Better | ⭐⭐⭐⭐ Best |
| **Future-proof** | ⭐ Limited | ⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **Time to implement** | 30 min | 2-3 hours | 1 day |

---

## Decision Guide

**Choose Alternative 1 if:**
- You need a quick fix to pass the test
- You don't expect much future CLI growth
- You want minimal risk

**Choose Alternative 2 if:**
- You want a good balance of effort vs. maintainability
- You expect moderate CLI growth
- You want better testability without over-engineering

**Choose Alternative 3 if:**
- You plan to add more CLI commands
- You want a sustainable, scalable architecture
- You have time for proper refactoring
- You want to follow best practices

---

## Implementation Notes

### Common to All Alternatives
- All alternatives maintain backward compatibility
- No changes to public API
- All tests should continue to pass
- Windows batch files continue to work

### Testing Strategy
- Unit tests for extracted functions
- Integration tests for CLI commands
- Ensure `test_scripts_are_thin_wrappers` passes after refactoring

### Migration Path
1. Create new module(s) with extracted code
2. Update imports in `transcribe_manager.py`
3. Run tests to verify functionality
4. Remove old code from script
5. Verify line count meets threshold

