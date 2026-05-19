# Fix Code Review Issues

## Issues Found

### 1. **Unnecessary Dict Support** (preprocess_steps.py)
**Problem**: Helper functions support both dict and PreprocessStep, but codebase only uses PreprocessStep objects.
- `variant.preprocess_steps` is `list[PreprocessStep]` (variant.py:74)
- All variants create PreprocessStep objects (registry.py)
- No dict usage found in codebase

**Best Practice Violation**: YAGNI (You Aren't Gonna Need It) - adding complexity for unused features.

**Fix**: Simplify to direct attribute access (Alternative 1 from original plan).

### 2. **Unused TYPE_CHECKING Import** (preprocess_steps.py:14-15)
**Problem**: TYPE_CHECKING block has `pass` instead of importing PreprocessStep.
**Fix**: Either use it properly or remove it.

### 3. **Anti-Pattern: hasattr + isinstance** (preprocess_steps.py)
**Problem**: Using `hasattr()` + `isinstance()` instead of EAFP (Easier to Ask for Forgiveness than Permission).
**Best Practice**: Python prefers try/except over pre-checking attributes.

**Fix**: Use direct attribute access with try/except, or just direct access since we know the type.

### 4. **Batch File Indentation Error** (compare_variants.bat:90)
**Problem**: `del` command is incorrectly indented (should be at column 0).
**Fix**: Remove indentation.

## Simplest Fix Plan

### Step 1: Simplify Helper Functions (preprocess_steps.py)
Since steps are always PreprocessStep objects:
- Remove dict support from `_get_step_name()` and `_is_step_enabled()`
- Use direct attribute access: `step.name` and `step.enabled`
- Remove unnecessary type: ignore comments

### Step 2: Remove Unused TYPE_CHECKING (preprocess_steps.py)
Remove the unused TYPE_CHECKING block (lines 14-15).

### Step 3: Fix Batch File Indentation (compare_variants.bat:90)
Fix indentation of `del` command.

## Implementation

**File**: `backend/variants/preprocess_steps.py`
1. Replace helper functions with direct attribute access
2. Remove TYPE_CHECKING block
3. Update lines 704 and 716 to use `step.name` and `step.enabled` directly

**File**: `scripts/windows/compare_variants.bat`
1. Fix line 90 indentation

## Rationale

- **Simpler is better**: Direct attribute access matches existing code pattern (lines 83, 86)
- **No dict support needed**: Codebase analysis confirms only PreprocessStep objects are used
- **Follows YAGNI**: Don't add complexity for unused features
- **Consistent with codebase**: Other code uses `step.name` and `step.enabled` directly
