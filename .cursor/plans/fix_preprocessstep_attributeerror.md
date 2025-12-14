# Fix PreprocessStep AttributeError

## Problem

Variants 22-32 fail with `AttributeError: 'PreprocessStep' object has no attribute 'get'` because the code treats `PreprocessStep` dataclass objects as dictionaries.

**Root Cause**: Two lines in `backend/variants/preprocess_steps.py` use dictionary-style `.get()` method:
- Line 667: `step.get("name", "unknown")` 
- Line 679: `step.get("enabled", True)`

**Why it fails**: `PreprocessStep` is a dataclass with required attributes `name: str` and `enabled: bool`, not a dictionary.

## Solution

Replace dictionary access with direct attribute access (consistent with existing code at lines 46, 49).

## Implementation

**File**: `backend/variants/preprocess_steps.py`

1. **Line 667**: Change `step.get("name", "unknown")` → `step.name`
2. **Line 679**: Change `step.get("enabled", True)` → `step.enabled`

## Verification

1. Run existing unit tests: `make unit`
2. Test one failing variant manually to confirm fix
3. Run pre-commit checks: `make pre-commit`

No new tests needed - existing tests should cover this once fixed.
