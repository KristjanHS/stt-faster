# Code Review: processed_folder Property Fix

## Changes Made in This Session

**File:** `backend/cli/transcription_commands.py` (lines 304-305)

**Change:** Fixed attempt to set read-only properties by directly accessing private attribute:
```python
# Before (broken):
processor.processed_folder = run_folder
processor.failed_folder = run_folder / "failed"

# After (current fix):
processor._file_mover_policy.processed_folder = run_folder
processor._file_mover_policy.failed_folder = run_folder / "failed"
```

## Best Practices Analysis

### Python Property Patterns

1. **Read-only properties** (current design):
   - Use `@property` without setter for immutable/delegated attributes
   - Good for: computed values, delegation, backward compatibility

2. **Settable properties**:
   - Use `@property` with `@property_name.setter` for controlled assignment
   - Good for: validation, encapsulation, controlled updates

3. **Direct attribute access**:
   - Use public attributes when no validation/control needed
   - Good for: simple data, performance-critical code

### Current Architecture

- `TranscriptionProcessor.processed_folder`: Read-only property delegating to `_file_mover_policy.processed_folder`
- `FileMoverPolicy.processed_folder`: Public attribute (no underscore)
- `TranscriptionProcessor._file_mover_policy`: Private attribute (single underscore)

### Issues with Current Fix

1. **Encapsulation violation**: Accesses private `_file_mover_policy` attribute
2. **Inconsistent pattern**: Other code uses `processor._output_base_dir` (private), but this is different
3. **Maintenance risk**: If `FileMoverPolicy` changes, this breaks silently

### Why Current Fix Works (But Isn't Ideal)

- ✅ **Functional**: `FileMoverPolicy.processed_folder` is a public attribute, so assignment works
- ✅ **Consistent with codebase**: Similar private attribute access exists elsewhere
- ❌ **Encapsulation**: Bypasses the property interface
- ❌ **Type safety**: No validation or type checking on assignment

## Recommended Solution

**Add property setters to `TranscriptionProcessor`** to maintain encapsulation while allowing controlled updates.

### Benefits:
- Maintains encapsulation (no private attribute access)
- Allows validation if needed in future
- Consistent with Python property patterns
- Type-safe and maintainable

### Implementation:
Add setters to `backend/processor.py`:
```python
@property
def processed_folder(self) -> Path:
    """Get the processed folder path (backward compatibility)."""
    return self._file_mover_policy.processed_folder

@processed_folder.setter
def processed_folder(self, value: Path) -> None:
    """Set the processed folder path."""
    self._file_mover_policy.processed_folder = Path(value)

@property
def failed_folder(self) -> Path:
    """Get the failed folder path (backward compatibility)."""
    return self._file_mover_policy.failed_folder

@failed_folder.setter
def failed_folder(self, value: Path) -> None:
    """Set the failed folder path."""
    self._file_mover_policy.failed_folder = Path(value)
```

Then update `backend/cli/transcription_commands.py` to use the properties:
```python
processor.processed_folder = run_folder
processor.failed_folder = run_folder / "failed"
```

## Validation

**Current fix is functional but not ideal.** It works because:
- `FileMoverPolicy.processed_folder` is a public attribute
- The codebase already uses similar private attribute access patterns
- No immediate breaking issues

**However, it violates encapsulation principles** and should be improved with property setters.
