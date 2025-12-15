# Fix Plan: Add Property Setters for processed_folder

## Problem
Current fix works but violates encapsulation by accessing private `_file_mover_policy` attribute.

## Solution
Add property setters to `TranscriptionProcessor` to allow controlled assignment while maintaining encapsulation.

## Steps

1. **Add property setters to `backend/processor.py`**
   - Add `@processed_folder.setter` method
   - Add `@failed_folder.setter` method
   - Ensure Path conversion for type safety

2. **Update `backend/cli/transcription_commands.py`**
   - Change from `processor._file_mover_policy.processed_folder = ...` 
   - To `processor.processed_folder = ...`
   - Same for `failed_folder`

3. **Run tests**
   - `make unit` - verify processor tests pass
   - `make integration` - verify integration tests pass
   - Verify multi-variant functionality works

## Files to Modify

1. `backend/processor.py` - Add property setters
2. `backend/cli/transcription_commands.py` - Use property setters instead of direct access

## Estimated Impact
- Low risk: Only affects multi-variant code path
- Backward compatible: Read-only access still works
- Improves encapsulation and maintainability
