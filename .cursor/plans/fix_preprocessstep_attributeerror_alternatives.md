# Fix PreprocessStep AttributeError - Three Alternative Approaches

## Alternative 1: Simplest - Direct Fix (Recommended)

**Philosophy**: Fix the bug with minimal changes. The code already works correctly elsewhere (lines 46, 49), so just align these two lines.

**Changes**:
- Line 667: `step.get("name", "unknown")` → `step.name`
- Line 679: `step.get("enabled", True)` → `step.enabled`

**Pros**:
- Minimal change, lowest risk
- Consistent with existing code pattern
- Fast to implement and verify

**Cons**:
- No protection against future similar mistakes
- Assumes `PreprocessStep` always has these attributes

**Verification**: Run existing tests + manual variant test

---

## Alternative 2: Defensive - Add Runtime Safety

**Philosophy**: Fix the bug and add defensive checks to handle edge cases gracefully.

**Changes**:
1. Line 667: `step.get("name", "unknown")` → `getattr(step, "name", "unknown")`
2. Line 679: `step.get("enabled", True)` → `getattr(step, "enabled", True)`

**Pros**:
- Handles edge cases (missing attributes, wrong types)
- More resilient to future changes
- Still simple, just uses `getattr` instead of direct access

**Cons**:
- Masks potential type errors (might hide real bugs)
- Slightly more verbose

**Verification**: Run existing tests + test with malformed step objects

---

## Alternative 3: Architecturally Strong - Type Safety & Abstraction

**Philosophy**: Fix the bug and improve the codebase architecture to prevent this class of errors.

**Changes**:
1. Fix the immediate bug (lines 667, 679)
2. Add helper function to abstract step attribute access:
   ```python
   def _get_step_name(step: Any) -> str:
       """Get step name, handling both dict and PreprocessStep objects."""
       if hasattr(step, "name"):
           return step.name
       elif isinstance(step, dict):
           return step.get("name", "unknown")
       return "unknown"
   
   def _is_step_enabled(step: Any) -> bool:
       """Check if step is enabled, handling both dict and PreprocessStep objects."""
       if hasattr(step, "enabled"):
           return step.enabled
       elif isinstance(step, dict):
           return step.get("enabled", True)
       return True
   ```
3. Use helpers at lines 667, 679
4. Optionally: Add `TYPE_CHECKING` import and proper type hints to reduce `Any` usage

**Pros**:
- Prevents future similar bugs
- Supports both dict and object (backward compatibility if needed)
- Centralizes attribute access logic
- Improves maintainability
- Can add type hints without circular imports

**Cons**:
- More code changes
- More complex than needed for current issue
- Might be over-engineering if dict support isn't needed

**Verification**: Run existing tests + add unit tests for helper functions + test with both dict and object inputs

---

## Recommendation

**Use Alternative 1** unless:
- You need backward compatibility with dict-based steps → Use Alternative 3
- You want extra safety for production → Use Alternative 2
- You're doing broader refactoring → Use Alternative 3
