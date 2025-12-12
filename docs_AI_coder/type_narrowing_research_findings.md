# Type Narrowing Research Findings

## Executive Summary

After researching Python type narrowing best practices, the recommended solution is to **extract the parameter resolution logic into a separate helper function** that returns non-optional types. This approach:
- ✅ Provides runtime safety (exceptions are handled in one place)
- ✅ Achieves natural type narrowing (return type is `float`, not `float | None`)
- ✅ Follows Python best practices (no `assert`, no `type: ignore`)
- ✅ Makes code more maintainable and testable

---

## Research Question Answers

### Q1: Python Type Narrowing Best Practices

**Answer**: The Python typing community recommends:

1. **Explicit conditional checks with `isinstance()` or `is not None`**
   - Type checkers (Pyright, mypy) recognize these patterns naturally
   - No need for `assert` statements
   - Works reliably regardless of optimization flags

2. **User-defined Type Guards (PEP 647)**
   - For complex validation logic
   - Use `TypeGuard` or `TypeIs` (Python 3.13+) to inform type checkers
   - Provides both runtime validation and type narrowing

3. **Avoid `assert` for production validation**
   - Can be disabled with `-O` flag
   - Security linters (Bandit) flag them
   - Not suitable for production code

**Key Insight**: Type checkers narrow types based on control flow (if statements, isinstance checks), not based on assignments that might fail.

---

### Q2: Error Handling vs Type Narrowing Trade-offs

**Answer**: The best approach is to **handle exceptions explicitly** rather than relying on post-assignment validation:

1. **Catch exceptions at the source**
   - If dictionary access might fail, wrap it in try-except
   - Raise appropriate exceptions (ValueError, KeyError) with clear messages
   - This provides both runtime safety and clear error messages

2. **Type narrowing happens naturally after exception handling**
   - If code reaches a point after try-except, type checkers know the operation succeeded
   - No need for explicit validation checks that Pyright thinks are "unnecessary"

**Key Insight**: Exception handling and type narrowing work together - handle exceptions properly, and type narrowing follows naturally.

---

### Q3: Refactoring Patterns for Optional Parameter Resolution

**Answer**: The recommended pattern is to **extract resolution into a helper function**:

**Pattern: Resolver Function**
```python
def resolve_loudnorm_params(
    target_i: float | None,
    target_tp: float | None,
    target_lra: float | None,
    loudnorm_preset: str,
) -> tuple[float, float, float]:
    """Resolve loudnorm parameters from args or preset config.
    
    Returns:
        Tuple of (target_i, target_tp, target_lra) as floats.
        
    Raises:
        ValueError: If parameters cannot be resolved from preset.
        KeyError: If required keys are missing from preset.
    """
    if target_i is None or target_tp is None or target_lra is None:
        preset = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
        try:
            if target_i is None:
                target_i = float(preset["I"])
            if target_tp is None:
                target_tp = float(preset.get("TP", -2.0))
            if target_lra is None:
                target_lra = float(preset["LRA"])
        except KeyError as e:
            raise ValueError(
                f"Missing required loudnorm parameter in preset '{loudnorm_preset}': {e}"
            ) from e
    
    # Type narrowing: Pyright knows these are float after the if block
    # Runtime safety: If any are still None, we'd catch it, but the logic ensures they're set
    return target_i, target_tp, target_lra
```

**Benefits**:
- Return type is `tuple[float, float, float]` - no optional types
- Exception handling is centralized
- Type narrowing happens naturally (no explicit checks needed)
- Function is testable in isolation
- Main function becomes cleaner

**Usage**:
```python
def _loudnorm_2pass_linear(...) -> StepMetrics:
    # Resolve parameters - type narrowing happens automatically
    target_i, target_tp, target_lra = resolve_loudnorm_params(
        target_i, target_tp, target_lra, loudnorm_preset
    )
    
    # Now target_i, target_tp, target_lra are guaranteed to be float
    # No validation checks needed - type checker knows this
    filter_graph = f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
    ...
```

**Key Insight**: Extract complex resolution logic into functions with non-optional return types. This makes type narrowing work naturally.

---

### Q4: Pyright Type Narrowing After Exception-Prone Operations

**Answer**: Pyright assumes assignments succeed unless there's explicit exception handling:

1. **Pyright's behavior**:
   - If you write `x = dict["key"]`, Pyright assumes it succeeds
   - It doesn't automatically account for potential KeyError
   - Type narrowing happens based on control flow, not exception possibilities

2. **Best practice**:
   - Wrap exception-prone operations in try-except
   - Handle exceptions explicitly
   - After try-except, Pyright knows the operation succeeded (if no exception was raised)

3. **Type guards can help**:
   - For complex validation, use `TypeGuard` functions
   - These inform Pyright about type narrowing after validation

**Key Insight**: Pyright doesn't track exception possibilities automatically. Handle exceptions explicitly, and type narrowing works.

---

### Q5: Production-Ready Validation Patterns

**Answer**: The best patterns are:

1. **Explicit if-raise checks** (what we tried, but can be improved)
   ```python
   if value is None:
       raise ValueError("value must be set")
   # Type narrowing happens here
   ```

2. **Type guard functions** (for complex cases)
   ```python
   from typing import TypeGuard
   
   def is_valid_float(value: float | None) -> TypeGuard[float]:
       return value is not None
   
   if not is_valid_float(target_i):
       raise ValueError("target_i must be set")
   # Type narrowing happens here
   ```

3. **Resolver/helper functions** (recommended for our case)
   - Extract resolution logic
   - Return non-optional types
   - Handle exceptions internally
   - Type narrowing happens naturally

**Key Insight**: The best solution depends on complexity. For our case (resolving from multiple sources), a resolver function is cleanest.

---

## Recommended Solution

### Implementation

**Step 1: Create resolver function**

```python
def _resolve_loudnorm_params(
    target_i: float | None,
    target_tp: float | None,
    target_lra: float | None,
    loudnorm_preset: str,
) -> tuple[float, float, float]:
    """Resolve loudnorm parameters from function args or preset config.
    
    Args:
        target_i: Explicit I parameter, or None to use preset
        target_tp: Explicit TP parameter, or None to use preset
        target_lra: Explicit LRA parameter, or None to use preset
        loudnorm_preset: Preset name to use if parameters are None
        
    Returns:
        Tuple of (target_i, target_tp, target_lra) as floats.
        
    Raises:
        ValueError: If required parameters cannot be resolved from preset.
    """
    if target_i is None or target_tp is None or target_lra is None:
        preset = PreprocessConfig.get_loudnorm_preset_config(loudnorm_preset)
        try:
            if target_i is None:
                target_i = float(preset["I"])
            if target_tp is None:
                target_tp = float(preset.get("TP", -2.0))
            if target_lra is None:
                target_lra = float(preset["LRA"])
        except KeyError as e:
            raise ValueError(
                f"Missing required loudnorm parameter in preset '{loudnorm_preset}': {e}"
            ) from e
    
    # All parameters are guaranteed to be float at this point
    # Type narrowing happens naturally - no explicit checks needed
    return target_i, target_tp, target_lra
```

**Step 2: Use in main function**

```python
def _loudnorm_2pass_linear(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int,
    target_channels: int,
    loudnorm_preset: str = "default",
    *,
    target_i: float | None = None,
    target_tp: float | None = None,
    target_lra: float | None = None,
) -> StepMetrics:
    """2-pass loudnorm in linear mode: measure, then apply file-wide gain."""
    import json
    import ffmpeg  # type: ignore[import-untyped]
    from backend.preprocess.config import PreprocessConfig

    start = time.time()

    try:
        # Resolve parameters - returns non-optional floats
        target_i, target_tp, target_lra = _resolve_loudnorm_params(
            target_i, target_tp, target_lra, loudnorm_preset
        )
        
        # Type narrowing: Pyright knows these are float
        # Runtime safety: Resolver function handles exceptions
        # No validation checks needed!
        
        # ----- 1. First pass: measure loudness stats -----
        first_pass_filter = f"aresample=resampler=soxr:osr={target_sample_rate},loudnorm=print_format=json"
        # ... rest of function uses target_i, target_tp, target_lra as floats
```

### Why This Solution is Best

1. ✅ **No `assert` statements** - Uses explicit exception handling
2. ✅ **No `type: ignore` comments** - Type narrowing works naturally
3. ✅ **Runtime safety** - Exceptions handled in resolver function
4. ✅ **Type safety** - Return type is `tuple[float, float, float]`
5. ✅ **Maintainable** - Resolution logic is isolated and testable
6. ✅ **Follows Python best practices** - Explicit error handling, clear separation of concerns
7. ✅ **Reusable** - Can be used by other functions that need the same resolution

---

## Alternative Solutions (Not Recommended)

### Option A: Keep current approach with type: ignore
- ❌ Feels like a workaround
- ❌ Pyright complains about unnecessary checks
- ⚠️ Works but not elegant

### Option B: Use assert statements
- ❌ Can be disabled with `-O` flag
- ❌ Security linter flags them
- ❌ Not production-ready

### Option C: Use TypeGuard function
- ⚠️ More complex than needed
- ⚠️ Still requires explicit checks
- ✅ Would work but overkill for this case

---

## Conclusion

The **resolver function pattern** is the recommended solution because it:
- Solves the type narrowing problem naturally
- Provides runtime safety through explicit exception handling
- Follows Python best practices
- Makes code more maintainable and testable
- Eliminates the need for workarounds (`assert`, `type: ignore`)

This pattern is commonly used in Python codebases for resolving optional parameters from multiple sources.

