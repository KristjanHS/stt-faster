# Simple Fix Plan: Problem 9 Session Code Issues

## Critical Issue Found

**`get_run_by_id()` method missing type conversion** - Same bug that was fixed in `get_run_history()` but not applied here.

---

## Fix Plan (Priority Order)

### 1. Fix `get_run_by_id()` Type Conversion (CRITICAL - 5 min)

**File**: `backend/database.py`

**Change at line ~1763**:
```python
# BEFORE:
for param_row in params_cursor.fetchall():
    category, name, value, _value_type = param_row  # ❌ Discards type
    if category == "preprocess" and name == "profile":
        result["preprocess_profile"] = value
    elif category == "preprocess":
        result[name] = value
    elif category == "transcription":
        result[name] = value
    elif category == "model":
        if name not in result or result[name] is None:
            result[name] = value

# AFTER:
for param_row in params_cursor.fetchall():
    category, name, value, value_type = param_row  # ✅ Use type
    converted_value = _convert_parameter_value(value, value_type)  # ✅ Convert
    if category == "preprocess" and name == "profile":
        result["preprocess_profile"] = converted_value
    elif category == "preprocess":
        result[name] = converted_value
    elif category == "transcription":
        result[name] = converted_value
    elif category == "model":
        if name not in result or result[name] is None:
            result[name] = converted_value
```

**Why**: Ensures consistency with `get_run_history()` and returns correct types (float/int/bool instead of strings).

---

### 2. Add Error Handling (RECOMMENDED - 10 min)

**File**: `backend/database.py`

**Change `_convert_parameter_value()` function (lines ~215-230)**:
```python
def _convert_parameter_value(value: str, value_type: str) -> Any:
    """Convert parameter value string back to original type."""
    try:
        if value_type == "float":
            return float(value)
        elif value_type == "int":
            return int(value)
        elif value_type == "bool":
            return value.lower() in ("true", "1", "yes", "on")
        elif value_type == "str":
            return value
        else:
            # Fallback: try to infer type
            try:
                if "." in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value
    except (ValueError, TypeError) as e:
        LOGGER.warning(
            "Failed to convert parameter value '%s' (type: %s): %s. "
            "Returning original string.",
            value, value_type, e
        )
        return value  # Return original string as fallback
```

**Why**: Prevents crashes on invalid database data and provides logging for debugging.

---

## Testing

After fix #1:
- Run: `make unit` (should pass)
- Run: `make integration` (should pass)
- Verify: `get_run_by_id()` returns `patience` as `float(1.0)` not `"1.0"`

After fix #2:
- Add test case with invalid value (e.g., `value="abc"`, `value_type="float"`)
- Verify: Warning logged, original string returned

---

## Summary

**Must Fix**: Apply type conversion to `get_run_by_id()` (5 minutes)
**Should Fix**: Add error handling to conversion function (10 minutes)

**Total Time**: ~15 minutes
