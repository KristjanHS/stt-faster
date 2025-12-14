# Fix Pyright Type Ignore Issues

## Analysis of Current Fixes

### Issues Found

1. **Unnecessary Type Ignores on Lists** (generate_variant_report.py)
   - Problem: Lists are declared as `[]` without type annotations, then we ignore warnings on `.append()`
   - Best Practice: Declare lists with proper types (e.g., `list[float]`)
   - Impact: Lines 158-159, 173, 256, 261, 270, 275, 315, 317, 327, 333, 368

2. **Unnecessary Dict Support** (preprocess_steps.py)
   - Problem: Helper functions support both dict and PreprocessStep, but codebase only uses PreprocessStep
   - Best Practice: Use direct attribute access (as noted in code review plan)
   - Impact: Lines 28-59 (entire helper functions)

3. **Untyped Dict Parameters** (check_parameter_completeness.py)
   - Problem: Function parameter is `dict` instead of `dict[str, Any]`
   - Best Practice: Use explicit type annotations
   - Impact: Line 373

## Best Practices Validation

### ✅ Acceptable Uses of Type Ignore:
- JSON data from external sources (`dict[str, Any]` is appropriate)
- Third-party libraries without type stubs (ffmpeg-python in preprocess_steps.py)
- Dynamic/registry patterns where types can't be fully expressed

### ❌ Problematic Uses:
- Lists without type annotations (should be `list[float]`, `list[str]`, etc.)
- Unnecessary dict support when type is known
- Untyped function parameters

## Simplest Fix Plan

### Step 1: Add Type Annotations to Lists (generate_variant_report.py)
Replace untyped list declarations with typed ones:
- `no_speech_probs = []` → `no_speech_probs: list[float] = []`
- `avg_logprobs = []` → `avg_logprobs: list[float] = []`
- `words_in_range = []` → `words_in_range: list[str] = []`
- `silence_words = []` → `silence_words: list[str] = []`
- `unique_words = set()` → `unique_words: set[str] = set()`

Remove type: ignore comments from `.append()`, `.extend()`, `.add()` calls on these lists.

### Step 2: Remove Unnecessary Helper Functions (preprocess_steps.py)
As per code review plan, remove dict support:
- Delete `_get_step_name()` and `_is_step_enabled()` functions
- Replace calls with direct attribute access: `step.name`, `step.enabled`
- Remove all type: ignore comments from these functions

### Step 3: Fix Function Parameter Type (check_parameter_completeness.py)
Change `run_details: dict` → `run_details: dict[str, Any]` at line 373

Keep type: ignore on `.get()` calls for JSON data (acceptable for external JSON).

## Rationale

- **Lists**: Type annotations are simple and eliminate need for type: ignore
- **PreprocessStep**: Codebase analysis confirms only PreprocessStep objects are used (YAGNI)
- **Dict parameters**: Explicit typing improves code clarity
- **JSON data**: `dict[str, Any]` with type: ignore on `.get()` is acceptable for external JSON
