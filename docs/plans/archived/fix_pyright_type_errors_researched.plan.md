# Fix Pyright Type Errors - Researched Plan

## Problem Description

The variant step system uses generic types (`Step[ConfigT]`) and a registry pattern. Pyright reports 174 type errors, primarily due to:

1. **Generic type parameter loss**: Methods use `StepConfig` instead of `ConfigT`, breaking type inference
2. **Registry typing limitations**: `StepRegistry._registry: dict[str, type[Step]]` loses generic information
3. **Private function imports**: Helpers prefixed with `_` are imported across modules
4. **Abstract method return types**: `get_default_config()` returns `StepConfig` instead of `ConfigT`

**Key Insight from Research**: The Protocol vs ABC debate is a red herring. The real issue is preserving generic type parameters (`ConfigT`) throughout the type system, not the base type mechanism.

## Research-Based Solutions

### Phase 1: Fix Generic Type Parameter Preservation (Critical - Fixes ~100+ errors)

**Root Cause**: Methods use `StepConfig` (base type) instead of `ConfigT` (generic parameter), causing Pyright to lose type information.

**Solution**: Update all method signatures to use `ConfigT` instead of `StepConfig`.

**Files to modify**:
- `backend/variants/steps.py`

**Changes**:
1. Update `get_default_config()` abstract method to return `ConfigT` instead of `StepConfig` (line 192)
2. Verify all concrete step classes already return their specific config type (they should)
3. Keep `StepConfig` as Protocol - no need to change to ABC

**Expected impact**: Fixes majority of "Cannot access attribute" errors because Pyright can now infer specific config types.

### Phase 2: Fix Registry Typing with "Typed-at-Boundary" Pattern

**Root Cause**: Python's typing system can't express "mapping from string keys to a family of generic types" with full inference.

**Solution**: Use "typed-at-the-boundary" pattern - keep registry internally untyped, regain type safety at API boundary with casts.

**Files to modify**:
- `backend/variants/steps.py` (lines 480-501)

**Changes**:
1. Change `_registry` type to `dict[str, type[Step[Any]]]` (explicitly Any-typed)
2. Update `create_step()` to use cast at boundary:
   ```python
   @classmethod
   def create_step(cls, step_type: str, config: StepConfig | None = None) -> Step:
       step_class = cls.get_step_class(step_type)
       return step_class(config)  # type: ignore[arg-type, call-overload]
   ```
3. Add comment explaining why cast is necessary (dynamic registry pattern)
4. Keep `get_step_class()` returning `type[Step]` with cast

**Expected impact**: Fixes ~30 registry-related errors while admitting the dynamic nature of the registry.

### Phase 3: Fix Private Function Imports

**Root Cause**: Pyright's `reportPrivateUsage` flags `_private` functions used outside their module.

**Solution**: Make helpers public (they're used across modules, so they're not truly private).

**Files to modify**:
- `backend/variants/preprocess_steps.py` (rename functions)
- `backend/variants/steps.py` (update imports)

**Changes**:
1. Rename `_loudnorm_only` → `loudnorm_only` (line 88)
2. Rename `_volume_with_limiter` → `volume_with_limiter` (if exists)
3. Rename `_peak_normalize_2pass` → `peak_normalize_2pass` (if exists)
4. Rename `_loudnorm_with_highpass` → `loudnorm_with_highpass` (if exists)
5. Rename `_dynaudnorm_only` → `dynaudnorm_only` (if exists)
6. Rename `_highlow_aform_loudnorm` → `highlow_aform_loudnorm` (if exists)
7. Rename `_highlow_nosampl_loudnorm` → `highlow_nosampl_loudnorm` (if exists)
8. Rename `_aresampl_loudnorm_fixed` → `aresampl_loudnorm_fixed` (if exists)
9. Rename `_aresampl_loudnorm_fixed2` → `aresampl_loudnorm_fixed2` (if exists)
10. Rename `_loudnorm_2pass_linear` → `loudnorm_2pass_linear` (if exists)
11. Rename `_limiter_only` → `limiter_only` (if exists)
12. Rename `_sox_peak_normalize` → `sox_peak_normalize` (if exists)
13. Rename `_compressor_with_limiter` → `compressor_with_limiter` (if exists)
14. Rename `_dynaudnorm_conservative` → `dynaudnorm_conservative` (if exists)
15. Update all imports in `backend/variants/steps.py`

**Alternative**: If keeping private, move to shared internal module like `backend/variants/preprocess/_helpers.py` with public functions inside.

**Expected impact**: Fixes ~15 import-related errors and 1 unused function warning.

### Phase 4: Add Rule-Specific Type Ignores (Minimal, Surgical)

**Root Cause**: Some dynamic behavior (registry) can't be perfectly typed.

**Solution**: Add rule-specific `# pyright: ignore[...]` comments at registry boundary only.

**Files to modify**:
- `backend/variants/steps.py` (registry methods)

**Changes**:
1. Add `# pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]` to `create_step()` cast line
2. Document why ignore is necessary (dynamic registry pattern)
3. Avoid blanket ignores - be specific about rules

**Expected impact**: Suppresses remaining unavoidable errors at registry boundary.

## Implementation Order

1. **Phase 1** (Generic Type Parameters) - **CRITICAL** - Fixes most errors
2. **Phase 2** (Registry Typing) - Improves type safety at boundary
3. **Phase 3** (Private Imports) - Clean up API surface
4. **Phase 4** (Type Ignores) - Final cleanup for unavoidable cases

## Testing Strategy

After each phase:
1. Run `make pre-commit` to count remaining Pyright errors
2. Run `make unit` to ensure no runtime regressions
3. Run `make integration` to verify end-to-end functionality
4. Track error count reduction per phase

## Success Criteria

- Zero Pyright errors in `backend/variants/steps.py`
- Zero Pyright errors in `backend/variants/preprocess_steps.py`
- All unit tests pass (111/111)
- All integration tests pass (13/13)
- No runtime behavior changes
- Minimal use of type ignore comments (only at registry boundary)

## Key Research Insights Applied

1. **Protocol is fine** - Keep `StepConfig` as Protocol, fix generic parameter usage
2. **Typed-at-boundary** - Accept dynamic registry, regain safety at API boundary
3. **Make helpers public** - If used across modules, they're not private
4. **Preserve ConfigT** - Abstract methods must return `ConfigT`, not `StepConfig`
5. **Surgical ignores** - Use rule-specific ignores only where necessary

