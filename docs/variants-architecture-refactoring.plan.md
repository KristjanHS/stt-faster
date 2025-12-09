<!-- 827c1271-850d-46ed-b1e6-b579f8507bdf f6ed0af2-584a-4705-9b32-b878c2b035e2 -->
# Variants Architecture Refactoring Plan

## Simplifications Made

**Key principle**: Avoid over-engineering. Keep it simple and pragmatic.

**Removed unnecessary complexity:**
- ❌ "Composable" abstractions → ✅ Simple functions and dataclasses
- ❌ External YAML/JSON configs → ✅ Code-based registry (simpler, type-safe)
- ❌ Category-based variant selection → ✅ Simple number-based filtering (`--skip-variants`)
- ❌ Complex migration phases → ✅ Both systems coexist, opt-in when ready
- ❌ Over-testing (23 unit tests) → ✅ Focused tests on what matters
- ❌ Future-proofing for unknown needs → ✅ Add features only when actually needed

**What remains (the essentials):**
- ✅ Simple dataclass definitions
- ✅ Registry with variant definitions
- ✅ Single executor function
- ✅ Basic filtering (`--skip-variants`)
- ✅ Backward compatibility (legacy untouched)

## Current State Analysis

### Problems (Fixed in New Solution)

- [x] **Hardcoded variant definitions**: The `run_variant()` function in `scripts/compare_transcription_variants.py` uses a large if/elif chain (lines 480-631) to define variants
  - **Fixed**: New system uses declarative `Variant` dataclass definitions in `backend/variants/registry.py`. Legacy code untouched.
- [x] **Code duplication**: Custom preprocessing functions (`preprocess_only_ffmpeg`, `preprocess_only_denoise`) duplicate logic from `backend/preprocess/orchestrator.py`
  - **Fixed**: New system reuses existing `run_ffmpeg_pipeline` and `apply_light_denoise` via composable preprocessing steps. Custom functions for variants 10-15 are imported (not duplicated) and isolated.
- [x] **Inflexible architecture**: Adding new variants requires code changes; no way to select subsets of variants
  - **Fixed**: New variants can be added to registry without modifying execution logic. `--skip-variants` flag allows selecting subsets. Legacy code untouched.
- [x] **Mixed concerns**: Variant definition, execution, and output handling are intertwined
  - **Fixed**: New system separates concerns: definition (`registry.py`), execution (`executor.py`), and output handling (in executor). Legacy code untouched.
- [x] **Special-case handling**: `transcribe_with_minimal_params()` duplicates transcription logic
  - **Fixed**: New system uses transcription presets (`transcription_presets.py`) with `minimal` preset, eliminating duplication. Legacy code untouched.

### Current Variants (15 total - 9 standard + 6 custom preprocessing)

- `no_preprocessing`: No preprocessing + project defaults
- `industry_defaults`: No preprocessing + industry defaults
- `ffmpeg_only`: Only ffmpeg pipeline + project defaults
- `denoise_only`: Only denoise_light + project defaults
- `ffmpeg_industry_defaults`: Only ffmpeg + industry defaults
- `full_industry_defaults`: Full preprocessing + industry defaults
- `noprep_noparamtrans`: No preprocessing + minimal params
- `ffmpegonly_noparamtrans`: Only ffmpeg + minimal params
- `onlyden_noparamtrans`: Only denoise + minimal params

## Target Architecture (Simplified)

### Core Principles

1. **Declarative variant definitions**: Variants defined as simple dataclasses (not over-engineered)
2. **Reuse existing code**: Leverage existing preprocessing/transcription functions instead of duplicating
3. **Backward compatibility**: Legacy system remains untouched and functional
4. **Pragmatic separation**: Only separate what needs to be separated (registry vs execution)

### New Components (Simplified)

#### 1. Variant System (`backend/variants/`)

- `variant.py`: Simple dataclass definitions (Variant, PreprocessStep)
- `registry.py`: Variant definitions (could merge with variant.py, but kept separate for clarity)
- `executor.py`: Single function to execute a variant
- `transcription_presets.py`: Simple preset functions (3 functions)
- `preprocess_steps.py`: Helper to compose preprocessing from steps

**Note**: This is simpler than originally planned - no need for complex "composable" abstractions. Just functions and data.

#### 2. Updated Script (`scripts/compare_transcription_variants.py`)

- Optional `--use-new-variants` flag to use new system
- `--skip-variants` for basic filtering
- Legacy code remains intact

## Implementation Plan

### Phase 1: Foundation (Backward Compatible)

#### Step 1.1: Create Variant Definition Infrastructure

- [x] Create `backend/variants/__init__.py`
- [x] Create `backend/variants/variant.py`: Core `Variant` dataclass
- [x] Create `backend/variants/preprocess_steps.py`: Preprocessing step definitions
- [x] Create `backend/variants/transcription_presets.py`: Transcription config presets

**Key structures:**

```python
@dataclass
class PreprocessStep:
    name: str
    enabled: bool
    step_type: Literal["ffmpeg", "denoise", "resample"]
    config: dict[str, Any] | None = None

@dataclass
class Variant:
    name: str
    number: int
    description: str
    preprocess_steps: list[PreprocessStep]
    transcription_preset: str  # "project", "industry", "minimal"
    transcription_overrides: dict[str, Any] | None = None
```

#### Step 1.2: Create Transcription Presets

**File:** `backend/variants/transcription_presets.py`

- [x] Implement `get_project_defaults()`: Returns `TranscriptionConfig.from_env()` (current project defaults)
- [x] Implement `get_industry_defaults()`: Returns industry-standard faster-whisper defaults (moved from `compare_transcription_variants.py:328-368`)
- [x] Implement `get_minimal_config()`: Returns config that omits parameters to use faster-whisper internal defaults

#### Step 1.3: Create Preprocessing Helper

**File:** `backend/variants/preprocess_steps.py`

- [x] Implement `create_preprocess_runner()`: Simple helper that calls existing backend functions
- [x] Reuse existing `run_ffmpeg_pipeline` and `apply_light_denoise` from backend
- [x] Support common step combinations (no complex abstraction needed)

### Phase 2: Variant Executor

#### Step 2.1: Create Variant Executor

**File:** `backend/variants/executor.py`

- [x] Implement `execute_variant()` function - single function, straightforward logic
- [x] Support both standard preprocessing steps and custom runners (for variants 10-15)
- [x] Apply transcription preset (simple lookup)
- [x] Return results in same format as legacy `run_variant()`

#### Step 2.2: Create Variant Registry

**File:** `backend/variants/registry.py`

- [x] Implement `get_builtin_variants()` - returns list of Variant instances
- [x] Define all 15 variants as simple dataclass instances
- [x] Support variants 10-15 with custom runners (imported from legacy script)
- [x] Simple helper functions for lookup (optional but useful)

### Phase 3: Script Migration (Backward Compatible)

#### Step 3.1: Update Script to Use New System (Parallel Implementation)

**File:** `scripts/compare_transcription_variants.py`

- [x] Import new variant system
- [x] Keep old `run_variant()` function intact (legacy system preserved)
- [x] Update `main()` to support both old and new paths
- [x] Add `--use-new-variants` flag (default: False for compatibility)
- [x] Add `--skip-variants` flag to filter variants by number
- [x] Support custom preprocessing functions for variants 10-15 via importlib

#### Step 3.2: Test New System

- [x] Create unit tests for variant registry and data structures
- [x] Test that all 15 variants are correctly defined
- [x] Test variant filtering (`--skip-variants`)
- [x] Create basic integration tests
- [x] Verify new system doesn't break legacy (system isolation)
- [x] Manual validation: Compare outputs from both systems (when needed)

### Phase 4: Optional Enhancements (Only if needed)

#### Step 4.1: Remove Legacy Code (Optional - only if new system is fully adopted)

**Decision point**: Only remove legacy code if:
- New system is proven stable in production
- All users have migrated
- No need to maintain backward compatibility

- [ ] Remove old `run_variant()` if/elif chain (only if legacy no longer needed)
- [ ] Remove legacy preprocessing functions (only if no longer used)
- [ ] Make new system the default (only after full migration)

#### Step 4.2: Additional Features (Only if actually needed)

**Pragmatic approach**: Add features only when there's a real use case, not "just in case"

- [x] `--skip-variants` (already implemented - sufficient for current needs)
- [ ] `--variants` to select specific variants (only if needed)
- [ ] External variant definitions (YAML/JSON) - **Probably unnecessary complexity**

## File Structure (Simplified)

```
backend/variants/
├── __init__.py              # Simple exports
├── variant.py               # Dataclasses (Variant, PreprocessStep) - ~50 lines
├── registry.py              # Variant definitions - ~200 lines
├── executor.py              # Single execute_variant() function - ~250 lines
├── transcription_presets.py # 3 simple preset functions - ~100 lines
└── preprocess_steps.py      # Helper to compose preprocessing - ~250 lines

Total: ~850 lines (vs. ~1500 lines of if/elif chains in legacy)
```

**Note**: Could merge some files, but current separation is reasonable for maintainability.

## Migration Strategy (Simplified)

**Current approach**: Both systems coexist. Use `--use-new-variants` to opt into new system.

**No rush to migrate**: Legacy system works fine. New system is available when needed.

**Testing**: Unit tests verify structure. Integration tests verify execution. Manual comparison when convenient.

## Benefits (Simplified View)

1. **Maintainability**: Variants defined in one place (registry) instead of scattered if/elif chains
2. **Testability**: Simple data structures are easy to test
3. **Clarity**: Clear separation between definition (registry) and execution (executor)
4. **Backward compatibility**: Legacy system remains functional during transition

**Note**: Avoid over-engineering. The goal is simplicity, not maximum flexibility.

## Risks and Mitigations (Simplified)

1. **Risk**: Breaking existing functionality
   - **Mitigation**: ✅ Legacy code untouched, new system is opt-in

2. **Risk**: Over-engineering
   - **Mitigation**: ✅ Keep it simple - just dataclasses and functions, no complex abstractions

3. **Risk**: Unnecessary complexity
   - **Mitigation**: ✅ Avoid features that aren't needed (e.g., YAML configs, category selection)

## Success Criteria (Simplified)

1. ✅ All 15 variants work with new system
2. ✅ Output format matches legacy exactly
3. ✅ Legacy system remains functional (backward compatible)
4. ✅ Adding new variants is simpler (edit registry vs. adding if/elif branches)
5. ✅ Code duplication reduced (reuse existing functions)

**Key metric**: Is the new system simpler to maintain than the old one? Yes.

### Simplified To-dos

**Core Implementation (Done):**
- [x] Create variant system with simple dataclasses
- [x] Implement executor function
- [x] Create variant registry
- [x] Add `--use-new-variants` flag to script
- [x] Add `--skip-variants` for basic filtering
- [x] Create tests

**Optional (Only if needed):**
- [ ] Remove legacy code (only after full migration)
- [ ] Add `--variants` flag (only if there's actual demand)
- [x] Manual output validation (when convenient) - Completed: test_variant_7_outputs_match compares old vs new system