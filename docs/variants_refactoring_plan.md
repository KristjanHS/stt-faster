<!-- 827c1271-850d-46ed-b1e6-b579f8507bdf f6ed0af2-584a-4705-9b32-b878c2b035e2 -->
# Variants Architecture Refactoring Plan

## Executive Summary

**Status**: Phase 1-3 Complete, Phase 4.0 Complete, Phase 4.1-4.2 Pending

**Progress**: ~95% Complete

- ✅ **Foundation Complete**: Variant system infrastructure, executor, and registry implemented
- ✅ **Migration Complete**: Script supports both old and new systems via `--use-new-variants` flag
- ✅ **Step Types Complete**: All 12 step types supported (`ffmpeg`, `denoise`, `resample`, `loudnorm_only`, `loudnorm_highpass`, `dynaudnorm`, `denoise_custom`, `highlow_aform_loudnorm`, `highlow_nosampl_loudnorm`, `aresampl_loudnorm_fixed`, `aresampl_loudnorm_fixed2`, `loudnorm_2pass_linear`)
- ✅ **Variants 10-11 Migrated**: Now use declarative steps instead of custom functions
- ✅ **Variants 12-16 Migrated**: Now use declarative steps instead of custom functions
- ❌ **Cleanup Pending**: Old code still exists, waiting for verification before removal

**Next Steps**:
1. ✅ Add missing step types to `PreprocessStep.step_type` - **DONE**
2. ✅ Migrate variants 10-11 from `custom_preprocess_runner` to declarative steps - **DONE**
3. ✅ Migrate variants 12-16 to declarative steps - **DONE**
4. Verify output compatibility between old and new systems
5. Remove legacy code after verification

## Implementation Status

**Last Updated**: Dec 9, 2025 (Phase 4.0 completed: all variants 1-16 now use declarative steps)

### ✅ Completed (Phase 1-3 Foundation)

- [x] **Phase 1.1: Variant Definition Infrastructure** ✅
  - Created `backend/variants/__init__.py`
  - Created `backend/variants/variant.py` with `Variant` and `PreprocessStep` dataclasses
  - Created `backend/variants/preprocess_steps.py` with `create_preprocess_runner()`
  - Created `backend/variants/transcription_presets.py` with all three presets

- [x] **Phase 1.2: Transcription Presets** ✅
  - Implemented `get_project_defaults()` - returns `TranscriptionConfig.from_env()`
  - Implemented `get_industry_defaults()` - industry-standard faster-whisper defaults
  - Implemented `get_minimal_config()` - minimal params for faster-whisper internal defaults
  - Implemented `get_transcription_config(preset)` helper

- [x] **Phase 1.3: Composable Preprocessing System** ✅
  - Implemented `create_preprocess_runner()` in `preprocess_steps.py`
  - Supports `ffmpeg`, `denoise`, `resample` step types
  - Reuses existing `run_ffmpeg_pipeline` and `apply_light_denoise`
  - Supports step combinations and intermediate file copying

- [x] **Phase 2.1: Variant Executor** ✅
  - Created `backend/variants/executor.py` with `execute_variant()`
  - Handles preprocessing pipeline composition
  - Applies transcription presets
  - Manages intermediate file output
  - Returns results in same format as legacy `run_variant()`
  - Includes `_transcribe_with_minimal_params()` for minimal preset

- [x] **Phase 2.2: Variant Registry** ✅
  - Created `backend/variants/registry.py` with `get_builtin_variants()`
  - Defines **16 variants** (1-16) as `Variant` dataclass instances
  - All variants 1-16 use declarative `preprocess_steps`
  - Includes helper functions: `get_variant_by_name()`, `get_variant_by_number()`

- [x] **Phase 3.1: Script Migration (Parallel Implementation)** ✅
  - Added `--use-new-variants` CLI flag to `compare_transcription_variants.py`
  - New system can be enabled via flag (default: False for backward compatibility)
  - Legacy `run_variant()` function still intact
  - Both systems produce compatible output format
  - Supports `--skip-variants` for filtering

## Urgent Technical Debt Checklist

**Priority: High - These issues are blocking maintainability and causing rapid code growth**

### Critical Issues (Fix First)

- [x] **Missing preprocessing step types in new system** ✅ **COMPLETED**
  - ✅ New system now supports: `ffmpeg`, `denoise`, `resample`, `loudnorm_only`, `loudnorm_highpass`, `dynaudnorm`, `denoise_custom`, `highlow_aform_loudnorm`, `highlow_nosampl_loudnorm`, `aresampl_loudnorm_fixed`, `aresampl_loudnorm_fixed2`, `loudnorm_2pass_linear` (12 step types total)
  - ✅ Extended `PreprocessStep.step_type` Literal to include all step types
  - ✅ Implemented handlers in `create_preprocess_runner()` for all new step types
  - ✅ **Current status**: All variants 1-16 now use declarative steps (no `custom_preprocess_runner` needed)
  - **Files updated**: `backend/variants/variant.py`, `backend/variants/preprocess_steps.py`

- [x] **Massive code duplication in preprocessing functions** ✅ **RESOLVED FOR NEW SYSTEM**
  - ✅ Helper functions moved to `preprocess_steps.py`: `_loudnorm_only()`, `_loudnorm_with_highpass()`, `_dynaudnorm_only()`, `_highlow_aform_loudnorm()`, `_highlow_nosampl_loudnorm()`, `_aresampl_loudnorm_fixed()`, `_aresampl_loudnorm_fixed2()`, `_loudnorm_2pass_linear()`
  - ✅ All variants 1-16 now use declarative steps (no custom functions needed in new system)
  - ⚠️ Legacy functions still exist in script: `preprocess_only_ffmpeg()`, `preprocess_only_denoise()`, `preprocess_loudnorm_only()`, etc. (will be removed after verification)
  - **Impact**: Code duplication eliminated in new system; legacy code can be removed after verification
  - **Solution**: Remove all legacy functions after verifying new system produces identical outputs

- [ ] **Giant if/elif chain in `run_variant()`** (Still exists, ~240 lines)
  - Each variant requires a new elif branch with wrapper function
  - Pattern repeats: create wrapper, set config, set runner, set transcription provider
  - **Impact**: Adding one variant = ~20-30 lines of boilerplate
  - **Solution**: Complete migration to new system, then remove old `run_variant()` function

### High Priority Issues (Fix Next)

- [x] **Custom denoise parameters not supported in new system** ✅ **COMPLETED**
  - ✅ `PreprocessStep.config` dict now used for `denoise_custom` step type
  - ✅ `create_preprocess_runner()` extracts custom params from `step.config` when `step_type == "denoise_custom"`
  - ✅ Supports: `noise_clip_duration_s`, `n_std_thresh_stationary`, `prop_decrease`
  - **Impact**: Can now express variants with custom denoise params declaratively

- [x] **Lightweight ffmpeg steps not composable** ✅ **COMPLETED**
  - ✅ Implemented handlers in `create_preprocess_runner()` for `loudnorm_only`, `loudnorm_highpass`, `dynaudnorm`
  - ✅ Helper functions moved to `preprocess_steps.py`: `_loudnorm_only()`, `_loudnorm_with_highpass()`, `_dynaudnorm_only()`
  - ✅ Variants 10-11 now use declarative steps instead of custom functions
  - **Impact**: Reduced code duplication, easier to maintain filter graph variations

- [ ] **No way to select subset of variants**
  - Script supports `--skip-variants` but no way to select specific variants
  - **Impact**: Cannot test individual variants efficiently, wastes time/resources
  - **Solution**: Add `--variants` CLI argument to select by name/number (Phase 4.2)

### Medium Priority Issues (Nice to Have)

- [ ] **Statistics logging duplicated** (appears twice in main(), lines 1436-1470 and 1504-1538)
  - **Impact**: Code duplication, harder to maintain
  - **Solution**: Extract to shared function

- [ ] **Variant description mapping hardcoded** (`_get_variant_description()` dict)
  - **Impact**: Must update in two places when adding variants
  - **Solution**: Move to Variant dataclass or registry

- [ ] **No validation of variant definitions**
  - **Impact**: Runtime errors if variant misconfigured
  - **Solution**: Add validation in `execute_variant()`

## Current State Analysis

### Problems

1. **Hardcoded variant definitions**: The `run_variant()` function in `scripts/compare_transcription_variants.py` uses a large if/elif chain (lines 480-631) to define variants
2. **Code duplication**: Custom preprocessing functions (`preprocess_only_ffmpeg`, `preprocess_only_denoise`) duplicate logic from `backend/preprocess/orchestrator.py`
3. **Inflexible architecture**: Adding new variants requires code changes; no way to select subsets of variants
4. **Mixed concerns**: Variant definition, execution, and output handling are intertwined
5. **Special-case handling**: `transcribe_with_minimal_params()` duplicates transcription logic

### Current Variants (16 total in new system, legacy system still exists)

**New System Variants (16 total):**
1. `no_preprocessing`: No preprocessing + project defaults ✅ (declarative)
2. `industry_defaults`: No preprocessing + industry defaults ✅ (declarative)
3. `ffmpeg_only`: Only ffmpeg pipeline + project defaults ✅ (declarative)
4. `denoise_only`: Only denoise_light + project defaults ✅ (declarative)
5. `ffmpeg_industry_defaults`: Only ffmpeg + industry defaults ✅ (declarative)
6. `full_industry_defaults`: Full preprocessing + industry defaults ✅ (declarative)
7. `noprep_noparamtrans`: No preprocessing + minimal params ✅ (declarative)
8. `normonly_noparamtrans`: Only loudness normalization + minimal params ✅ (declarative, uses ffmpeg step)
9. `onlyden_noparamtrans`: Only denoise + minimal params ✅ (declarative)
10. `norm_highp_noparamtrans`: Loudness normalization + highpass + minimal params ✅ (declarative, uses loudnorm_highpass step)
11. `norm_dynaud_noparamtrans`: Dynamic audio normalization + minimal params ✅ (declarative, uses dynaudnorm step)
12. `lnorm_highlow_aform_noparamtrans`: Highpass + lowpass + aformat + loudnorm + minimal params ✅ (declarative, uses highlow_aform_loudnorm step)
13. `lnorm_highlow_nosampl_noparamtrans`: Highpass + lowpass + loudnorm (no aformat) + minimal params ✅ (declarative, uses highlow_nosampl_loudnorm step)
14. `lnorm2_aresampl_noparamtrans`: Aresample to 16kHz + loudnorm (fixed params) + minimal params ✅ (declarative, uses aresampl_loudnorm_fixed step)
15. `lnorm3_aresampl_noparamtrans`: Aresample to 16kHz + loudnorm (I=-24, LRA=15) + minimal params ✅ (declarative, uses aresampl_loudnorm_fixed2 step)
16. `loudnorm_2pass_linear_noparamtrans`: 2-pass loudnorm in linear mode + minimal params ✅ (declarative, uses loudnorm_2pass_linear step)

**Legacy System Status:**
- Old `run_variant()` function still exists (lines 1619+)
- Old preprocessing functions still exist: `preprocess_only_ffmpeg()`, `preprocess_loudnorm_only()`, `preprocess_loudnorm_with_highpass()`, `preprocess_dynaudnorm_only()`, `preprocess_only_denoise()`, `preprocess_only_denoise_custom()`, etc.
- Legacy system can be used when `--use-new-variants` flag is not set (default: False)

## Target Architecture

### Core Principles

1. **Declarative variant definitions**: Variants defined as data structures (dataclasses)
2. **Composable preprocessing**: Steps can be enabled/disabled independently
3. **Composable transcription configs**: Presets for different parameter sets
4. **Separation of concerns**: Definition, execution, and output handling are separate
5. **Backward compatibility**: Existing script remains functional during transition

### New Components

#### 1. Variant Definition System (`backend/variants/`)

- `variant.py`: Core variant dataclass and registry
- `preprocess_steps.py`: Composable preprocessing step definitions
- `transcription_presets.py`: Transcription config presets (project, industry, minimal)
- `registry.py`: Built-in variant registry

#### 2. Variant Executor (`backend/variants/executor.py`)

- Executes variant definitions
- Handles preprocessing pipeline composition
- Applies transcription configs
- Manages intermediate file output

#### 3. Updated Script (`scripts/compare_transcription_variants.py`)

- Uses new variant system
- Maintains same CLI interface
- Supports flexible variant selection (all, by name, by category)

## Implementation Plan

### Phase 1: Foundation (Backward Compatible)

#### Step 1.1: Create Variant Definition Infrastructure ✅ COMPLETED

**Files created:**

- ✅ `backend/variants/__init__.py`
- ✅ `backend/variants/variant.py`: Core `Variant` dataclass
- ✅ `backend/variants/preprocess_steps.py`: Preprocessing step definitions
- ✅ `backend/variants/transcription_presets.py`: Transcription config presets

**Current implementation:**
- ✅ `PreprocessStep` supports: `"ffmpeg"`, `"denoise"`, `"resample"`, `"loudnorm_only"`, `"loudnorm_highpass"`, `"dynaudnorm"`, `"denoise_custom"`, `"highlow_aform_loudnorm"`, `"highlow_nosampl_loudnorm"`, `"aresampl_loudnorm_fixed"`, `"aresampl_loudnorm_fixed2"`, `"loudnorm_2pass_linear"` (all 12 step types)
- `Variant` includes `custom_preprocess_runner` field (no longer used; all variants use declarative steps)

**Key structures (current vs planned):**

```python
@dataclass
class PreprocessStep:
    name: str
    enabled: bool
    step_type: Literal[
        "ffmpeg",                    # Full ffmpeg pipeline (highpass + resample + rnnoise + loudnorm)
        "loudnorm_only",             # Lightweight: resample + loudnorm only
        "loudnorm_highpass",         # Lightweight: highpass + resample + loudnorm
        "dynaudnorm",                # Lightweight: resample + dynamic audio normalization
        "denoise",                   # Light denoising (noisereduce)
        "denoise_custom",            # Light denoising with custom parameters
        "resample",                  # Simple resampling only
        "highlow_aform_loudnorm",    # Highpass + lowpass + aformat + loudnorm
        "highlow_nosampl_loudnorm",  # Highpass + lowpass + loudnorm (no aformat)
        "aresampl_loudnorm_fixed",   # Aresample to 16kHz + loudnorm (I=-23, LRA=11)
        "aresampl_loudnorm_fixed2",  # Aresample to 16kHz + loudnorm (I=-24, LRA=15)
        "loudnorm_2pass_linear"      # 2-pass loudnorm in linear mode
    ]
    config: dict[str, Any] | None = None  # Step-specific parameters (e.g., denoise params)

@dataclass
class Variant:
    name: str
    number: int
    description: str
    preprocess_steps: list[PreprocessStep]
    transcription_preset: str  # "project", "industry", "minimal"
    transcription_overrides: dict[str, Any] | None = None
```

#### Step 1.2: Create Transcription Presets ✅ COMPLETED

**File:** `backend/variants/transcription_presets.py`

**Presets implemented:**

- ✅ `get_project_defaults()`: Returns `TranscriptionConfig.from_env()` (current project defaults)
- ✅ `get_industry_defaults()`: Returns industry-standard faster-whisper defaults
- ✅ `get_minimal_config()`: Returns config that omits parameters to use faster-whisper internal defaults
- ✅ `get_transcription_config(preset)`: Helper function to get config by preset name

#### Step 1.3: Create Composable Preprocessing System ✅ COMPLETED

**File:** `backend/variants/preprocess_steps.py`

**Functions implemented:**

- ✅ `create_preprocess_runner(steps: list[PreprocessStep], config: PreprocessConfig) -> Callable`: Builds a preprocessing function from step definitions
- ✅ Reuses existing `run_ffmpeg_pipeline` and `apply_light_denoise` from backend
- ✅ Supports `_simple_resample()` for resample step type
- ✅ Supports step combinations: none, ffmpeg-only, denoise-only, full (ffmpeg + denoise)
- ✅ Supports intermediate file copying with variant naming
- ✅ **Support for lightweight ffmpeg steps**: `loudnorm_only`, `loudnorm_highpass`, `dynaudnorm` step types
- ✅ **Support for custom denoise parameters**: `denoise_custom` step type with config dict support
- ✅ **Support for complex filter combinations**: `highlow_aform_loudnorm`, `highlow_nosampl_loudnorm`, `aresampl_loudnorm_fixed`, `aresampl_loudnorm_fixed2`, `loudnorm_2pass_linear` step types

### Phase 2: Variant Executor ✅ COMPLETED

#### Step 2.1: Create Variant Executor ✅ COMPLETED

**File:** `backend/variants/executor.py`

**Key function implemented:**

```python
def execute_variant(
    variant: Variant,
    audio_path: str,
    preset: str = "et-large",
    language: str | None = None,
    *,
    output_dir: Path | None = None,
    output_base_path: Path | None = None,
    datetime_suffix: str | None = None,
    copy_intermediate: bool = False,
) -> dict[str, Any]:
    """Execute a single variant and return results."""
```

**Responsibilities implemented:**

- ✅ Build preprocessing pipeline from variant steps (or use custom_preprocess_runner)
- ✅ Apply transcription preset via `create_variant_transcribe_config()`
- ✅ Handle intermediate file output with variant naming
- ✅ Return results in same format as legacy `run_variant()`
- ✅ Includes `_transcribe_with_minimal_params()` for minimal preset
- ✅ Includes `_transcribe_with_config()` for full config presets

#### Step 2.2: Create Variant Registry ✅ COMPLETED

**File:** `backend/variants/registry.py`

**Function implemented:**

```python
def get_builtin_variants() -> list[Variant]:
    """Return list of all built-in variants (1-16)."""
```

**Implementation status:**

- ✅ Defines **16 variants** (1-16) as `Variant` dataclass instances
- ✅ All variants 1-16 use declarative `preprocess_steps`
- ✅ Includes helper functions: `get_variant_by_name()`, `get_variant_by_number()`

### Phase 3: Script Migration (Backward Compatible) ✅ COMPLETED

#### Step 3.1: Update Script to Use New System (Parallel Implementation) ✅ COMPLETED

**File:** `scripts/compare_transcription_variants.py`

**Strategy implemented:**

- ✅ New system uses `execute_variant()` from `backend.variants`
- ✅ Old `run_variant()` function still intact (lines 1619+)
- ✅ Feature flag `--use-new-variants` to switch between old/new
- ✅ Default to old system (False) for backward compatibility

**Changes implemented:**

- ✅ Imports new variant system: `from backend.variants import execute_variant, get_builtin_variants`
- ✅ `main()` supports both old and new paths via `if args.use_new_variants:`
- ✅ New system uses `get_builtin_variants()` and iterates through variants
- ✅ Supports `--skip-variants` for filtering variants
- ✅ Statistics logging after each variant (same format for both systems)

#### Step 3.2: Test Both Systems ⚠️ PARTIALLY COMPLETED

- ✅ Unit tests exist in `tests/unit/test_variants.py`
- ⚠️ Manual verification of identical outputs between systems may need confirmation
- ⚠️ Comparison of outputs from both systems may need verification

### Phase 4: Cleanup and Enhancement ⚠️ IN PROGRESS

#### Step 4.1: Remove Old Implementation ❌ NOT STARTED

**Prerequisites:**
- Complete Phase 4.0: Add missing step types and migrate variants 10-16 to declarative system
- Verify new system produces identical outputs to legacy system

**After verification:**

- ❌ Remove old `run_variant()` if/elif chain (lines 1619+)
- ❌ Remove `preprocess_only_ffmpeg()`, `preprocess_only_denoise()`, `preprocess_loudnorm_only()`, `preprocess_loudnorm_with_highpass()`, `preprocess_dynaudnorm_only()`, `preprocess_only_denoise_custom()`
- ❌ Remove helper functions: `_loudnorm_only()`, `_loudnorm_with_highpass()`, `_dynaudnorm_only()`, `_simple_resample()` (if moved to preprocess_steps.py)
- ❌ Remove `transcribe_with_minimal_params()` from script (already in executor.py)
- ❌ Remove `get_industry_default_transcription_config()`, `get_minimal_transcription_config()` from script (moved to transcription_presets.py)
- ❌ Update script to use new system by default (change `--use-new-variants` default to True or remove flag)

#### Step 4.0: Complete Step Type Support ✅ COMPLETED

**Tasks:**

- [x] Extend `PreprocessStep.step_type` Literal in `variant.py` to include: `"loudnorm_only"`, `"loudnorm_highpass"`, `"dynaudnorm"`, `"denoise_custom"`, `"highlow_aform_loudnorm"`, `"highlow_nosampl_loudnorm"`, `"aresampl_loudnorm_fixed"`, `"aresampl_loudnorm_fixed2"`, `"loudnorm_2pass_linear"` ✅
- [x] Implement handlers in `create_preprocess_runner()` for all new step types ✅
- [x] Update variants 10-11 in `registry.py` to use declarative steps instead of `custom_preprocess_runner` ✅
- [x] Test that variants 10-11 work with declarative steps ✅
- [x] Update variants 12-16 in `registry.py` to use declarative steps ✅
- [x] Move helper functions to `preprocess_steps.py`: `_highlow_aform_loudnorm()`, `_highlow_nosampl_loudnorm()`, `_aresampl_loudnorm_fixed()`, `_aresampl_loudnorm_fixed2()`, `_loudnorm_2pass_linear()` ✅
- [x] Test that variants 12-16 work with declarative steps ✅

#### Step 4.2: Add Flexible Variant Selection ❌ NOT STARTED

**Enhancement:**

- ❌ Add `--variants` CLI argument to select specific variants by name or number
- ❌ Add `--variant-category` to select by category (e.g., "no_preprocessing", "industry_defaults")
- ❌ Support comma-separated lists: `--variants 1,3,5` or `--variants no_preprocessing,ffmpeg_only`
- ⚠️ Note: `--skip-variants` already exists but only supports skipping, not selecting

#### Step 4.3: Support External Variant Definitions

**Future enhancement:**

- Support loading variants from YAML/JSON files
- Allow users to define custom variants without code changes

## File Structure

```
backend/variants/
├── __init__.py
├── variant.py              # Core Variant dataclass
├── preprocess_steps.py     # Preprocessing step definitions
├── transcription_presets.py # Transcription config presets
├── executor.py             # Variant execution engine
└── registry.py             # Built-in variant registry

scripts/
└── compare_transcription_variants.py  # Updated to use new system
```

## Migration Strategy

### Intermediate Stage Requirements

1. **Current script must remain runnable**: Keep all existing functions until new system is verified
2. **Gradual migration**: Use feature flag to switch between old/new
3. **Output compatibility**: New system produces identical output format
4. **Test coverage**: Ensure existing tests pass with both systems

### Testing Approach

1. Run both old and new systems on same audio file
2. Compare JSON outputs (should be identical)
3. Verify intermediate file naming matches
4. Check that all 9 variants work correctly

## Benefits

1. **Flexibility**: Easy to add new variants without code changes
2. **Maintainability**: Single source of truth for preprocessing and transcription configs
3. **Testability**: Variant definitions can be tested independently
4. **Extensibility**: Support for custom variants via config files (future)
5. **Clarity**: Clear separation between variant definition and execution

## Risks and Mitigations

1. **Risk**: Breaking existing functionality during migration

   - **Mitigation**: Keep old code until new system is fully tested

2. **Risk**: Performance regression

   - **Mitigation**: Profile both systems, optimize if needed

3. **Risk**: Output format changes

   - **Mitigation**: Ensure new executor returns identical format

## Success Criteria

1. ✅ All 16 variants work with new system (all variants 1-16 use declarative steps)
2. ✅ Output format matches current implementation exactly
3. ✅ Script remains backward compatible during transition (legacy system still available)
4. ✅ New system is easier to extend with new variants (declarative system in place)
5. ✅ Code duplication eliminated in new system (all variants use declarative steps; legacy functions can be removed after verification)
6. ❌ `run_variant()` if/elif chain not yet replaced (legacy system still active, waiting for verification)
7. ✅ All preprocessing step types fully supported (12/12 step types: ffmpeg, denoise, resample, loudnorm_only, loudnorm_highpass, dynaudnorm, denoise_custom, highlow_aform_loudnorm, highlow_nosampl_loudnorm, aresampl_loudnorm_fixed, aresampl_loudnorm_fixed2, loudnorm_2pass_linear)

### Remaining To-dos

**Critical (Blocking cleanup):**
- [x] Extend `PreprocessStep.step_type` to include: `"loudnorm_only"`, `"loudnorm_highpass"`, `"dynaudnorm"`, `"denoise_custom"`, `"highlow_aform_loudnorm"`, `"highlow_nosampl_loudnorm"`, `"aresampl_loudnorm_fixed"`, `"aresampl_loudnorm_fixed2"`, `"loudnorm_2pass_linear"` ✅
- [x] Implement handlers in `create_preprocess_runner()` for all new step types ✅
- [x] Update variants 10-11 in `registry.py` to use declarative steps instead of `custom_preprocess_runner` ✅
- [x] Test that variants 10-11 work with declarative steps ✅
- [x] Update variants 12-16 in `registry.py` to use declarative steps ✅
- [x] Move helper functions to `preprocess_steps.py` ✅
- [x] Test that variants 12-16 work with declarative steps ✅

**High Priority:**
- [ ] Verify both old and new systems produce identical outputs for all variants
- [ ] Remove old `run_variant()` if/elif chain after verification
- [ ] Remove custom preprocessing functions after verification (all variants now use declarative steps)
- [ ] Add `--variants` CLI argument for flexible variant selection

**Medium Priority:**
- [ ] Extract duplicated statistics logging to shared function
- [ ] Add validation of variant definitions in `execute_variant()`
- [ ] Update script to use new system by default (or remove `--use-new-variants` flag)

