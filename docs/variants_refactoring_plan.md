<!-- 827c1271-850d-46ed-b1e6-b579f8507bdf f6ed0af2-584a-4705-9b32-b878c2b035e2 -->
# Variants Architecture Refactoring Plan

## Urgent Technical Debt Checklist

**Priority: High - These issues are blocking maintainability and causing rapid code growth**

### Critical Issues (Fix First)

- [ ] **Massive code duplication in preprocessing functions** (11 functions, ~700+ lines)
  - `preprocess_only_ffmpeg()` - duplicates orchestrator logic
  - `preprocess_loudnorm_only()` - new lightweight variant
  - `preprocess_loudnorm_with_highpass()` - new lightweight variant with filter
  - `preprocess_dynaudnorm_only()` - new dynamic normalization variant
  - `preprocess_only_denoise()` - duplicates orchestrator logic
  - `preprocess_only_denoise_custom()` - custom denoise parameters
  - Plus 4 helper functions: `_loudnorm_only()`, `_loudnorm_with_highpass()`, `_dynaudnorm_only()`, `_simple_resample()`
  - **Impact**: Each new variant requires ~100-150 lines of duplicated code
  - **Solution**: Extend `backend/variants/preprocess_steps.py` to support all step types

- [ ] **Giant if/elif chain in `run_variant()`** (11 branches, ~240 lines, lines 968-1188)
  - Each variant requires a new elif branch with wrapper function
  - Pattern repeats: create wrapper, set config, set runner, set transcription provider
  - **Impact**: Adding one variant = ~20-30 lines of boilerplate
  - **Solution**: Complete migration to declarative variant system

- [ ] **Variant count mismatch** (New system: 9 variants, Legacy: 12 variants)
  - New variants missing from registry: `norm_highp_noparamtrans` (8a), `norm_dynaud_noparamtrans` (8b), `onlyden_noparamtrans_custom` (9a)
  - **Impact**: New system incomplete, cannot replace legacy system
  - **Solution**: Add missing variants to `backend/variants/registry.py` and extend step types

- [ ] **Missing preprocessing step types in new system**
  - New system only supports: `ffmpeg`, `denoise`, `resample`
  - Legacy system has: `loudnorm_only`, `loudnorm_with_highpass`, `dynaudnorm_only`, `denoise_custom`
  - **Impact**: Cannot express new variants declaratively
  - **Solution**: Extend `PreprocessStep.step_type` to include all step types

### High Priority Issues (Fix Next)

- [ ] **Custom denoise parameters not supported in new system**
  - `preprocess_only_denoise_custom()` uses hardcoded params: `noise_clip_duration_s=5.0`, `n_std_thresh_stationary=0.75`, `prop_decrease=0.25`
  - **Impact**: Cannot express variant 9a in declarative system
  - **Solution**: Add `config` dict to `PreprocessStep` for step-specific parameters

- [ ] **Lightweight ffmpeg steps not composable**
  - Three separate functions for similar operations (loudnorm, loudnorm+highpass, dynaudnorm)
  - **Impact**: Code duplication, hard to maintain filter graph variations
  - **Solution**: Create composable ffmpeg filter builder in `preprocess_steps.py`

- [ ] **No way to select subset of variants**
  - Script always runs all variants or uses hardcoded list
  - **Impact**: Cannot test individual variants, wastes time/resources
  - **Solution**: Add `--variants` CLI argument (Phase 4.2)

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

### Current Variants (12 total in legacy system, 9 in new system)

**Legacy System Variants:**
1. `no_preprocessing`: No preprocessing + project defaults (currently disabled)
2. `industry_defaults`: No preprocessing + industry defaults
3. `ffmpeg_only`: Only ffmpeg pipeline + project defaults
4. `denoise_only`: Only denoise_light + project defaults
5. `ffmpeg_industry_defaults`: Only ffmpeg + industry defaults
6. `full_industry_defaults`: Full preprocessing + industry defaults
7. `noprep_noparamtrans`: No preprocessing + minimal params
8. `normonly_noparamtrans`: Only loudness normalization (lightweight) + minimal params
8a. `norm_highp_noparamtrans`: Loudness normalization + highpass filter + minimal params
8b. `norm_dynaud_noparamtrans`: Dynamic audio normalization + minimal params
9. `onlyden_noparamtrans`: Only denoise + minimal params
9a. `onlyden_noparamtrans_custom`: Only denoise with custom params + minimal params

**New System Variants (incomplete):**
- Only supports variants 1-9 (missing 8a, 8b, 9a)
- Missing step types: `loudnorm_only`, `loudnorm_with_highpass`, `dynaudnorm_only`, `denoise_custom`

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

#### Step 1.1: Create Variant Definition Infrastructure

**Files to create:**

- `backend/variants/__init__.py`
- `backend/variants/variant.py`: Core `Variant` dataclass
- `backend/variants/preprocess_steps.py`: Preprocessing step definitions
- `backend/variants/transcription_presets.py`: Transcription config presets

**Key structures:**

```python
@dataclass
class PreprocessStep:
    name: str
    enabled: bool
    step_type: Literal[
        "ffmpeg",           # Full ffmpeg pipeline (highpass + resample + rnnoise + loudnorm)
        "loudnorm_only",    # Lightweight: resample + loudnorm only
        "loudnorm_highpass", # Lightweight: highpass + resample + loudnorm
        "dynaudnorm",       # Lightweight: resample + dynamic audio normalization
        "denoise",          # Light denoising (noisereduce)
        "denoise_custom",   # Light denoising with custom parameters
        "resample"          # Simple resampling only
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

#### Step 1.2: Create Transcription Presets

**File:** `backend/variants/transcription_presets.py`

**Presets to implement:**

- `get_project_defaults()`: Returns `TranscriptionConfig.from_env()` (current project defaults)
- `get_industry_defaults()`: Returns industry-standard faster-whisper defaults (move from `compare_transcription_variants.py:328-368`)
- `get_minimal_config()`: Returns config that omits parameters to use faster-whisper internal defaults

#### Step 1.3: Create Composable Preprocessing System

**File:** `backend/variants/preprocess_steps.py`

**Functions:**

- `create_preprocess_runner(steps: list[PreprocessStep], config: PreprocessConfig) -> Callable`: Builds a preprocessing function from step definitions
- Reuse existing `run_ffmpeg_pipeline` and `apply_light_denoise` from backend
- Add support for lightweight ffmpeg steps: `_loudnorm_only()`, `_loudnorm_with_highpass()`, `_dynaudnorm_only()`
- Support custom denoise parameters via step config
- Support step combinations: none, ffmpeg-only, loudnorm variants, denoise-only, denoise-custom, full (ffmpeg + denoise)

### Phase 2: Variant Executor

#### Step 2.1: Create Variant Executor

**File:** `backend/variants/executor.py`

**Key function:**

```python
def execute_variant(
    variant: Variant,
    audio_path: str,
    preset: str,
    language: str | None,
    *,
    output_dir: Path | None = None,
    output_base_path: Path | None = None,
    datetime_suffix: str | None = None,
    copy_intermediate: bool = False,
) -> dict[str, Any]:
    """Execute a single variant and return results."""
```

**Responsibilities:**

- Build preprocessing pipeline from variant steps
- Apply transcription preset
- Handle intermediate file output
- Return results in same format as current `run_variant()`

#### Step 2.2: Create Variant Registry

**File:** `backend/variants/registry.py`

**Function:**

```python
def get_builtin_variants() -> list[Variant]:
    """Return list of all 9 built-in variants."""
```

**Implementation:**

- Define all 12 current variants as `Variant` dataclass instances
- Include new variants: `norm_highp_noparamtrans`, `norm_dynaud_noparamtrans`, `onlyden_noparamtrans_custom`
- Map to existing variant names for backward compatibility
- Support custom denoise parameters via `PreprocessStep.config`

### Phase 3: Script Migration (Backward Compatible)

#### Step 3.1: Update Script to Use New System (Parallel Implementation)

**File:** `scripts/compare_transcription_variants.py`

**Strategy:**

- Add new functions that use variant system
- Keep old `run_variant()` function intact
- Add feature flag or environment variable to switch between old/new
- Default to old system initially

**Changes:**

- Import new variant system
- Add `run_variant_v2()` that uses `execute_variant()`
- Update `main()` to support both old and new paths
- Add `--use-new-variants` flag (default: False for compatibility)

#### Step 3.2: Test Both Systems

- Run existing tests
- Manually verify both old and new paths produce identical results
- Compare outputs from both systems

### Phase 4: Cleanup and Enhancement

#### Step 4.1: Remove Old Implementation

**After verification:**

- Remove old `run_variant()` if/elif chain
- Remove `preprocess_only_ffmpeg()`, `preprocess_only_denoise()`
- Remove `transcribe_with_minimal_params()`
- Remove `get_industry_default_transcription_config()`, `get_minimal_transcription_config()`
- Update script to use new system by default

#### Step 4.2: Add Flexible Variant Selection

**Enhancement:**

- Add `--variants` CLI argument to select specific variants by name or number
- Add `--variant-category` to select by category (e.g., "no_preprocessing", "industry_defaults")
- Support comma-separated lists: `--variants 1,3,5` or `--variants no_preprocessing,ffmpeg_only`

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

1. All 12 existing variants work with new system (including 8a, 8b, 9a)
2. Output format matches current implementation exactly
3. Script remains backward compatible during transition
4. New system is easier to extend with new variants
5. Code duplication is eliminated (remove 11 preprocessing functions)
6. `run_variant()` if/elif chain replaced with declarative system
7. All preprocessing step types supported in composable system

### To-dos

- [ ] Create backend/variants/ package with variant.py (core dataclass), preprocess_steps.py (step definitions), and transcription_presets.py (config presets)
- [ ] Implement transcription presets: project_defaults, industry_defaults, and minimal_config in transcription_presets.py
- [ ] Create composable preprocessing system in preprocess_steps.py that can build preprocessing pipelines from step definitions
- [ ] Implement execute_variant() function in executor.py that runs variants using the new system
- [ ] Create registry.py with get_builtin_variants() that defines all 9 current variants as Variant dataclass instances
- [ ] Update compare_transcription_variants.py to support both old and new systems via feature flag, keeping old code intact
- [ ] Test both old and new systems produce identical outputs, verify all 9 variants work correctly
- [ ] Remove old run_variant() if/elif chain, custom preprocessing functions, and transcribe_with_minimal_params() after verification
- [ ] Add --variants and --variant-category CLI arguments for flexible variant selection

