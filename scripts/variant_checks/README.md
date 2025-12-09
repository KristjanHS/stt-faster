# Variant Parameter Verification Scripts

This directory contains scripts to verify what transcription parameters each variant actually uses when calling `model.transcribe()`.

## Scripts

### `verify_variant_7.py`

Verifies that variant 7 uses the expected minimal parameters:
- `beam_size: 5` (hardcoded)
- `word_timestamps: False` (hardcoded)
- `task: "transcribe"` (hardcoded)
- All other parameters omitted (use faster-whisper defaults)

**Usage:**
```bash
.venv/bin/python scripts/variant_checks/verify_variant_7.py
```

### `verify_variant_17.py`

Verifies that variant 17 uses the base parameters plus the `no_speech_threshold` override:
- `beam_size: 5` (hardcoded)
- `word_timestamps: False` (hardcoded)
- `task: "transcribe"` (hardcoded)
- `no_speech_threshold: 0.5` (from override)
- All other parameters omitted (use faster-whisper defaults)

**Usage:**
```bash
.venv/bin/python scripts/variant_checks/verify_variant_17.py
```

### `verify_all_variants.py`

Verifies transcription parameters for all currently enabled variants (7, 17, 18, 19).

**Usage:**
```bash
.venv/bin/python scripts/variant_checks/verify_all_variants.py
```

## Purpose

These scripts help ensure that:
1. Variants are configured correctly
2. Overrides are properly applied
3. Only allowed override parameters are included
4. The actual parameters passed to `model.transcribe()` match expectations

## How It Works

The scripts:
1. Load variant definitions from the registry
2. Create transcription configs using the variant's preset and overrides
3. Show what parameters would be passed to `model.transcribe()` based on the code in `transcribe_with_minimal_params()`
4. Verify that overrides are in the `allowed_override_params` set

## Notes

- Variants using the "minimal" preset only pass a few hardcoded parameters
- Additional parameters from `transcription_overrides` are only included if they're in the `allowed_override_params` set
- All other parameters are omitted, allowing faster-whisper to use its internal defaults

