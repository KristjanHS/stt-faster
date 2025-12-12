# Variant Parameter Verification Scripts

This directory contains scripts to verify what transcription parameters each variant actually uses when calling `model.transcribe()`.

## Scripts

### `verify_all_variants.py`

Verifies transcription parameters for all currently enabled variants by actually executing each variant (with mocked model) to verify what parameters are passed to `model.transcribe()`.

The script verifies:
- Base minimal parameters: `beam_size: 5`, `word_timestamps: False`, `task: "transcribe"`
- Override parameters from variant configs (e.g., `no_speech_threshold`, `chunk_length`, etc.)
- That only allowed override parameters are included
- That the actual parameters passed to `model.transcribe()` match expectations

**Usage:**
```bash
.venv/bin/python scripts/variant_checks/verify_all_variants.py
```

**Or via Make:**
```bash
make verify-variants
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

