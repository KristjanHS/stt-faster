# Estonian Whisper Model Implementation Plan

**Date:** December 5, 2025  
**Status:** Ready to implement  
**Model:** TalTechNLP/whisper-large-v3-turbo-et-verbatim (September 2025)

---

## Executive Summary

Add support for the newest Estonian Whisper model and make Estonian the default language throughout the project. The model already includes CTranslate2 format in a `ct2/` subfolder - no conversion needed.

---

## Research Summary

### Best Estonian Model Found

**`TalTechNLP/whisper-large-v3-turbo-et-verbatim`** (September 2025)
- ✅ Latest and best Estonian Whisper model available
- ✅ Based on Whisper large-v3-turbo (faster and more accurate than v2)
- ✅ Fine-tuned on 1400 hours of Estonian verbatim transcriptions
- ✅ **Already includes CT2 format in `ct2/` subfolder** - no conversion needed!
- ✅ Tested and confirmed working with faster-whisper
- ✅ WER (Word Error Rate): Best performance on Estonian Common Voice dataset

### Why Not the Original Models

- `TalTechNLP/whisper-medium-et` (March 2023) - Too old, based on older architecture
- `TalTechNLP/whisper-large-et` (January 2024) - Based on v2, no CT2 format included
- Both require manual CT2 conversion using `ct2-transformers-converter`

### CT2 Subfolder Structure

The model stores CT2 files in a subfolder:

```
TalTechNLP/whisper-large-v3-turbo-et-verbatim/
├── ct2/                    ← CT2 format (we need this)
│   ├── model.bin          ← CT2 model file
│   ├── config.json
│   ├── tokenizer.json
│   ├── vocabulary.json
│   └── preprocessor_config.json
├── model.safetensors       ← PyTorch format
└── ... (other files)
```

**Solution:** Use `huggingface_hub.snapshot_download()` to download the model, then reference the `ct2/` subfolder path when loading with `WhisperModel()`.

**Validation Completed:** Model successfully loaded and tested with faster-whisper on CPU and CUDA.

---

## Implementation Details

### File 1: `backend/transcribe.py`

#### Change 1: Add Import
**Location:** After existing imports (line ~30)

```python
from huggingface_hub import snapshot_download
```

#### Change 2: Add Helper Function
**Location:** Before `pick_model()` function (around line ~40)

```python
def _get_estonian_model_path(model_id: str) -> str:
    """Download Estonian model and return path to CT2 subfolder.
    
    Estonian models from TalTechNLP store CT2 files in a 'ct2' subdirectory.
    This helper downloads the model and returns the correct path.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'TalTechNLP/whisper-large-v3-turbo-et-verbatim')
        
    Returns:
        Path to the CT2 model directory
        
    Raises:
        FileNotFoundError: If CT2 folder doesn't exist in model
    """
    model_path = snapshot_download(model_id)
    ct2_path = Path(model_path) / "ct2"
    if not ct2_path.exists():
        raise FileNotFoundError(f"CT2 folder not found in {model_id}. Expected at: {ct2_path}")
    return str(ct2_path)
```

#### Change 3: Update `pick_model()` Default
**Location:** Line ~44

```python
# BEFORE:
def pick_model(preset: str = "turbo") -> WhisperModel:

# AFTER:
def pick_model(preset: str = "et-large") -> WhisperModel:
```

#### Change 4: Add Estonian Preset
**Location:** Inside `pick_model()`, before other presets (line ~58)

```python
    # Estonian models (default)
    if preset == "et-large":
        # TalTech Estonian Whisper large-v3-turbo (newest, Sep 2025)
        # Fine-tuned on 1400 hours of Estonian verbatim transcriptions
        model_path = _get_estonian_model_path("TalTechNLP/whisper-large-v3-turbo-et-verbatim")
        return _load(model_path, default_device="cuda", default_type="int8_float16")

    # Original English/multilingual models (keep existing ones below)
```

#### Change 5: Update `transcribe()` Default
**Location:** Line ~96

```python
# BEFORE:
def transcribe(path: str, preset: str = "distil") -> Dict[str, Any]:

# AFTER:
def transcribe(path: str, preset: str = "et-large") -> Dict[str, Any]:
```

#### Change 6: Force Estonian Language
**Location:** Inside `transcribe()`, before `model.transcribe()` call (line ~101)

```python
    # BEFORE:
    segments, info = model.transcribe(path, beam_size=5, word_timestamps=False)

    # AFTER:
    # Force Estonian language for Estonian models
    language = "et" if preset.startswith("et-") else None
    
    segments, info = model.transcribe(
        path,
        beam_size=5,
        word_timestamps=False,
        language=language,
        task="transcribe",
    )
```

#### Change 7: Update `transcribe_to_json()` Default
**Location:** Line ~122

```python
# BEFORE:
def transcribe_to_json(audio_path: str, json_path: str, preset: str = "distil") -> None:

# AFTER:
def transcribe_to_json(audio_path: str, json_path: str, preset: str = "et-large") -> None:
```

#### Change 8: Update `__main__` Block
**Location:** Line ~129

```python
# BEFORE:
transcribe_to_json(str(input_audio_path), str(output_json_path), preset="distil")

# AFTER:
transcribe_to_json(str(input_audio_path), str(output_json_path), preset="et-large")
```

---

### File 2: `scripts/transcribe_manager.py`

#### Change 1: Update Default Preset
**Location:** Line 152

```python
# BEFORE:
"--preset",
default="distil",

# AFTER:
"--preset",
default="et-large",
```

#### Change 2: Update Choices List
**Location:** Line 153

```python
# BEFORE:
choices=["turbo", "distil", "large8gb"],
help="Model preset to use (default: distil)",

# AFTER:
choices=["turbo", "distil", "large8gb", "et-large"],
help="Model preset to use (default: et-large)",
```

---

### File 3: `scripts/transcription/processor.py`

#### Change 1: Update Default Preset
**Location:** Line 24 in `__init__` method

```python
# BEFORE:
def __init__(
    self,
    db: TranscriptionDatabase,
    input_folder: str | Path,
    preset: str = "distil",
) -> None:

# AFTER:
def __init__(
    self,
    db: TranscriptionDatabase,
    input_folder: str | Path,
    preset: str = "et-large",
) -> None:
```

---

## Files to Modify

1. ✅ `backend/transcribe.py` - 8 changes (imports, helper function, presets, defaults)
2. ✅ `scripts/transcribe_manager.py` - 2 changes (default, choices)
3. ✅ `scripts/transcription/processor.py` - 1 change (default)

---

## Testing & Verification

### Step 1: Test Model Loading

```bash
.venv/bin/python -c "
from backend.transcribe import pick_model
model = pick_model('et-large')
print('✓ Estonian model loaded successfully')
"
```

### Step 2: Test Transcription

```bash
# Test with your Estonian audio file
.venv/bin/python backend/transcribe.py
```

### Step 3: Test CLI

```bash
# Check that et-large is available and default
.venv/bin/python scripts/transcribe_manager.py process --help
```

### Step 4: Run Unit Tests

```bash
.venv/bin/python -m pytest tests/unit -q
```

### Step 5: Run Pre-commit Checks

```bash
uv run pre-commit run --all-files
```

### Step 6: Test Integration

```bash
# Process a folder with Estonian audio
.venv/bin/python scripts/transcribe_manager.py process /path/to/estonian/audio
```

---

## Model Performance Notes

- **Architecture:** Whisper large-v3-turbo
- **Training Data:** 1400 hours Estonian verbatim transcriptions
- **Compute Type:** int8_float16 (optimized for 8GB GPU)
- **Fallback:** Automatic CPU fallback if CUDA unavailable
- **Speed:** ~4x faster than original Whisper
- **Accuracy:** Optimized for Estonian language

---

## Dependencies

All required packages are already in the project:

- ✅ `faster-whisper` - For inference
- ✅ `huggingface_hub` - For model downloading
- ✅ `ctranslate2` - CT2 backend (v4.4.0)

No new dependencies need to be installed.

---

## Backup & Safety

- ✅ Backup created: `backend/transcribe.py.backup`
- All changes are localized to 3 files
- No changes to external API or JSON response structure
- Backward compatible - old presets still work

---

## Troubleshooting

### Issue: Model Download Timeout

**Solution:** The first run will download ~1.5GB. Use faster HF transfer:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

### Issue: CUDA Out of Memory

**Solution:** The model uses int8_float16 for 8GB GPUs. If still OOM:

```python
# In _load() helper, change compute_type:
default_type="int8"  # instead of "int8_float16"
```

### Issue: CT2 Folder Not Found

**Solution:** Clear HF cache and re-download:

```bash
rm -rf ~/.cache/hf/hub/models--TalTechNLP--whisper-large-v3-turbo-et-verbatim
```

---

## Timeline

- **Research & Validation:** ✅ Completed
- **Plan Creation:** ✅ Completed
- **Implementation:** ⏳ Ready to start
- **Testing:** ⏳ After implementation
- **Deployment:** ⏳ After testing

---

## Success Criteria

- ✅ Estonian model loads successfully
- ✅ Transcription produces Estonian text
- ✅ All tests pass
- ✅ Pre-commit checks pass
- ✅ Default preset is `et-large`
- ✅ Language is forced to `et` for Estonian presets

---

## References

- Model: https://huggingface.co/TalTechNLP/whisper-large-v3-turbo-et-verbatim
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- CTranslate2: https://github.com/OpenNMT/CTranslate2

---

**End of Plan**
