# Audio Transcription - Technical Reference

Batch audio transcription system with Estonian and multilingual Whisper models.

## Architecture

```
backend/transcribe.py          # Model loading & transcription
scripts/transcribe_manager.py  # CLI interface
scripts/transcription/
├── database.py                # SQLite state tracking
└── processor.py               # File processing pipeline
scripts/windows/*.bat          # Windows one-click launchers
```

**Workflow**: Scan → Register (SQLite) → Transcribe → Move to `processed/`|`failed/` → Update status

## Models

### Estonian (Default)
**TalTechNLP/whisper-large-v3-turbo-et-verbatim** - Fine-tuned on 1400h Estonian audio, ~1.5GB, preset: `et-large`

### English/Multilingual
| Preset | Model | Size | Language | Use Case |
|--------|-------|------|----------|----------|
| `large8gb` | Whisper large-v3 | ~3GB | Multi | Highest accuracy |
| `turbo` | Whisper large-v3-turbo | ~1.5GB | Multi | Fast + accurate |
| `distil` | Distilled Whisper | ~756MB | English | Fastest |

**Caching**: All models use `snapshot_download()` for HuggingFace cache (`~/.cache/huggingface/hub/`). Download once, reuse forever.

## Database

```sql
CREATE TABLE transcriptions (
    id INTEGER PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,        -- 'pending', 'completed', 'failed'
    error_message TEXT
);
```

**Design**: Minimal schema. File location is source of truth, database tracks status only.

## Implementation Details

### Model Loading Fix
**Problem**: Models re-downloaded every run  
**Solution**: Use `snapshot_download()` for all presets to find cached models

```python
def _get_cached_model_path(model_id: str) -> str:
    return snapshot_download(model_id)  # Uses HF cache

def pick_model(preset: str = "et-large") -> WhisperModel:
    if preset == "et-large":
        path = _get_estonian_model_path("TalTechNLP/whisper-large-v3-turbo-et-verbatim")
        return WhisperModel(path, device="cuda", compute_type="int8_float16")
    # ... other presets use snapshot_download() too
```

### Language Detection
```python
language = "et" if preset.startswith("et-") else None
segments, info = model.transcribe(path, language=language, task="transcribe")
```

### Error Handling
- **GPU fail** → CPU fallback (automatic)
- **File fail** → Move to `failed/`, log error
- **Model fail** → Raise `ModelLoadError`

## Key Design Decisions

1. **File Location = Truth**: File in root → pending, in `processed/` → done, in `failed/` → failed
2. **Estonian Default**: Changed from English to `et-large` throughout codebase
3. **Alternative 1**: Simplest approach (no retry/concurrency). Future: Alt 2 (retry, workers) or Alt 3 (production-grade)
4. **Minimal Docs**: README → User guide → Technical details (no duplication)

## Testing

**Metrics**: 72 tests, 83% coverage, B+ grade  
**Refactoring**: Removed 295 lines of duplicate/low-value tests (D+ → B+)

```
tests/unit/ (64 tests)          # Core logic, mocked dependencies
tests/integration/ (8 tests)    # CLI workflows, real components
```

## Performance

- **Speed**: Estonian ~4-8x realtime (GPU), ~1-2x (CPU). English ~3-6x (GPU), ~1x (CPU)
- **Resources**: 4-8GB VRAM, model cache ~1.5-3GB per model
- **Example**: 40min audio → 1-5min processing

## Windows Integration

**Batch files**: `transcribe_estonian.bat`, `transcribe_english.bat`  
**Usage**: Copy to audio folder, double-click. Auto-processes all files.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Re-downloads model | Fixed via `snapshot_download()` |
| "Folder not found" | Use `/mnt/c/Users/...` for WSL |
| "No files found" | Check extensions, files must be in root folder |
| Stuck pending | Run `status --verbose`, or delete `.db` file |

## Future Enhancements

**Alt 2** (~5h): Auto-retry (3×), concurrency, timestamps  
**Alt 3** (~2wks): State machine, YAML config, migrations, audit trail, deduplication

## Dependencies

Already in `pyproject.toml`: `faster-whisper`, `huggingface_hub`, `ctranslate2==4.4.0`

## Validation

✅ Estonian/English models load correctly  
✅ Model caching works  
✅ Files move to correct folders  
✅ Database tracking accurate  
✅ Windows batch files functional  
✅ All 72 tests pass  
✅ E2E test: 40min audio → 80sec processing → 96KB JSON  

## References

- [Estonian Model](https://huggingface.co/TalTechNLP/whisper-large-v3-turbo-et-verbatim)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)

---

**Status**: ✅ Production-ready | **Version**: Alternative 1 | **Updated**: Dec 5, 2025
