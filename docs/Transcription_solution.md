# Audio Transcription - Technical Reference

Batch audio transcription system with Estonian and multilingual Whisper models.

## Models

### Estonian (Default)
**TalTechNLP/whisper-large-v3-turbo-et-verbatim** - Fine-tuned on 1400h Estonian audio, ~1.5GB, preset: `et-large`

### English/Multilingual
| Preset | Model | Size | Language | Use Case |
|--------|-------|------|----------|----------|
| `large8gb` | Whisper large-v3 | ~3GB | Multi | Highest accuracy |
| `turbo` | Whisper large-v3-turbo | ~1.5GB | Multi | Fast + accurate |
| `distil` | Distilled Whisper | ~756MB | English | Fastest |

**Caching**: All models use `snapshot_download()` for HuggingFace cache (`~/.cache/hf/hub/`). Download once, reuse forever.

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
