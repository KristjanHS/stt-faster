# Audio Transcription - Technical Reference

Batch audio transcription system with Estonian and multilingual Whisper models, featuring flexible variant configurations for preprocessing and transcription parameter optimization.

## Overview

The system provides automated batch transcription of audio files with:
- **Multiple model presets** (Estonian fine-tuned, multilingual, fast/slow variants)
- **Variant system** (16 pre-configured combinations of preprocessing and transcription parameters)
- **Automatic file management** (processed/failed folder organization)
- **Database tracking** (SQLite-based history and metrics)
- **Flexible output formats** (TXT, JSON, or both)

## Models

### Estonian (Default)
**TalTechNLP/whisper-large-v3-turbo-et-verbatim** - Fine-tuned on 1400h Estonian audio, ~1.5GB, preset: `et-large`

### English/Multilingual
| Preset | Model | Size | Language | Use Case |
|--------|-------|------|----------|----------|
| `large8gb` | Whisper large-v3 | ~3GB | Multi | Highest accuracy |
| `turbo` | Whisper large-v3-turbo | ~1.5GB | Multi | Fast + accurate |
| `distil` | Distilled Whisper | ~756MB | English | Fastest |
| `et-32` | Estonian 32-bit | ~1.5GB | Estonian | CPU-optimized |

**Caching**: All models use `snapshot_download()` for HuggingFace cache (`~/.cache/hf/hub/`). Download once, reuse forever.

## Variant System

The variant system allows selecting from 16 pre-configured combinations of preprocessing steps and transcription parameters. This enables systematic comparison and optimization of transcription quality.

### Variant Categories

**Transcription Presets:**
- **`project`**: Project-specific defaults (optimized for Estonian)
- **`industry`**: Industry-standard faster-whisper defaults
- **`minimal`**: Minimal parameters, letting faster-whisper use internal defaults

**Preprocessing Options:**
- **No preprocessing**: Raw audio input
- **FFmpeg pipeline**: Resampling, normalization, optional denoising
- **Denoise only**: Light denoising with resampling
- **Lightweight normalization**: Specialized steps (loudnorm_only, loudnorm_highpass, dynaudnorm)
- **Custom preprocessing**: Advanced variants (12-16) with specialized pipelines

### Available Variants (1-16)

| # | Name | Preprocessing | Transcription | Description |
|---|------|----------------|---------------|-------------|
| 1 | `no_preprocessing` | None | Project defaults | Baseline with project config |
| 2 | `noprep_minimal` | None | Minimal defaults | Baseline with minimal config |
| 3 | `ffmpeg_only` | FFmpeg pipeline | Project defaults | Normalization only |
| 4 | `denoise_only` | Denoise light | Project defaults | Denoising only |
| 5 | `ffmpeg_minimal` | FFmpeg pipeline | Minimal defaults | Normalization + minimal config |
| 6 | `full_minimal` | FFmpeg + denoise | Minimal defaults | Full preprocessing + minimal |
| 7 | `baseline_raw_defaults` | None | Baseline | **No preprocessing, true baseline (raw whisper defaults, no parameter overrides)** |
| 8 | `normonly_noparamtrans` | FFmpeg pipeline | Minimal | Normalization + minimal params |
| 9 | `onlyden_noparamtrans` | Denoise light | Minimal | Denoising + minimal params |
| 10 | `norm_highp_noparamtrans` | Loudnorm + highpass | Minimal | Lightweight normalization with highpass filter |
| 11 | `norm_dynaud_noparamtrans` | Dynamic audio normalization | Minimal | Dynamic normalization (no loudnorm) |
| 12-16 | Custom preprocessing | Various | Minimal | Advanced preprocessing pipelines (custom functions) |

**Note**: Variant 7 is particularly useful for fast processing without VAD filtering. Variants 1-11 use declarative preprocessing steps, while variants 12-16 use custom preprocessing functions for specialized audio processing scenarios.

## CLI Usage

### Basic Commands

```bash
# Process folder with default settings (Estonian model)
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio

# Check processing status
.venv/bin/python scripts/transcribe_manager.py status

# Verbose status with file details
.venv/bin/python scripts/transcribe_manager.py status --verbose
```

### Advanced Options

```bash
# Use different model preset
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio --preset large8gb

# Force language detection
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio --language et

# Output format options
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio --output-format json
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio --output-format both

# Use variant (NEW!)
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio --variant 7
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio --preset et-32 --variant 7 --language et
```

### Command Options

**`process` command:**
- `input_folder` (required): Path to folder containing audio files
- `--preset`: Model preset (`turbo`, `distil`, `large8gb`, `et-large`, `et-32`, `small`)
- `--language`: Force language code (e.g., `en`, `et`, `ru`). Auto-detect if not specified
- `--output-format`: Output format (`txt`, `json`, `both`) - default: `txt`
- `--variant`: Variant number (1-16) - default: uses standard configuration
- `--db-path`: Custom database path (default: XDG data home)

**`status` command:**
- `--verbose`: Show detailed file list with status markers
- `--db-path`: Custom database path

## Architecture

### Core Components

1. **`TranscriptionProcessor`** (`backend/processor.py`)
   - Manages file lifecycle (scan, process, move to processed/failed)
   - Handles variant-aware transcription when variant is specified
   - Records metrics and run statistics to database

2. **Variant System** (`backend/variants/`)
   - **`registry.py`**: Defines 16 built-in variants (1-11 use declarative steps, 12-16 use custom functions)
   - **`executor.py`**: Executes variants with preprocessing and transcription
   - **`preprocess_steps.py`**: Composable preprocessing step system with support for: ffmpeg, denoise, resample, loudnorm_only, loudnorm_highpass, dynaudnorm, denoise_custom
   - **`transcription_presets.py`**: Transcription configuration presets (project, industry, minimal)

3. **Transcription Engine** (`backend/transcribe.py`)
   - Core transcription function with metrics collection
   - Supports custom preprocessing runners and transcription configs
   - Handles model loading, caching, and transcription execution

4. **Database** (`backend/database.py`)
   - SQLite-based tracking of processed files
   - Records run metadata and file-level metrics
   - Provides status summaries and history

### File Flow

```
Input Folder/
├── audio1.wav          → Processing
├── audio2.mp3          → Processing
└── processed/         → Completed files + transcripts
└── failed/            → Failed files
```

**Status determination**: File location is the source of truth:
- Files in input folder → will be processed
- Files in `processed/` → already completed
- Files in `failed/` → previously failed

### Supported Formats

- `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.wma`

## Dependencies

Core dependencies (in `pyproject.toml`):
- `faster-whisper`: Fast Whisper implementation via CTranslate2
- `huggingface_hub`: Model downloading and caching
- `ctranslate2==4.4.0`: Inference engine
- `ffmpeg-python`: Audio preprocessing (for variants with preprocessing)

## Output Formats

### TXT Format
Plain text file with one segment per line:
```
Segment 1 text
Segment 2 text
...
```

### JSON Format
Structured output with full metadata:
```json
{
  "audio": "filename.wav",
  "language": "et",
  "language_probability": 0.99,
  "duration": 240.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "Segment text"
    }
  ]
}
```

### Both Format
Generates both `.txt` and `.json` files for the same audio file.

## Validation

✅ Estonian/English models load correctly  
✅ Model caching works  
✅ Files move to correct folders  
✅ Database tracking accurate  
✅ Windows batch files functional  
✅ Variant system integrated (16 variants: 1-11 declarative, 12-16 custom)  
✅ All unit tests pass (103 tests)  
✅ E2E test: 40min audio → 80sec processing → 96KB JSON  

## Performance

Typical processing speeds:
- **Fast preset** (`distil`): ~1-2 seconds per minute of audio
- **Balanced** (`turbo`, `et-large`): ~2-3 seconds per minute
- **Accurate** (`large8gb`): ~3-5 seconds per minute

Speed varies based on:
- Model preset (larger = slower)
- Audio quality (preprocessing overhead)
- Variant selection (minimal params = faster)
- Hardware (CPU vs GPU)

## References

- [Estonian Model](https://huggingface.co/TalTechNLP/whisper-large-v3-turbo-et-verbatim)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)

---

**Status**: ✅ Production-ready | **Version**: 2.0 (with Variant System) | **Updated**: Dec 9, 2025 (Phase 4.0: Step types extended, variants 10-11 migrated to declarative)
