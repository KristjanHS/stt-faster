# Speech-to-Text Pipeline Stages

This document lists all stages in the speech-to-text pipeline with their parameters and current values.

## Pipeline Stages

| Stage | Parameter | Current Value | C value | Comments |
|-------|-----------|---------------|---------|----------|
| **Audio Inspection** | Inspects | Channels, sample rate, duration, bit depth | Channels, sample rate, duration, bit depth | |
| **Preprocessing** | Enabled | `True` | `False` | env: `STT_PREPROCESS_ENABLED` |
| **Preprocessing** | Target sample rate | `16000` Hz | `16000` Hz | env: `STT_PREPROCESS_TARGET_SR` |
| **Preprocessing** | Target channels | `None` | `1` | Auto/inherit, env: `STT_PREPROCESS_TARGET_CH` |
| **Preprocessing** | Profile | `cpu` | `auto` | env: `STT_PREPROCESS_PROFILE` |
| **Preprocessing** | Temp directory | `None` | `None` | System default, env: `STT_PREPROCESS_TMP_DIR` |
| **Preprocessing** | Loudnorm preset | `default` | N/A | env: `STT_PREPROCESS_LOUDNORM_PRESET` (didn't exist) |
| **Downmix/Resample** | Target sample rate | `16000` Hz | `16000` Hz | From config |
| **Downmix/Resample** | Target channels | Auto/inherit or `1` | `1` | Resolved from config |
| **Downmix/Resample** | Resampler | `soxr` | Default (not specified) | Hardcoded (ffmpeg) |
| **Downmix/Resample** | Sample format | `s16` | `s16` | Hardcoded (16-bit signed) |
| **Loudness Normalization** | Preset | `default` | N/A | Didn't exist (hardcoded values) |
| **Loudness Normalization** | Target loudness (I) | `-20.0` LUFS | `-18.0` LUFS | From "default" preset |
| **Loudness Normalization** | True peak (TP) | `-2.0` dB | `-1.5` dB | From "default" preset |
| **Loudness Normalization** | Backend | `pyloudnorm` | `ffmpeg loudnorm` | pyloudnorm only, no fallback |
| **Light Denoise** | Method | `spectral_gate` | `spectral_gate` | Hardcoded default |
| **Light Denoise** | Library | `noisereduce` | `noisereduce` | Hardcoded |
| **SNR Estimation** | Method | `estimate_snr_db` | `estimate_snr_db` | Hardcoded |
| **Model Loading** | Default preset | `et-large` | `et-large` | Function parameter default |
| **Model Loading** | Device | `cuda` | `cuda` | From preset config |
| **Model Loading** | Compute type | `float16`, `int8_float16`, `int8` | `float16`, `int8_float16`, `int8` | From preset config |
| **Model Loading** | Estonian model | `TalTechNLP/whisper-large-v3-turbo-et-verbatim` | `TalTechNLP/whisper-large-v3-turbo-et-verbatim` | Hardcoded per preset |
| **Model Loading** | Turbo model | `Systran/faster-distil-whisper-large-v3` | `Systran/faster-distil-whisper-large-v3` | Hardcoded per preset |
| **Model Loading** | Large model | `Systran/faster-whisper-large-v3` | `Systran/faster-whisper-large-v3` | Hardcoded per preset |
| **Transcription** | Beam size | `5` | `5` | DEFAULT_BEAM_SIZE constant |
| **Transcription** | Word timestamps | `False` | `False` | DEFAULT_WORD_TIMESTAMPS constant |
| **Transcription** | Task | `transcribe` | `transcribe` | DEFAULT_TASK constant |
| **Transcription** | Language | Auto-detect or forced | `None` (auto-detect) or `et` (if preset starts with "et-") | Default: auto-detect, forced based on preset |
| **Transcription** | Estonian preset language | `et` | `et` | Forced when preset starts with "et-" |
| **Output Generation** | Format | `txt` | `JSON` | DEFAULT_OUTPUT_FORMAT constant |
| **Output Generation** | Float precision | `3` | `3` | FLOAT_PRECISION constant |


## Model Presets

| Preset | Model | Device | Compute Type | Language |
|--------|-------|--------|--------------|----------|
| `et-large` | TalTechNLP/whisper-large-v3-turbo-et-verbatim | cuda | float16 | Estonian |
| `et-32` | TalTechNLP/whisper-large-v3-turbo-et-verbatim | cpu | float32 | Estonian |
| `turbo` | Systran/faster-distil-whisper-large-v3 | cuda | float16 | Multi |
| `distil` | Systran/faster-distil-whisper-large-v3 | cuda | float16 | Multi |
| `large8gb` | Systran/faster-whisper-large-v3 | cuda | int8_float16 | Multi |
| `small` | small | cpu | int8 | Multi |

## Loudness Presets

| Preset | Target Loudness (I) | True Peak (TP) |
|--------|---------------------|----------------|
| `default` | -20.0 LUFS | -2.0 dB |
| `boost-quiet-voices` | -18.0 LUFS | -2.0 dB |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_PREPROCESS_ENABLED` | `True` | Enable/disable preprocessing pipeline |
| `STT_PREPROCESS_TARGET_SR` | `16000` | Target sample rate in Hz |
| `STT_PREPROCESS_TARGET_CH` | `None` | Target channels (None = auto/inherit) |
| `STT_PREPROCESS_PROFILE` | `cpu` | Processing profile (cpu/gpu/auto) |
| `STT_PREPROCESS_TMP_DIR` | `None` | Temporary directory for preprocessing |
| `STT_PREPROCESS_OUTPUT_DIR` | `None` | Directory to save intermediate preprocessing stage outputs for quality debugging |
| `STT_PREPROCESS_LOUDNORM_PRESET` | `default` | Loudness normalization preset |

