# Speech-to-Text Pipeline Stages

This document lists all stages in the speech-to-text pipeline with their parameters and current values.

## Pipeline Stages

| Stage | Parameter | Current Value | C value | Comments |
|-------|-----------|---------------|---------|----------|
| **Preprocessing** | Target sample rate | `16000` Hz | `16000` Hz | env: `STT_PREPROCESS_TARGET_SR` |
| **Preprocessing** | Target channels | `None` | `1` | Auto/inherit, env: `STT_PREPROCESS_TARGET_CH` |
| **Preprocessing** | Profile | `cpu` | `auto` | env: `STT_PREPROCESS_PROFILE` |
| **Preprocessing** | Loudnorm preset | `default` | N/A | env: `STT_PREPROCESS_LOUDNORM_PRESET` |
| **Preprocessing** | RNNoise model | `models/general.rnnn` | `models/general.rnnn` | env: `STT_PREPROCESS_RNNOISE_MODEL` (class default: `models/sh.rnnn`) |
| **Preprocessing** | RNNoise mix | `0.5` | `0.5` | Denoising strength (0.0-1.0), env: `STT_PREPROCESS_RNNOISE_MIX` |
| **Downmix/Resample** | Target sample rate | `16000` Hz | `16000` Hz | From config |
| **Downmix/Resample** | Target channels | Auto/inherit or `1` | `1` | Resolved from config |
| **Downmix/Resample** | Resampler | `soxr` | Default (not specified) | Hardcoded (ffmpeg) |
| **Downmix/Resample** | Sample format | `s16` | `s16` | Hardcoded (16-bit signed) |
| **Loudness Normalization** | Preset | `default` | N/A | From config |
| **Loudness Normalization** | Target loudness (I) | `-20.0` LUFS | `-18.0` LUFS | From "default" preset |
| **Loudness Normalization** | True peak (TP) | `-2.0` dB | `-1.5` dB | From "default" preset |
| **Loudness Normalization** | LRA | `7.0` | N/A | Loudness range, from preset |
| **Loudness Normalization** | Backend | `ffmpeg loudnorm` | `ffmpeg loudnorm` | Via ffmpeg-python library, using ffmpeg's built-in loudnorm filter |
| **Light Denoise** | Method | `spectral_gate` | `spectral_gate` | Hardcoded default |
| **Light Denoise** | Library | `noisereduce` | `noisereduce` | Hardcoded |
| **SNR Estimation** | Method | `estimate_snr_db` | `estimate_snr_db` | Hardcoded |
| **Model Loading** | Compute type | `float16`, `int8_float16`, `int8` | `float16`, `int8_float16`, `int8` | From preset config |
| **Model Loading** | Estonian model | `TalTechNLP/whisper-large-v3-turbo-et-verbatim` | `TalTechNLP/whisper-large-v3-turbo-et-verbatim` | Hardcoded per preset |
| **Model Loading** | Turbo model | `Systran/faster-distil-whisper-large-v3` | `Systran/faster-distil-whisper-large-v3` | Hardcoded per preset |
| **Model Loading** | Large model | `Systran/faster-whisper-large-v3` | `Systran/faster-whisper-large-v3` | Hardcoded per preset |
| **Transcription** | Beam size | `5` | `5` | env: `STT_TRANSCRIBE_BEAM_SIZE` |
| **Transcription** | Patience | `1.1` | `1.0` | Beam search patience, env: `STT_TRANSCRIBE_PATIENCE` |
| **Transcription** | Word timestamps | `False` | `False` | env: `STT_TRANSCRIBE_WORD_TIMESTAMPS` |
| **Transcription** | Chunk length | `20` s | `30` s | Audio chunk length in seconds, env: `STT_TRANSCRIBE_CHUNK_LENGTH` |
| **Transcription** | Language | Auto-detect or forced | `None` (auto-detect) or `et` (if preset starts with "et-") | Default: auto-detect, forced based on preset |
| **Transcription** | VAD filter | `True` | `True` | Enable voice activity detection, env: `STT_TRANSCRIBE_VAD_FILTER` |
| **Transcription** | VAD threshold | `0.35` | `0.4` | Speech probability threshold, env: `STT_TRANSCRIBE_VAD_THRESHOLD` |
| **Transcription** | VAD min speech duration | `250` ms | `250` ms | Minimum speech segment duration, env: `STT_TRANSCRIBE_VAD_MIN_SPEECH_DURATION_MS` |
| **Transcription** | VAD max speech duration | `inf` s | `inf` s | Maximum speech segment duration, env: `STT_TRANSCRIBE_VAD_MAX_SPEECH_DURATION_S` |
| **Transcription** | VAD min silence duration | `800` ms | `800` ms | Minimum silence between segments, env: `STT_TRANSCRIBE_VAD_MIN_SILENCE_DURATION_MS` |
| **Transcription** | VAD speech pad | `300` ms | `200` ms | Padding around speech segments, env: `STT_TRANSCRIBE_VAD_SPEECH_PAD_MS` |
| **Transcription** | Temperature | `[0.0, 0.2, 0.4, 0.8]` | `[0.0, 0.2, 0.5]` | Temperature for sampling (single float or list), env: `STT_TRANSCRIBE_TEMPERATURE` |
| **Transcription** | Temperature increment | `0.2` | `0.2` | Increment on fallback, env: `STT_TRANSCRIBE_TEMPERATURE_INCREMENT` |
| **Transcription** | Best of | `5` | `5` | Number of candidates for non-zero temperature, env: `STT_TRANSCRIBE_BEST_OF` |
| **Transcription** | Compression ratio threshold | `2.4` | `2.4` | Quality threshold, env: `STT_TRANSCRIBE_COMPRESSION_RATIO_THRESHOLD` |
| **Transcription** | Logprob threshold | `-1.0` | `-1.0` | Quality threshold, env: `STT_TRANSCRIBE_LOGPROB_THRESHOLD` |
| **Transcription** | No speech threshold | `0.5` | `0.5` | Threshold for detecting no speech, env: `STT_TRANSCRIBE_NO_SPEECH_THRESHOLD` |
| **Transcription** | Length penalty | `1.0` | `1.0` | Length penalty for decoding, env: `STT_TRANSCRIBE_LENGTH_PENALTY` |
| **Transcription** | Repetition penalty | `1.0` | `1.0` | Penalty for repetition, env: `STT_TRANSCRIBE_REPETITION_PENALTY` |
| **Transcription** | No repeat ngram size | `3` | `3` | N-gram size for repetition prevention, env: `STT_TRANSCRIBE_NO_REPEAT_NGRAM_SIZE` |
| **Transcription** | Suppress tokens | `"-1"` | `"-1"` | Token IDs to suppress (comma-separated or "-1" for default), env: `STT_TRANSCRIBE_SUPPRESS_TOKENS` |
| **Transcription** | Condition on previous text | `True` | `True` | Use previous text as context, env: `STT_TRANSCRIBE_CONDITION_ON_PREVIOUS_TEXT` |
| **Transcription** | Initial prompt | `None` | `None` | Initial text prompt for transcription, env: `STT_TRANSCRIBE_INITIAL_PROMPT` |
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

| Preset | Target Loudness (I) | True Peak (TP) | LRA |
|--------|---------------------|----------------|-----|
| `default` | -20.0 LUFS | -2.0 dB | 7.0 |
| `boost-quiet-voices` | -18.0 LUFS | -2.0 dB | 7.0 |
