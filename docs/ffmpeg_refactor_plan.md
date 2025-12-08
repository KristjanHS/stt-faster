# FFMPEG Refactor Plan

## 1. Goal

The goal is to switch from `pyloudnorm` to `ffmpeg-python` for audio normalization. All ffmpeg operations should be performed in a single step for efficiency.

## 2. Current State

- The audio preprocessing pipeline is orchestrated in `backend/preprocess/orchestrator.py`.
- The `preprocess_audio` function calls `_run_pipeline`, which defines the sequence of operations.
- The current pipeline uses a single ffmpeg command to perform resampling and loudness normalization. This is implemented in `backend/preprocess/steps/ffmpeg_pipeline.py`.
- The `run_ffmpeg_pipeline` function constructs a filter graph with `aresample` and `loudnorm` and executes it using `subprocess.run`.
- The entire audio file is processed at once. There is no concept of per-segment processing in the current preprocessing pipeline.
- The `pyloudnorm` library is no longer used in the codebase.

## 3. Implementation Status: ✅ COMPLETED

### Completed Changes:

1. **Switched to `ffmpeg-python` library**: 
   - Added `ffmpeg-python>=0.2.0,<1.0.0` to `pyproject.toml` dependencies
   - Refactored `run_ffmpeg_pipeline` to use `ffmpeg-python` instead of `subprocess.run`
   - All ffmpeg operations now use the Pythonic `ffmpeg-python` API
   - The entire audio file is processed in a single pass for efficiency

2. **Updated tests**:
   - Refactored unit tests to work with `ffmpeg-python` mocking
   - Added tests for filter graph construction and preset selection

### Key Implementation Details:

- The `run_ffmpeg_pipeline` function now uses `ffmpeg-python` for all operations
- The entire audio file is processed in a single pass for maximum efficiency
- The implementation maintains backward compatibility with the existing API
- All operations (resampling, filtering, normalization) are performed efficiently within a single ffmpeg pipeline
