# FFMPEG Refactor Plan

## 1. Goal

The goal is to switch from `pyloudnorm` to `ffmpeg-python` for audio normalization, with the ability to process each audio segment independently to amplify it to the maximum allowed level. All ffmpeg operations should be performed in a single step for efficiency.

## 2. Current State

- The audio preprocessing pipeline is orchestrated in `backend/preprocess/orchestrator.py`.
- The `preprocess_audio` function calls `_run_pipeline`, which defines the sequence of operations.
- The current pipeline uses a single ffmpeg command to perform resampling and loudness normalization. This is implemented in `backend/preprocess/steps/ffmpeg_pipeline.py`.
- The `run_ffmpeg_pipeline` function constructs a filter graph with `aresample` and `loudnorm` and executes it using `subprocess.run`.
- The entire audio file is processed at once. There is no concept of per-segment processing in the current preprocessing pipeline.
- The `pyloudnorm` library is no longer used in the codebase.
