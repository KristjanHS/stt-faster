# Audio Pre-Processing Module — Plan

## Top simplifications to pursue
- [x] Keep a single default pipeline: implement only downmix/resample + loudnorm + light denoise for now. Heavy enhancer is paused; leave a note to reintroduce later if needed (via a simple flag).
- [x] Collapse profiles to “default CPU” with an optional “GPU heavy” switch; avoid per-step budget overrides unless they prove necessary.
- [x] Pick one loudness backend (ffmpeg `loudnorm`) and drop pluggable backends until there's evidence they’re needed.
- [ ] Keep dependency surface tiny: `soundfile` + `noisereduce` only; add `resemble-enhance` strictly behind the heavy extra; skip `librosa`/analysis extras unless required.
- [ ] Simplify config: a small config object with a handful of booleans/thresholds; avoid deep nesting and per-step knobs until a real use-case appears.
- [ ] Trim metrics: log wall-clock times and SNR deltas only; skip extra telemetry until stability and value are demonstrated.
- [ ] Restrict failure policy: fail fast on light steps; for heavy, single fallback to light path and move on—no broader retry/circuit complexity yet.
- [ ] Minimize temp handling surface: one `TemporaryDirectory` per run, fixed cap, delete intermediates—no per-step temp roots or quota tuning until needed.
- [ ] Keep tests lean: a handful of synthetic clips + the tiny golden set; defer broad E2E permutations until the core path is stable.
- [ ] Limit integration points: wire preprocessing through a single orchestrator call in `transcribe` without additional CLI flags initially; add toggles later if demanded.

Goal: add a decoupled audio pre-processing pipeline (with a Python orchestrator) that prepares audio before Whisper. The pipeline should be callable from the transcription code but remain independently testable and replaceable.

## Latency/resource decisions (simplified)
- Profile: `default_cpu` targeting ≤1x realtime for the light path (downmix/resample + loudnorm + light denoise). Heavy enhancer deferred.
- Config surface: simple toggle for preprocessing on/off; leave a commented hint for a future `enable_gpu_heavy` switch (no active logic).
- Observability: log per-step duration and total wall time; no budget logic for now.

## Quality bar decisions (option 1)
- Golden set: record 3–5 short clips covering clean speech, noisy cafe/office, and phone-quality audio. Match the deployment context (same mic, distance, environment) so improvements translate.
- Targets: aim for +3–5 dB SNR gain on noisy clips and no WER regression on the clean clip compared to baseline (no preprocessing).
- Checks: manual/automated spot-checks using the fixed Whisper checkpoint; log SNR/LUFS deltas per clip when running the pipeline.

## Failure handling decisions (option 2)
- Retries: single retry for transient ffmpeg failures; no retries for pure Python steps unless explicitly marked safe.
- Fallbacks: n/a for heavy enhancer (not implemented now); future heavy step would fall back to light output and set `fallback_used=true`.
- Metadata/telemetry: include per-step status (`ok`, `failed`, `skipped`), stderr snippet for failed external steps, and `step_failed` flag.
- Policy: if any light step fails, bubble the error with step context; future heavy step would degrade gracefully to light output.

## Safety & temp management decisions (option 1)
- Input validation: basic extension/mime sniff before processing; reject unsupported types early.
- Temp handling: per-run `TemporaryDirectory`; delete intermediates after completion; enforce a simple size cap (e.g., 500 MB) and abort if exceeded.
- ffmpeg hygiene: no untrusted filter strings; pass explicit input/output paths; avoid shell globbing.

## Dependency/packaging decisions (option 1, using uv)
- Default install (minimal): `soundfile`, `noisereduce`, and ffmpeg runtime documented. Keep this as the uv lock baseline.
- Optional heavy extra (deferred): keep `.[heavy]` (resemble-enhance) noted for future reintroduction; not installed or used in the current plan.
- Optional analysis extra (if needed): add `librosa` as `.[anal]` without making it default; keep Docker base slim with minimal deps only.

## Objectives
- Downmix/resample to Whisper-friendly format when needed (e.g., 1 channel, 16 kHz).
- Normalize loudness into a comfortable speech range (ffmpeg `loudnorm` or `ffmpeg-normalize`).
- Apply lightweight denoise by default (spectral gating with `timsainb/noisereduce`).
- Heavy enhancer: deferred; consider later via a simple flag if justified.
- Emit processed audio for the Whisper pipeline without altering existing transcription APIs.

## Architecture
- `backend/preprocess/` (new package)
  - `orchestrator.py`: high-level `preprocess_audio(input_path, config)`; orchestrates steps, handles temp files, metrics, error reporting.
  - `io.py`: helpers for loading/saving WAV/PCM, temp file management, format inspection (channels, sample rate, bit depth), and preservation of original filenames/paths.
  - `steps/`
    - `downmix_resample.py`: ffmpeg-based resample/downmix; no-op when already mono/16 kHz; configurable target rate/channels.
    - `loudness.py`: normalization with ffmpeg `loudnorm` or `ffmpeg-normalize`; pluggable backend selection.
    - `denoise_light.py`: spectral gating via `noisereduce` on waveform; optional parameterization for n_fft, prop_decrease, stationary vs. non-stationary.
    - `denoise_heavy.py`: deferred; keep a placeholder comment for future reintroduction if needed.
  - `config.py`: dataclasses for step toggles, temp directory control, backend selection, and a placeholder for future heavy enable flag.
  - `metrics.py`: utilities to estimate SNR/simple energy metrics to decide “problem file” status and to log per-step timing.
  - `errors.py`: step-specific exception types that carry context (file path, step name).
- Integration point: `backend/transcribe.py` (or `processor.py`) calls `preprocess_audio` before invoking Whisper; retains original API by writing processed audio to a temp path and passing that into transcribe.

## Pipeline Flow
1) Inspect input audio (channels, sample rate, duration, bit depth). Decide if resample/downmix is needed.
2) **Downmix/Resample (conditional)**: `ffmpeg -i in.wav -ac {target_channels} -ar {target_sr} tmp.wav` (fail-fast on errors).
3) **Loudness Normalize**: apply `loudnorm` (two-pass if feasible, else single-pass). Keep `ffmpeg-normalize` only as a note/TODO if loudnorm proves brittle.
4) **Light Denoise (default)**: load waveform (librosa/soundfile), run `noisereduce.spectral_gate`.
5) **Quality Check**:
   - Compute simple SNR/energy ratio and heuristic speech-presence check for logging/analysis.
6) Emit final WAV path (temp location), plus metadata (applied steps, timing, SNR before/after). Clean up intermediates.

## Config & Defaults
- Target format: mono, 16 kHz, 16-bit WAV.
- Loudness: default to `loudnorm` with target IL = -18 LUFS (tunable).
- Light denoise: enabled by default.
- Heavy enhancer: deferred; add back later via a simple flag if justified.
- Latency profile: default CPU-only, target ≤1x realtime for the light path.
- Temp handling: use `TemporaryDirectory` with cleanup; allow override for debugging.
- Logging: structured info per step (duration, backend used, before/after SNR when available); no budget flags.

## Dependencies to Add
- `soundfile` (for WAV IO) and `librosa` or `torchaudio` (for analysis/loading; pick one to minimize weight).
- `noisereduce` (`timsainb/noisereduce`) for spectral gating.
- `ffmpeg` runtime dependency (documented; not a Python package).
- `resemble-enhance` (future optional heavy enhancer; behind extra/dependency group and currently unused).

## Testing Strategy
- Unit tests per step:
  - Downmix/resample: verify channel/sample rate changes and no-op when already correct.
  - Loudness: assert LUFS moves toward target on synthetic tones/speech-like noise.
  - Light denoise: ensure noise floor drops on noisy synthetic clip without erasing tone.
- Orchestrator tests:
  - Happy path through all enabled steps; confirm temp cleanup and metadata.
  - Error propagation: failure in a step surfaces with context; earlier outputs cleaned.
- E2E integration:
  - Pipeline produces a temp WAV fed into Whisper mock/fixture; verify call uses processed path.

## Rollout Plan
- Phase 1: build package skeleton, add light-path step stubs, add dependencies, and unit tests.
- Phase 2: integrate orchestrator behind a feature flag in processor/transcribe; default on for light path (downmix+normalize+light denoise).
- Phase 3: benchmarks on sample noisy/clean clips; tune defaults; document in README.
- Phase 4 (future): consider reintroducing heavy enhancer behind a simple flag if justified by results.
