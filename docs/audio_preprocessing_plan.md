# Audio Pre-Processing Module — Plan

Goal: add a decoupled audio pre-processing pipeline (Python orchestrator) callable from transcription while staying independently testable and replaceable.

## Rollout (5 iterations)
- [x] Iteration 1 — Minimal skeleton: add `backend/preprocess/` with orchestrator + IO, default CPU profile, temp dir handling, enabled flag, downmix/resample only, fail-fast errors, per-step/total timing logs; wire a single call in `transcribe` behind a feature flag.
- [ ] Iteration 2 — Add loudnorm: keep loudness to ffmpeg `loudnorm` (two-pass if feasible, else single-pass); note `ffmpeg-normalize` only as a future fallback; keep metrics minimal (total wall time + overall SNR delta).
- [ ] Iteration 3 — Add light denoise: use `noisereduce.spectral_gate`; keep failure policy fail-fast; maintain minimal deps (`soundfile`, `noisereduce`, ffmpeg runtime); keep heavy enhancer deferred.
- [ ] Iteration 4 — Quality bar + tests: record tiny golden set (3–5 clips) and run manual spot-checks (+3–5 dB SNR gain on noisy, no WER regression on clean); add unit tests for light steps and one E2E smoke with a synthetic noisy clip into a Whisper mock.
- [ ] Iteration 5 — Future-ready hooks: keep commented hints for `enable_gpu_heavy` and heavy extra `.[heavy]`; keep optional `.[anal]` commented; add simple SNR/speech-presence logging for analysis; postpone automated LUFS/SNR per-clip and WER harness until the light path proves value.

## Decisions and defaults
- Pipeline: light only (downmix/resample → loudnorm → light denoise); heavy enhancer deferred for a future flag.
- Profile: `default_cpu` targeting ≤1x realtime; no budget logic; per-step + total timing logs.
- Config: single `enabled` flag; pipeline params hard-coded; future heavy flag commented.
- Loudness: ffmpeg `loudnorm`; `ffmpeg-normalize` only as a future fallback if loudnorm is brittle.
- Dependencies: default `soundfile`, `noisereduce`, ffmpeg runtime; optional `.[heavy]` (resemble-enhance) deferred; optional `.[anal]` (librosa) commented.
- Failure policy: no retries; fail fast on light steps; future heavy would fall back to light output.
- Temp: per-run `TemporaryDirectory` auto-deleted; no per-step roots or caps for now; basic input validation and ffmpeg hygiene (no globbing/untrusted filters).
- Logging/metrics: per-step duration/backend, total wall time, overall SNR delta; no per-step SNR/LUFS; no budgets.
- Quality bar: tiny golden set, +3–5 dB SNR gain on noisy clips, no WER regression on clean; manual spot-checks with fixed Whisper checkpoint; future automation deferred.
- Integration: single orchestrator call in `transcribe`; CLI/env flags commented for future.

## Architecture
- `backend/preprocess/`
  - `orchestrator.py`: `preprocess_audio(input_path, config)`; orchestrates steps, temps, metrics, error reporting.
  - `io.py`: WAV/PCM load/save, temp management, format inspection.
  - `steps/`: `downmix_resample.py` (ffmpeg), `loudness.py` (ffmpeg loudnorm; TODO fallback to ffmpeg-normalize if loudnorm brittle), `denoise_light.py` (noisereduce), `denoise_heavy.py` placeholder.
  - `config.py`: minimal (`enabled` flag; future heavy placeholder).
  - `metrics.py`: total wall time + overall SNR delta.
  - `errors.py`: step-specific exceptions with context.
- Integration: `backend/transcribe.py` calls `preprocess_audio` once before Whisper; writes processed temp path; flags deferred.

## Pipeline flow
1) Inspect input (channels, sample rate, duration, bit depth).
2) Downmix/resample if needed (`ffmpeg -ac {target_channels} -ar {target_sr}`) with fail-fast errors.
3) Loudnorm (two-pass if feasible; single-pass fallback; fallback to ffmpeg-normalize only if loudnorm proves brittle).
4) Light denoise (`noisereduce.spectral_gate` on waveform).
5) Quality check: simple SNR/energy + speech-presence for logging.
6) Emit temp WAV + metadata (steps, timing, SNR delta); clean intermediates.

## Config & defaults
- Target: mono, 16 kHz, 16-bit WAV; loudnorm IL -18 LUFS; light denoise on.
- Heavy enhancer: deferred for a future flag.
- Temp handling: per-run `TemporaryDirectory` with cleanup; allow override for debugging.
- Logging: structured info per step + overall SNR delta; no budgets or per-step SNR/LUFS.
- Toggle: `STT_PREPROCESS_ENABLED` (default off); `STT_PREPROCESS_TMP_DIR` optional for temp override.

## Testing (lean)
- Unit: downmix/resample no-op/changes; loudnorm moves LUFS toward target; denoise lowers noise floor without removing tone.
- E2E smoke: synthetic noisy clip through light pipeline into Whisper mock; verify processed path used.
- Manual QA: run tiny golden set through pipeline; check SNR/WER expectations with fixed Whisper checkpoint.

## Dependencies
- Default: `soundfile`, `noisereduce`, ffmpeg runtime.
- Optional heavy (deferred): `resemble-enhance` via `.[heavy]`.
- Optional analysis (commented): `librosa` via `.[anal]`.
