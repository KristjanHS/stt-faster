# Diarization progress logging — unified format

**Status:** Design — ready to implement
**Date:** 2026-05-22
**Touches:** `backend/diarize/pyannote_runner.py` (only)

## Problem

Transcription and diarization currently emit progress lines in two different
shapes:

```
⌛ Transcription progress: 17.2/58.4 min (29.5%), elapsed 1.0 min
⌛ Diarization progress: elapsed 1.0 min on 58.4 min audio (1.7% wall/audio)
```

The diarization line was modelled on the assumption that pyannote runs as a
single opaque forward pass with no mid-stream signal — so the runner uses a
60s wall-clock heartbeat thread and reports `wall/audio %` as a proxy.

That assumption is wrong. Pyannote 4.x ships a hook protocol the pipeline
calls during execution with real per-stage `(completed, total)` updates.

## Goal

Replace the wall-clock proxy with hook-driven, stage-aware progress that
shares transcription's visual prefix (`⌛ {Phase} progress: …, elapsed N.N
min`) and 60s cadence — but reports what is actually progressing inside
pyannote rather than a synthesised audio-minutes value.

Non-goal: making the two lines literally identical. Diarization's natural
progress unit is **stages**, not audio time; pretending otherwise would
trade an honest "elapsed/audio" proxy for a dishonest "processed/total" one.

## Background: pyannote's hook protocol

Found in `.venv/lib/python3.12/site-packages/pyannote/audio/pipelines/utils/hook.py`:

```python
def __call__(self,
             step_name: Text,
             step_artifact: Any,
             file: Optional[Mapping] = None,
             total: Optional[int] = None,
             completed: Optional[int] = None):
```

The pipeline calls the hook:
- Once per stage entry with `completed=None, total=None` (transition).
- Repeatedly within long stages with monotonic `(completed, total)` ints.

`community-1` stages (per the model card + WeSpeaker/VBx citations):
**speaker_segmentation → embeddings → speaker_counting → discrete_diarization**.
Embeddings extraction is the long pole and is the stage that emits
fine-grained `(completed, total)` updates.

Pyannote's bundled `ProgressHook` uses Rich's `Progress` UI. That is not
compatible with this project's `LOGGER.info`-only console policy (CLAUDE.md
critical rule #3 — no `print`, ruff T201 enforced). We supply our own hook
that emits via `LOGGER` instead.

## Design

### Public surface

`run_pyannote(audio_path, *, num_speakers=2, audio_duration=None)` keeps
its current signature. `audio_duration` is retained for the opening
`🎙️ Starting diarization …` line (logged by the caller, not this file)
and as context in the new progress lines.

### Hook implementation

A new class `_DiarizeProgressHook` replaces `_DiarizeHeartbeat`:

- Used as a context manager, like `_DiarizeHeartbeat` today.
- Records `start_time` on `__enter__`.
- `__call__(step_name, step_artifact, file=None, total=None, completed=None)`
  signature matches pyannote's exactly so `pipeline(..., hook=self)` works.
- Throttles output to one log line per `PROGRESS_LOG_INTERVAL_SECONDS = 60.0`,
  matching transcription's interval constant — exposed locally in
  `pyannote_runner.py` (no cross-module import; the value is duplicated
  with a comment pointing at `backend/transcribe.py`).
- Always logs on **step transition** (regardless of throttle) — these are
  rare (≤4 per file) and mark the most informative boundary in the run.

### Log line shape

```
⌛ Diarization progress: {step_name} {completed}/{total} ({pct}%), elapsed {N.N} min
```

When the stage has no quantitative progress (transition events, or stages
that never emit `(completed, total)`):

```
⌛ Diarization progress: {step_name}, elapsed {N.N} min
```

Step names are passed through from pyannote (`speaker_segmentation`,
`embeddings`, `speaker_counting`, `discrete_diarization`) — no remapping.
They are already snake_case and self-describing; any rename layer would be
dead weight and would silently break when pyannote adds or renames a stage.

### Wiring into `run_pyannote`

```python
with _DiarizeProgressHook(audio_duration) as hook:
    diarization: Any = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=num_speakers,
        hook=hook,
    )
```

The `_DiarizeHeartbeat` class is deleted entirely — no fallback thread.
If pyannote ever ships a version that stops calling the hook, the absence
of progress lines is the right signal (we should investigate), not a
silent regression to wall-clock estimation.

### Completion line

Unchanged — the existing `✅ Diarization complete: elapsed N.N min` line
emitted by the caller after `run_pyannote` returns is the natural mirror
of `✅ Transcription complete: …`. Out of scope for this change.

## Why not the alternatives

**Synthesize a unified "processed audio min" across stages** (computed by
weighting stages by their typical RTF share, then projecting current stage
progress onto an overall audio-minutes axis). Would yield a line
visually identical to transcription's. Rejected: the weights are an
estimate, they drift with model versions, and the resulting number is not
an audio-minutes measurement in any defensible sense — it would mean "if
the run were on track, you'd be roughly N minutes in". Cleaner to be
honest that diarization progress is per-stage.

**Keep the wall-clock heartbeat, only reshape the surface text.** Smallest
change but throws away the real signal pyannote already provides.
Embeddings extraction on a 1h file produces ~80–100 hook calls; ignoring
all of them in favour of an elapsed-vs-audio ratio is strictly worse than
sampling them at 60s.

## Decisions log

1. **Stage names passed through unchanged.** Rationale: pyannote-side
   strings are already log-friendly; a translation layer adds maintenance
   cost (silent breakage on pyannote stage rename) for no clarity gain.
2. **No fallback heartbeat thread.** Rationale: pyannote 4.x is the only
   supported version (per `pyproject.toml` constraints from the
   community-1 migration). If hook calls disappear, that is the signal —
   not something to mask.
3. **60s throttle matches transcription.** Rationale: deliberate cadence
   parity is the user-facing point of the change. Stepping outside this
   interval (e.g. logging every hook call) would re-introduce console
   noise.
4. **Hook implemented as a context manager.** Rationale: drop-in
   replacement for `_DiarizeHeartbeat`'s existing `with … :` block; no
   change to the surrounding `try/except DiarizationRuntimeError` shape.

## Out of scope

- Changing the opening `🎙️ Starting diarization …` line (lives in
  `backend/diarize/pipeline.py` or its caller; not part of this file).
- The completion line (`✅ Diarization complete: …`).
- ETA computation. Pyannote does not give us an audio-minutes axis, so
  any ETA we computed across all four stages would be the same kind of
  weighted-RTF estimate rejected above.
- Threading model around the hook. Pyannote calls the hook from the same
  thread that runs the pipeline — there is no concurrency we need to
  handle.

## Implementation sketch (for the follow-up plan)

Single-file change in `backend/diarize/pyannote_runner.py`:

1. Delete `_DiarizeHeartbeat` (class + its constant
   `DIARIZE_HEARTBEAT_INTERVAL_SECONDS`).
2. Add `_DiarizeProgressHook` per the design above (~40 lines).
3. Pass `hook=hook` in the `pipeline(...)` call.
4. Update the small e2e fixture's expected log lines if a snapshot assert
   exists (a `grep -rn "Diarization progress" tests/` sweep before the
   edit will confirm; current evidence suggests the e2e checks only `✅
   Diarization complete`).

No public API changes. No new dependencies.

## Verification

- Unit test the hook in isolation: instantiate, call with a sequence of
  `(step_name, None, None, completed, total)` tuples, monkey-patch
  `time.time` to control throttling, assert on captured LOGGER output.
  This is the gating case for cadence behaviour and is cheap to write.
- e2e run against `tests/test.mp3` (the trimmed 2-speaker fixture, per
  [project state memory](file:///home/kristjans/.claude/projects/-home-kristjans-projects-stt-faster/memory/project_state.md)).
  Confirm at least one `embeddings` line with `M/N (P%)` appears on a
  short file, and that transitions log instantly.
- Visual diff against transcription: paste both progress sequences from
  one run, eyeball that the prefix/suffix shape matches.
