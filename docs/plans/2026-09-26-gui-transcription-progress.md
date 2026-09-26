# GUI transcription progress — structured channel

**Status:** rungs 1–3 shipped; backlog below is unordered, re-pick after each rung.

## Decision (2026-09-26)

Chose option C over (A) regex-parsing the `⌛ Transcription progress` log line and (B) faster-whisper's
`log_progress=True` tqdm bar: a dedicated `@@progress {json}` line on stdout, emitted by
`backend/progress.py` only when `STT_PROGRESS=1` (set by `gui.build_env`). Human logs are untouched;
the GUI reads only prefixed lines. A/B were rejected because they couple the GUI to a human/third-party
text format and cover only the transcribe stage.

Wire format: `@@progress {"file": 2, "files": 3, "stage": "diarize", "detail": "embeddings", "done": 31, "total": 90}`
— `done`/`total` omitted on a bare stage change, `detail` optional. Stage changes are never throttled;
`advance` is throttled to 1/s (`MIN_INTERVAL_SECONDS`). Bar = the current stage's own `done/total`
(resets per stage/detail; indeterminate on a bare stage); no overall % (ruled 2026-09-26, rung 2).

## Rung 1 (shipped)

- `ProgressReporter` + `parse_progress` in `backend/progress.py`.
- `TranscriptionProcessor.process_all_files` → `start_file(i, n)` (stage `prepare`).
- Both segment loops (`transcribe._collect_segments`, `executor._collect_executor_segments`) →
  `advance("transcribe", seconds_done, total)`.
- GUI: bar goes determinate on the first event; new detail label `38% · File 2/4 · Transcribing`;
  raw last-log line stays in the grey status label.
- Review fixes: forced end-of-file `advance` (the throttle dropped the last point); a `Retrying…` line
  resets the bar to indeterminate; `_finish` fills the bar only when `JobResult.ok`.

## Rung 2 (shipped) — diarize + stage names

- `transcribe()` emits `preprocess` → `load model` → `transcribe` → `diarize` stages via a defaulted `reporter`.
- `_DiarizeProgressHook` takes an `on_progress(step, completed, total)` callback (diarize stays free of
  `backend.progress`); `ProgressReporter.substeps("diarize")` adapts it to `stage/advance(…, detail=step)`.
- GUI: bar = `stage_fraction`, label `File 2/3 · Identifying speakers · embeddings`; `overall_fraction` removed.

## Rung 3 (shipped) — stage ETA + terminal bar

- GUI label appends `· ~6 min left` from `EtaEstimator`: the current stage's rate, after `ETA_MIN_SECONDS`; no wire change.
- TTY stdout without `STT_PROGRESS` → `RichProgressBar` replaces the `⌛` log lines; piped output unchanged.

## Rung 4 — whole-job ETA (ruled 2026-09-26)

- `process_all_files` probes durations via `inspect_audio` (only when `reporter.enabled` and >1 file) → `durations` on `start_file`.
- GUI `JobEtaEstimator`: wall s per audio s over finished files × audio left; replaces the stage ETA as `~N min left (all)`; none before file 1 ends, never for 1 file; terminal bar unchanged.

## Rung 5 — retry numbering (ruled 2026-09-26)

- `run_job` renumbers a retry batch's `@@progress` lines onto the original list (`File 4/5`), dropping `durations`; GUI appends ` (retry)` after a `Retrying…` line. No backend change.
