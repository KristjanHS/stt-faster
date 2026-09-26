# GUI transcription progress — structured channel

**Status:** rung 2 in flight; backlog below is unordered, re-pick after each rung.

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

## Rung 2 — diarize + stage names

- `transcribe()` emits `preprocess` → `load model` → `transcribe` → `diarize` stages via a defaulted `reporter`.
- `_DiarizeProgressHook` takes an `on_progress(step, completed, total)` callback (diarize stays free of
  `backend.progress`); `transcribe()` adapts it to `stage/advance("diarize", …, detail=step)`.
- GUI: bar = `stage_fraction`, label `File 2/3 · Identifying speakers · embeddings`; `overall_fraction` removed.

## Backlog

- **ETA** — the GUI measures the rate of `done` against wall time per file → `~6 min left` for the current
  file; the whole-job ETA needs the durations of later files (emit `durations` in `start_file`?).
- **Retry file numbering** — a retry resets the bar to indeterminate with a `Retrying…` label, then counts
  `File 1/k` over the retried subset only; mapping back onto the original N would need the GUI to pass offsets.
- **CLI terminal** — optional Rich progress bar rendered from the same reporter when stdout is a TTY.
