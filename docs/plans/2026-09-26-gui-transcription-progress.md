# GUI transcription progress — structured channel

**Status:** rung 1 shipped; backlog below is unordered, re-pick after each rung.

## Decision (2026-09-26)

Chose option C over (A) regex-parsing the `⌛ Transcription progress` log line and (B) faster-whisper's
`log_progress=True` tqdm bar: a dedicated `@@progress {json}` line on stdout, emitted by
`backend/progress.py` only when `STT_PROGRESS=1` (set by `gui.build_env`). Human logs are untouched;
the GUI reads only prefixed lines. A/B were rejected because they couple the GUI to a human/third-party
text format and cover only the transcribe stage.

Wire format: `@@progress {"file": 2, "files": 3, "stage": "transcribe", "done": 312.4, "total": 1800.0}`
— `done`/`total` omitted on a bare stage change. Stage changes are never throttled; `advance` is
throttled to 1/s (`MIN_INTERVAL_SECONDS`). Overall fraction = `(file - 1 + done/total) / files`.

## Rung 1 (shipped)

- `ProgressReporter` + `parse_progress` in `backend/progress.py`.
- `TranscriptionProcessor.process_all_files` → `start_file(i, n)` (stage `prepare`).
- Both segment loops (`transcribe._collect_segments`, `executor._collect_executor_segments`) →
  `advance("transcribe", seconds_done, total)`.
- GUI: bar goes determinate on the first event; new detail label `38% · File 2/4 · Transcribing`;
  raw last-log line stays in the grey status label.
- Review fixes: forced end-of-file `advance` (the throttle dropped the last point); a `Retrying…` line
  resets the bar to indeterminate; `_finish` fills the bar only when `JobResult.ok`.

## Backlog

- **Diarization stage** — feed `_DiarizeProgressHook` (`backend/diarize/pyannote_runner.py`) into the
  reporter: `stage("diarize")` on entry, `advance("diarize", completed, total)` per pyannote step. Needs a
  stage-weight split per file (e.g. transcribe 0.7 / diarize 0.3 when diarizing), else the bar reaches
  100 % of a file before diarization starts.
- **Preprocess / model-load stage names** — `stage("preprocess")`, `stage("load model")` so the long
  pre-segment wait is named instead of a generic "Preparing".
- **ETA** — the GUI measures the rate of `done` against wall time per file → `~6 min left` for the current
  file; the whole-job ETA needs the durations of later files (emit `durations` in `start_file`?).
- **Retry file numbering** — a retry resets the bar to indeterminate with a `Retrying…` label, then counts
  `File 1/k` over the retried subset only; mapping back onto the original N would need the GUI to pass offsets.
- **CLI terminal** — optional Rich progress bar rendered from the same reporter when stdout is a TTY.
