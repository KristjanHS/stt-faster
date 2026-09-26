# Installer — real byte progress for model downloads

Status: open. Staged from the 2026-09-26 Windows eyeball run.

## Problem (reproduced locally, `Systran/faster-whisper-small`)

- Bar = `dir_size(hub cache) / expected_model_size`: hf-xet leaves a 0-byte `.incomplete` for 10–28 s, then the
  file size jumps to full while `hf` keeps writing ~33 s more — bar sits at 0, then full while still downloading.
- Status text = last `Fetching N files` line; it only ticks per finished file, so it reads `2/5 [00:00<…` for minutes.

## Fix

- Add `TQDM_POSITION=-1` to the `hf download` env: hf 1.2.1 then prints its aggregate byte bar
  (`Downloading (incomplete total...): 17%| 83.9M/484M [00:40<01:51, 3.60MB/s]`) through the pipe.
- `run_process`: strip ANSI (`\x1b[A`), take fraction from the `Downloading` line (`done/total` with k/M/G units;
  total grows while metadata arrives), fall back to `poll()`; show the latest `Downloading` line as status.
- Falsifier: unit test feeding captured bar lines → fractions; probe script re-run shows fraction rising mid-download.

## Open

- Updates are bursty with xet (one run: 17 % at 40 s → done at 47 s) — accept, or add a "downloading…" spinner text
  between updates.
