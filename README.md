# stt-faster

Offline speech-to-text for Estonian and English recordings, with speaker labels.

[![CI](https://github.com/KristjanHS/stt-faster/actions/workflows/python-lint-test.yml/badge.svg)](https://github.com/KristjanHS/stt-faster/actions/workflows/python-lint-test.yml) [![Latest release](https://img.shields.io/github/v/release/KristjanHS/stt-faster?sort=semver)](https://github.com/KristjanHS/stt-faster/releases/latest) [![License: MIT](https://img.shields.io/github/license/KristjanHS/stt-faster)](LICENSE)

Predecessor of [transcribe-offline](https://github.com/KristjanHS/transcribe-offline), which is where development continues.

## Quick start

Windows: download [Transcribe-Setup.exe](https://github.com/KristjanHS/stt-faster/releases/latest/download/Transcribe-Setup.exe) and run it. The installer is unsigned, so Windows warns twice: click Keep, then More info → Run anyway.

From source on Linux or WSL2, with [uv](https://astral.sh/uv) installed:
```bash
./run_uv.sh
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio
```

## What it does

- Transcribes every recording in a folder to a timestamped `.txt` file, moves done files to `processed/` and failures to `failed/`.
- Model presets: `et-large` (Estonian, default), `large8gb` (English and other languages), `en-distil` (fastest, English only), `ja-en-distil` (Japanese with English).
- Labels speakers by default; `--no-diarize` turns that off. Setup: [docs/diarization_setup.md](docs/diarization_setup.md).
- Only the installer goes online. Transcription makes no network calls and there is no telemetry ([NOTICE](NOTICE) lists the models).
- Also runs in Docker, no local Python needed: [docker/README.md](docker/README.md). How the pipeline works: [docs/Transcription_solution.md](docs/Transcription_solution.md).

## Tests

`make unit` runs the 353 unit tests in `tests/unit` (static count of `def test_`). CI (`python-lint-test.yml`) gates on ruff, pyright and the same unit tests. Built with the agent workflow described in [claudeconf](https://github.com/KristjanHS/claudeconf).

## Limits

- Unsigned installer. Smart App Control may block reopening the downloaded file, so update or remove through Start → Transcribe → Repair / Uninstall.
- No releases after v1.2.7 (2026-11-15); new work goes to transcribe-offline.
- Source builds: Linux and WSL2 only.
- The Docker image has no models in it. Mount the speaker and RNNoise models read-only.

## License

MIT, see [LICENSE](LICENSE).
