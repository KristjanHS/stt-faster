# Windows Batch Files for Transcription

One-click audio transcription for Windows users. Each `.bat` runs `scripts/transcribe_manager.py` with a preset, language, and one or more variant numbers baked into the file.

The bats auto-detect WSL: if WSL2 is installed and the repo lives at `/home/kristjans/projects/stt-faster` inside WSL, they run there (preserving the current dev-machine workflow). Otherwise they fall back to a Docker container built from the production `Dockerfile` (CPU-only). The runtime banner at the top of each run prints the choice.

## First-time setup (any Windows machine without WSL)

1. Install Docker Desktop from https://docs.docker.com/desktop/install/windows-install/ and start it.
2. `git clone <repo> C:\projects\stt-faster`
3. `cd C:\projects\stt-faster`
4. `setup.bat` — builds the `stt-faster:latest` image (~5-15 min on first run). Models download lazily on the first transcription run (~3 GB into `%USERPROFILE%\.cache\hf`).

After setup, every bat in `scripts\windows\` works without further configuration. WSL users can skip steps 1 and 4; the bats detect WSL automatically.

## Files

| File | Language | Default variant(s) | Tuned for |
|------|----------|--------------------|-----------|
| `transcribe_estonian_Desk.bat` | Estonian | 44 | Desk-mic / close-talk Estonian |
| `transcribe_estonian_Teams.bat` | Estonian | 61 | Teams / video-call Estonian |
| `transcribe_estonian_Online.bat` | Estonian | 52 | Online / mixed-source Estonian |
| `transcribe_estonian_32bit_cpu.bat` | Estonian | 7 (preset `et-32`) | CPU-only machines, no GPU |
| `transcribe_english_Desk.bat` | English | 44 | Desk-mic English (`turbo` preset) |
| `transcribe_english_Teams.bat` | English | 61 | Teams / video-call English (`turbo` preset) |
| `transcribe_english_Online.bat` | English | 52 | Online / mixed-source English (`turbo` preset) |
| `compare_variants.bat` | Estonian | all builtin variants | Comparison run across every variant in the registry; also generates the HTML report |
| `report_only.bat` | – | – | Regenerates the variant HTML report from existing outputs |
| `check_recent.bat` | – | – | Shows recent transcription run history (`stt-faster db recent`) |
| `_*.bat` | – | – | Shared helpers — see [Helpers](#helpers) below. Not invoked directly. |

To change which variant a file uses, edit the `set "VARIANTS=..."` line at the top — space-separated for multiple (e.g. `set "VARIANTS=1 36 44"`).

## Helpers

The transcribe bats delegate the boilerplate to six helper files. None of them are meant to be double-clicked — they're called from the top-level bats. Each helper documents its caller contract + outputs + scratch-var leaks in a header comment.

| Helper | Purpose |
|--------|---------|
| `_runtime.bat` | Probes WSL / Docker, resolves audio dir, translates Windows path → `/mnt/...`, composes `STT_DIARIZE_ARGS`, prints the runtime banner. Owns the docker-image auto-build prompt (opt out via `STT_NO_AUTOBUILD=1`). |
| `_variants.bat` | Normalizes `VARIANTS` (space-separated) → `VARIANTS_COMMA` + `VARIANT_COUNT`. |
| `_banner.bat` | Prints the title / model / language / variant(s) header. Honors `STT_TITLE_SUFFIX` (e.g. `[DOCKER FORCED]`). |
| `_transcribe.bat` | Dispatches WSL vs Docker. WSL branch invokes `wsl -e bash -c "..."`; docker branch delegates to `_docker_run.bat`. |
| `_docker_run.bat` | Composes `DOCKER_ENV_ARGS` from `HF_TOKEN` / `HF_XET_HIGH_PERFORMANCE`, runs the diarize fail-fast HF_TOKEN guard, and executes `docker run`. |
| `_footer.bat` | Prints the "Processing Complete!" trailer + `pause`. |

## Adding a new transcribe variant

A new transcribe bat is ~19 lines. Copy any existing one (e.g. `transcribe_english_Online.bat`) and edit only the knobs at the top + four `STT_*` lines:

```bat
@echo off
set "VARIANTS=52"
set "DIARIZE="          REM "" = runtime default (wsl=on, docker=off); "0" = off; "1" = on
set "NUM_SPEAKERS=2"    REM only used when diarize resolves to on; "" = auto-detect

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
set "STT_TITLE=Audio Transcription - <LANG> (<source>)"
set "STT_MODEL=<model display string>"
set "STT_LANG=<English | Estonian | ...>"
set "STT_CLI_TAIL=--preset <preset> --language <code> --output-format txt"
call "%~dp0_runtime.bat" || ( pause & exit /b 1 )
call "%~dp0_variants.bat"
call "%~dp0_banner.bat"
call "%~dp0_transcribe.bat"
call "%~dp0_footer.bat"
```

For docker-forced variants, also `set "STT_FORCE_DOCKER=1"` and `set "STT_TITLE_SUFFIX=[DOCKER FORCED]"` before the `_runtime.bat` call. See `transcribe_english_Online_docker.bat` for the canonical example.

## Usage

Two ways to point a bat at an audio folder:

**(a) Copy-into-folder (legacy, no env var):** copy the bat into the folder that holds your audio (e.g. `C:\Users\PC\Downloads\transcribe\`) and double-click it. The bat treats its own folder as the audio dir.

**(b) Env var + run-in-place:**
```bat
set STT_AUDIO_DIR=C:\Users\PC\Downloads\transcribe
```
then double-click any bat directly under `scripts\windows\`. The bats honor `STT_AUDIO_DIR` over their own folder when both are present.

Supported audio formats: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`.

Results land next to the audio file:
- Single-variant runs write `processed/<filename>.json` and `processed/<filename>.txt`.
- Multi-variant runs create a timestamped folder (e.g. `2026-05-19T14-30-00/`) next to the audio, containing one `variant_NNN_<name>_<audio>.json` (and `.txt`) per variant plus a `run_meta.json` describing the run.
- Failed inputs land in `failed/`.

## Requirements

- Windows 10/11.
- Either: WSL2 with the repo at `/home/kristjans/projects/stt-faster` and a `.venv` (current dev-machine setup), **or** Docker Desktop with the `stt-faster:latest` image built by `setup.bat`.
- Optional: CUDA + cuDNN on the WSL side for GPU acceleration. The Docker fallback is CPU-only by design.

## Troubleshooting

**Banner says `runtime: docker` but you expected `wsl`**
→ The WSL probe failed. Check `where wsl` returns a path and `wsl -e test -x /home/kristjans/projects/stt-faster/.venv/bin/python` exits 0. Edit `STT_WSL_REPO=` in `_runtime.bat` if the WSL repo lives elsewhere.

**ERROR: no runtime available**
→ Neither WSL nor Docker is usable. Install Docker Desktop, start it, and run `setup.bat` from the repo root to build the image.

**Processing is slow**
→ First run downloads the model. The Docker fallback is CPU-only; for GPU acceleration use WSL (the bats prefer it automatically when available).

**Files stuck in pending**
→ Look in `<audio_folder>/failed/` for files that errored; check `logs/` for stack traces.
→ Inspect recent run history: `.venv/bin/python -m backend.cli.main db recent --limit 10` (WSL) or `docker run --rm -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" --entrypoint python stt-faster:latest -m backend.cli.main db recent --limit 10` (Docker).

## How the runtime dispatch works

Every transcribe bat calls `_runtime.bat` first. The helper:

1. Probes WSL: `where wsl` AND `wsl -e test -x <STT_WSL_REPO>/.venv/bin/python`. Both pass → `STT_RUNTIME=wsl`. Skipped when the caller sets `STT_FORCE_DOCKER=1`.
2. If still unset, probes Docker: `where docker` AND `docker image inspect stt-faster:latest`. Both pass → `STT_RUNTIME=docker`.
3. If docker is on PATH but the image is missing AND `STT_NO_AUTOBUILD` is not `1`, prompts to build it (`docker build -t stt-faster:latest .` from the repo root). On Y + successful build → `STT_RUNTIME=docker`. On N / build-failed / repo-not-detected, exits non-zero.
4. If both probes (and auto-build) failed, prints `ERROR: no runtime available` and exits non-zero.
5. Resolves the audio dir: `STT_AUDIO_DIR` env var → caller's `STT_CALLER_DIR` (set to `%~dp0` by the bat) → `%CD%`. Trailing backslash trimmed.
6. Translates the resolved Windows path to `/mnt/<drive>/...` form for WSL invocations.
7. Composes `STT_DIARIZE_ARGS` from caller `DIARIZE` / `NUM_SPEAKERS` knobs + runtime default (WSL=on, Docker=off).
8. Prints a banner: `[stt-faster] runtime: <wsl|docker>`, audio dir, WSL repo path (when WSL), diarize state.

After `_runtime.bat` returns, the bat calls `_variants.bat`, `_banner.bat`, `_transcribe.bat`, and `_footer.bat` in sequence. `_transcribe.bat` is the one that branches on `STT_RUNTIME` — it dispatches the same Python invocation to either WSL (`wsl -e bash -c "..."`) or the Docker image (via `_docker_run.bat`).

## Model presets

| Preset | Model | Size | Language | Notes |
|--------|-------|------|----------|-------|
| `et-large` (default for Estonian) | TalTechNLP/whisper-large-v3-turbo-et-verbatim | ~1.5 GB | Estonian | Best Estonian accuracy |
| `et-32` | Same model, float32 CPU | ~1.5 GB | Estonian | CPU-only |
| `turbo` | Systran/faster-distil-whisper-large-v3 | ~1.5 GB | Multi | Used by English `.bat`s |
| `large8gb` | Systran/faster-whisper-large-v3 | ~3 GB | Multi | Highest accuracy |
| `distil` | Distilled Whisper | ~756 MB | English | Fastest |

See `docs/Transcription_solution.md` for the full preset / variant matrix.
