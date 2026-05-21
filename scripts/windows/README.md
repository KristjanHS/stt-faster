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
| `_runtime.bat` | – | – | Shared helper: runtime probe, audio-dir resolution, WSL path translation. Not invoked directly. |

To change which variant a file uses, edit the `set "VARIANTS=..."` line at the top — space-separated for multiple (e.g. `set "VARIANTS=1 36 44"`).

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

Every bat calls `_runtime.bat` first. The helper:

1. Probes WSL: `where wsl` AND `wsl -e test -x <STT_WSL_REPO>/.venv/bin/python`. Both pass → `STT_RUNTIME=wsl`.
2. If still unset, probes Docker: `where docker` AND `docker image inspect stt-faster:latest`. Both pass → `STT_RUNTIME=docker`.
3. If neither probe matches, prints an error pointing at `setup.bat` and exits non-zero.
4. Resolves the audio dir: `STT_AUDIO_DIR` env var → caller's `STT_CALLER_DIR` (set to `%~dp0` by the bat) → `%CD%`. Trailing backslash trimmed.
5. Translates the resolved Windows path to `/mnt/<drive>/...` form for WSL invocations.
6. Prints a banner: `[stt-faster] runtime: <wsl|docker>`, audio dir, and (when WSL) the WSL repo path.

The calling bat then branches on `STT_RUNTIME` and dispatches the same Python invocation to either WSL (`wsl -e bash -c "..."`) or the Docker image (`docker run --rm -v ... stt-faster:latest ...`).

## Model presets

| Preset | Model | Size | Language | Notes |
|--------|-------|------|----------|-------|
| `et-large` (default for Estonian) | TalTechNLP/whisper-large-v3-turbo-et-verbatim | ~1.5 GB | Estonian | Best Estonian accuracy |
| `et-32` | Same model, float32 CPU | ~1.5 GB | Estonian | CPU-only |
| `turbo` | Systran/faster-distil-whisper-large-v3 | ~1.5 GB | Multi | Used by English `.bat`s |
| `large8gb` | Systran/faster-whisper-large-v3 | ~3 GB | Multi | Highest accuracy |
| `distil` | Distilled Whisper | ~756 MB | English | Fastest |

See `docs/Transcription_solution.md` for the full preset / variant matrix.
