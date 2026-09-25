# Windows one-click installer + simple GUI — design

**Status:** approved design, not started. Execute slice by slice (`/qimpag` per slice).

## Goals

1. A non-IT person on a **fresh Windows PC** (no admin, no WSL, no Docker, no git, no Python) installs by downloading **one small `.exe`** and clicking through.
2. They transcribe through a **simple native window** modelled on the owner's reference screenshot (drop zone, language toggle, timestamps checkbox).
3. **Nothing changes for the owner's workflow:** the CLI, `scripts/windows/*.bat`, `run_uv.sh`, Docker and WSL paths keep working as they do now.
4. Keep build time and token spend minimal: **reuse the CLI instead of re-implementing it**, spend no GitHub Actions minutes, and rebuild the installer `.exe` only rarely.

## Decisions (user-ruled 2026-09-25)

| Topic | Ruling |
|---|---|
| UI toolkit | **tkinter** (native, offline, no browser/server). Drag-drop via `tkinterdnd2`. No HTML exists to reuse. |
| Install channel | **Tiny bootstrap installer `.exe`** (PyInstaller, stdlib-only script, ~10 MB). It downloads the dependencies and the models **in parallel during install**, showing **one progress bar per download**. No fat frozen app, and no model is bundled. |
| Settings exposed | **Language only.** No Desk/Teams/Online choice: settings always follow the Teams bats (see *GUI profiles*). |
| Diarization | **Off in the lean install.** Enabled later from a collapsed **"Extras"** section: HF token + how-to-get-a-token hint → on-demand install of torch/pyannote. |
| Build / release | **Build locally, release via GitHub** (`gh release`), tag-driven, **zero Actions minutes**. |
| Delivery | Public repo `KristjanHS/stt-faster` → stable link `releases/latest/download/Transcribe-Setup.exe`. |

## Architecture

```
Transcribe-Setup.exe  (stdlib tkinter, PyInstaller --onefile --windowed; rebuilt rarely;
                       copies itself into <install> so the GUI can re-invoke it later)
  1. install dir = %LOCALAPPDATA%\stt-faster   (no admin)
  2. fetch uv.exe (pinned astral-sh/uv release zip)
  3. fetch app source = latest GitHub Release source zip  ← app updates need NO installer rebuild
  4. in parallel, one bar each:
       [deps  ]  uv sync --frozen --no-dev --extra gui           (uv also fetches Python 3.12)
       [models]  uvx --from huggingface_hub hf download
                   TalTechNLP/whisper-large-v3-turbo-et-verbatim --include "ct2/*"   (Estonian)
                   Systran/faster-distil-whisper-large-v3                              (English)
                 → standard HF cache; the app's existing snapshot_download calls find them
       [ffmpeg]  static ffmpeg zip → <install>\ffmpeg\bin   (need ffprobe.exe; see Risks)
  5. Desktop + Start-menu shortcut → .venv\Scripts\stt-faster-gui.exe (gui-script: no console)
  Mode `Transcribe-Setup.exe --extras`: close-wait → uv sync --frozen --no-dev --extra gui --extra cpu
       (one bar) → relaunch the GUI.

stt-faster-gui  (backend/gui.py, tkinter)
  selected files ─copy─► <install>\work\<run-id>\          (the user's originals are never moved)
      └► subprocess: stt-faster transcribe process <work dir> --preset P --variant 61
                     --language L --no-diarize [--no-timestamps]
         env: STT_DEVICE=cpu, PATH += <install>\ffmpeg\bin
      ◄── log lines → progress text/bar
  result .txt copied next to each original audio file; [Open result] opens that folder
```

**Why the GUI calls the CLI as a subprocess instead of in-process code:** one code path shared with the bats, crash isolation, Cancel = kill the process, and the GUI stays thin.
**Why the GUI copies files into a work dir:** the CLI accepts only a folder (`transcription_commands.py:464-471`), and after a successful run it *moves* the source into `processed/` (`components.py:414,428-430`). Copying into a work dir means the CLI needs no file-input change and the user's file stays where it is.

### GUI profiles (no variant choice for the user)
Per `scripts/windows/README.md` (the owner confirms the bats' `VARIANTS=` still match):

| Toggle | Mirrors | Preset | Variant | Language |
|---|---|---|---|---|
| **Estonian** | `transcribe_estonian_Teams.bat` | `et-large` | 61 | `et` |
| **English** | `transcribe_english_Teams.bat` | `turbo` | 61 | `en` |

Keep both profiles in one `GUI_PROFILES` constant. Label the second toggle **English**, not "English/Other": the `turbo` preset is `Systran/faster-distil-whisper-large-v3` (`model_config.py:47-48`), which as far as we know handles English only.

### Progress reporting (kept simple)
- **models / ffmpeg:** poll the target directory size against the expected total (HF API file sizes or HTTP `Content-Length`), shown as "1.4 / 3.1 GB".
- **deps:** uv prints no percentage, so show an indeterminate bar plus the last uv output line; the bar switches to ✓ when uv exits with code 0.

### GUI layout
- Drop zone + **Browse Files** (the extensions in `SUPPORTED_AUDIO_EXTENSIONS`, `components.py:27`, plus mp4/mkv; see change #2)
- Language toggle: **Estonian** | **English**
- ☑ Include timestamps (on by default = today's TXT output)
- ▸ Extras (collapsed): HF token + 3-step hint (create an account → accept the `pyannote/speaker-diarization-community-1` licence → create a read token at hf.co/settings/tokens). **Save & install** stores the token in `%APPDATA%\stt-faster\` and runs `Transcribe-Setup.exe --extras`, then quits. The GUI can't run `uv sync` itself because Windows locks files in the running venv. After the relaunch, ☐ **Identify speakers** (+ number of speakers) appears, and the GUI passes `--diarize --num-speakers N` with `HF_TOKEN` set in the subprocess environment.
- START + progress + **Open result**

**Advanced VAD sliders from the screenshot → backlog, not v1.** Variant 61 runs with VAD **off**: it sets only beam, patience and repetition, so it takes the minimal path, `vad_filter` is never passed, and faster-whisper defaults it to off. Sliders would either do nothing or quietly change 61's behaviour. That also contradicts "the user picks only the language".

## Code changes (whole scope)

| # | Change | Files |
|---|---|---|
| 1 | Move `pyannote.audio` from base `dependencies` into **both** `cpu` and `cu130` extras. uv.lock confirms only pyannote and the packages under it pull torch. Every owner install path already passes an extra (`run_uv.sh:26`, `Makefile:140,212,217,231,337,358`, `Dockerfile:35,41`, `docker/app.Dockerfile:35,44`, `export-reqs`), so nothing there changes. Add a `gui` extra = `tkinterdnd2`. Add gui-script `stt-faster-gui = "backend.gui:main"`. Rewrite the policy comment at `pyproject.toml:57-60` ("bare sync = undefined behaviour"): a bare sync + `gui` extra is now the supported lean Windows install. | `pyproject.toml`, `uv.lock` |
| 2 | CLI `transcribe process`: add `--timestamps/--no-timestamps` (default on). It's a new output option: today the TXT always has timestamps (`transcribe.py:417-428`); don't confuse it with the unrelated `word_timestamps`. Add `.mp4`/`.mkv` to `SUPPORTED_AUDIO_EXTENSIONS` (`components.py:27`). Owner can use both too. | `backend/cli/transcription_commands.py`, `backend/transcribe.py`, `backend/components.py` |
| 3 | GUI | `backend/gui.py` (new) |
| 4 | Bootstrap installer (stdlib only, incl. `--extras` mode) | `installer/setup_gui.py` (new) |
| 5 | Build the installer: manual, on the Windows host, rare: `uvx pyinstaller --onefile --windowed installer/setup_gui.py` | `installer/build_installer.bat` (new) |
| 6 | Release: `make release V=x.y.z` → tag + push + `gh release create --generate-notes` + re-attach `Transcribe-Setup.exe` (downloaded from the previous release with `gh release download`, or from `dist/` when rebuilt) | `Makefile` |

The bats, `run_uv.sh`, the Dockerfile and the CLI's default behaviour are untouched.

## Slices (thinnest visible first; re-pick after each)

1. **GUI + CLI flag**: `backend/gui.py` + change #2 + gui-script. Visible right away: `uv run stt-faster-gui` opens under WSLg.
2. **Lean deps**: change #1. Verify: a fresh `uv sync --no-dev --extra gui` (no cpu extra) → `stt-faster transcribe process <dir> --variant 61 --no-diarize` works and `import torch` fails. `backend/__init__.py:5-7` → `preload_bundled_cudnn()` must swallow the ImportError silently. `./run_uv.sh` still installs pyannote.
3. **Installer**: `installer/setup_gui.py`. The logic is testable on Linux (paths are parameterized); the real run is on the Windows host.
4. **Build + release**: `build_installer.bat` + `make release`. First release attaches the `.exe`.
5. **Extras / diarization** panel + `--extras` installer mode.

Backlog (unordered): VAD sliders (needs a `--vad` override that sets `vad_filter=True` via `config.set(...)` on the variant's config before `ServiceFactory.create_transcription_service`, `transcription_commands.py:250`; `vad_threshold` / `vad_parameters.min_silence_duration_ms` live in `preprocess/config.py:262-276`) · in-app update check · multilingual "Other" profile.

## Risks / verify-first

- **Native Windows has never been run.** ctranslate2, av and onnxruntime ship Windows wheels; confirm on first install. The Windows-specific `torchcodec` problem is already sidestepped (`pyannote_runner.py` feeds waveforms directly).
- **External binary:** variant 61 has no preprocess steps (`registry.py:252`), but `inspect_audio` always calls **`ffprobe`** (`preprocess/io.py:30,44`). The installer's ffmpeg zip must include `ffprobe.exe`, and the GUI prepends its `bin` dir to `PATH`.
- **CPU fallback banner:** both presets say `device="cuda"`. The GUI sets `STT_DEVICE=cpu` to skip the probe and the warning banner.
- **CLI details:** the command is `stt-faster transcribe process` (`cli/main.py:19`). Use `--variant`, never `-v`, which is bound to both `--variant` and `--verbose` (`transcription_commands.py:501,503`). The CLI default is `--diarize` (`:504-506`), so the GUI must pass `--no-diarize` unless Extras is set up.
- **SmartScreen** warns on an unsigned `.exe` ("More info → Run anyway"). Put this in the release notes; code signing is out of scope.

## Out of scope

Code signing · GPU on the target PC · a fat frozen app · uninstaller beyond "delete folder + shortcut" · Mac/Linux installers.
