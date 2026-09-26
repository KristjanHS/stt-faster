# Windows one-click installer + simple GUI — design

**Status:** in progress — slice 1 shipped (`e59ac04`, `0653df2`); slice 2 shipped (`0895ab9`, `4799a25`); slice 3 shipped (`bd70690`, `830674e`); slice 4 shipped (`9aef2a4`, `7a4884c`, bat placed by the owner); slice 4b shipped (`c5eab3a` + review fixes + e2e guard); monkeypatch-free tests shipped (`7a9461f`..`da82b47`); slice 5 shipped (`d7d30c2`, `6407156`, `f61bfe3`); slice 6 shipped (`e7fae5f` + review fixes; first real GPU run owed on the owner host); slice 7 (signing) dropped 2026-09-26 — replaced by one-launch-only exe (setup runs from the venv launcher afterwards). Execute slice by slice (`/qimpag` per slice).

## Goals

1. A non-IT person on a **fresh Windows PC** (no admin, no WSL, no Docker, no git, no Python) installs by downloading **one small `.exe`** and clicking through.
2. They transcribe through a **simple native window** modelled on the owner's reference screenshot (drop zone, language toggle, timestamps checkbox).
3. **Nothing changes for the owner's workflow:** the CLI, `scripts/windows/*.bat`, `run_uv.sh`, Docker and WSL paths keep working as they do now.
4. Keep build time and token spend minimal: **reuse the CLI instead of re-implementing it**, spend no paid GitHub Actions minutes, and rebuild the installer `.exe` only rarely.

## Decisions (user-ruled 2026-09-25)

| Topic | Ruling |
|---|---|
| UI toolkit | **tkinter** (native, offline, no browser/server). Drag-drop via `tkinterdnd2`. No HTML exists to reuse. |
| Install channel | **Tiny bootstrap installer `.exe`** (PyInstaller, stdlib-only script, ~10 MB). It downloads the dependencies and the models **in parallel during install**, showing **one progress bar per download**. No fat frozen app, and no model is bundled. |
| Settings exposed | **Language only.** No Desk/Teams/Online choice: settings always follow the Teams bats (see *GUI profiles*). |
| No admin rights | **Hard requirement: nothing may trigger a UAC prompt.** Install to `%LOCALAPPDATA%\stt-faster`. Put shortcuts in the per-user Desktop and Start menu (`%APPDATA%\Microsoft\Windows\Start Menu`). Keep settings and token in `%APPDATA%\stt-faster`. uv, Python, the models, ffmpeg and the NVIDIA DLL wheels are all user-space downloads. No drivers, services, registry HKLM or system PATH changes. GPU mode uses the NVIDIA driver only if one is already installed. The `.exe` is built with an `asInvoker` manifest (PyInstaller's default; keep `--uac-admin` off). This matters because Windows' installer-detection heuristic auto-elevates un-manifested exes named `*Setup*`/`*Install*`. Verify with `sigcheck -m` or by running as a standard user. |
| CPU / GPU | **Auto-detect, no question.** The installer runs `nvidia-smi` (it ships with every NVIDIA driver). A capable NVIDIA GPU → GPU mode; anything else, or any doubt → **CPU (default)**. The result shows on screen with a one-click **"Use CPU instead"** override. See *GPU mode*. |
| Diarization | **Off in the lean install.** Enabled later from a collapsed **"Extras"** section: HF token + how-to-get-a-token hint → on-demand install of torch/pyannote. |
| Build / release | **Release via GitHub** (`gh release`), tag-driven; `.github/workflows/release-installer.yml` on release publish rebuilds the exe on `windows-latest` only if `installer/` changed since the previous release (else re-attaches that exe, keeping its hash and SmartScreen reputation) and attaches it, unsigned. |
| SmartScreen / Smart App Control (ruled 2026-09-26) | No signing. README click-through text; the downloaded exe runs once, then Repair / Uninstall / Apps & features / Extras run `.venv\Scripts\stt-faster-setup.exe` (SAC blocks later exe launches, not the venv launchers). Store/MSIX = backlog. |
| Delivery | Public repo `KristjanHS/stt-faster` → stable link `releases/latest/download/Transcribe-Setup.exe`. |

## Architecture

```
Transcribe-Setup.exe  (stdlib tkinter, PyInstaller --onefile --windowed; rebuilt rarely;
                       copies itself into <install> so the GUI can re-invoke it later)
  1. install dir = %LOCALAPPDATA%\stt-faster   (no admin)
  2. fetch uv.exe (pinned astral-sh/uv release zip)
  3. fetch app source = latest GitHub Release source zip  ← app updates need NO installer rebuild
  4. in parallel, one bar each:
       [deps  ]  uv sync --frozen --no-dev --extra gui [--extra gpu-win]   (uv also fetches Python 3.12)
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
         env: STT_DEVICE=cpu|cuda (from %APPDATA%\stt-faster\config), PATH += <install>\ffmpeg\bin
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

### GPU mode (NVIDIA only)
**Transcription on GPU doesn't need torch.** faster-whisper runs on ctranslate2, which only needs the CUDA **cuBLAS + cuDNN DLLs**. On Linux the `cu130` extra supplies them through torch's bundled `nvidia/*` wheels plus `_cudnn_preload.py`. On Windows, skip torch: a new `gpu-win` extra installs the NVIDIA DLL wheels directly (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, marker `sys_platform == 'win32'`, about 1 GB). That is smaller than a CUDA torch (about 2.5 GB) and doesn't touch the existing `cpu`/`cu130` conflict set.

Detection in the installer:
```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  missing / error / VRAM < 4 GB / driver < CUDA-12 minimum   → CPU   (default)
  otherwise                                                  → GPU   "Found NVIDIA RTX 3060 (12 GB) – using GPU  [Use CPU instead]"
```
The choice is saved as `device=cpu|cuda` in `%APPDATA%\stt-faster\config`. The GUI passes it as `STT_DEVICE`. If GPU inference fails at runtime, the GUI re-runs the same job once with `STT_DEVICE=cpu` and writes `device=cpu`. The user never sees a CUDA error, and a wrong detection fixes itself.

Diarization stays on CPU torch (the `cpu` extra) in both modes. GPU diarization on Windows is out of scope. *Superseded 2026-09-26:* GPU mode on a driver ≥ 580 (CUDA 13) syncs `cu130` instead of `cpu`, keeping `gpu-win` for ctranslate2's CUDA 12 cuBLAS; older drivers keep `cpu` and setup says so. `run_pyannote` retries on the CPU if the GPU run throws (e.g. a cuDNN clash with ctranslate2's copy in the same process) and honours `STT_DEVICE=cpu`.

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
| 2b | `gpu-win` extra (see *GPU mode*). Add a Windows branch to `preload_bundled_cudnn()`: `os.add_dll_directory()` for each `site-packages/nvidia/*/bin` before ctranslate2 loads. It's a no-op when those dirs are absent, and the Linux `.so` path is unchanged. | `pyproject.toml`, `uv.lock`, `backend/_cudnn_preload.py` |
| 3 | GUI | `backend/gui.py` (new) |
| 4 | Bootstrap installer (stdlib only, incl. `--extras` mode) | `installer/setup_gui.py` (new) |
| 5 | Build the installer: `uvx pyinstaller --onefile --windowed installer/setup_gui.py` (locally for testing; releases build in CI) | `installer/build_installer.bat` (new) |
| 6 | Release: `make release V=x.y.z` → tag + push + `gh release create --generate-notes`; the `release-installer.yml` job then attaches `Transcribe-Setup.exe` (rebuilt only if `installer/` changed) | `Makefile` |

The bats, `run_uv.sh`, the Dockerfile and the CLI's default behaviour are untouched.

## Slices (thinnest visible first; re-pick after each)

1. **GUI + CLI flag**: `backend/gui.py` + change #2 + gui-script + `gui` extra (tkinterdnd2; drop zone falls back to click-to-browse). Visible right away: `uv run stt-faster-gui` opens under WSLg.
   Linux/WSLg: drag-drop off (tkdnd's system libX11 vs uv-Python libtk's embedded Xlib → xcb abort); Windows unaffected.
   Rulings: result `.txt` never overwrites (`name (2).txt`); `--no-timestamps` = one line per segment; no config → `STT_DEVICE` unset; a non-CPU job that fails re-runs once with `cpu` and writes `device=cpu` (pulled forward from slice 6).
2. **Lean deps**: change #1. Verify: a fresh `uv sync --no-dev --extra gui` (no cpu extra) → `stt-faster transcribe process <dir> --variant 61 --no-diarize` works and `import torch` fails. `backend/__init__.py:5-7` → `preload_bundled_cudnn()` must swallow the ImportError silently. `./run_uv.sh` still installs pyannote.
   Rulings: `--diarize` without pyannote installed → warn + continue without speakers (CLI-level `find_spec` gate); guard = `tests/unit/test_lean_deps.py` (uv.lock base closure has no torch) + one scratch-venv real run.
3. **Installer**: `installer/setup_gui.py`. The logic is testable on Linux (paths are parameterized); the real run is on the Windows host.
   Rulings: source = latest release (none yet → clear error) + `--source url|zip|dir` override; existing install → user picks Repair (default) / Clean reinstall; uv `0.12.19` + GyanD ffmpeg `9.0.2` essentials pinned; writes `device=cpu` if unset; done = unit tests + a headless Linux e2e into a scratch dir.
4. **Build + release**: `build_installer.bat` + `make release`. First release attaches the `.exe`.
   Rulings: `make release V=X.Y.Z` → `scripts/release.sh`: preflight (main, clean, tag absent local+origin) before any change → bump `pyproject` version + `uv lock` + commit → tag → push main + tag → `gh release create` (no exe; CI attaches it, see `archived/2026-09-25-smartscreen-code-signing.md` §D); first release v1.1.0; bat self-fetches pinned uv + PyInstaller into `%TEMP%`; guard = pytest with fake git/gh/uv on `PATH`.
4b. **Self-contained install + uninstall** (before 5 — Extras installs into the same caches). Everything lives under `%LOCALAPPDATA%\stt-faster\`; nothing shared is touched.
   Moves: `deps_env` + `model_command` env set `UV_CACHE_DIR=<install>\uv-cache`, `UV_PYTHON_INSTALL_DIR=<install>\python`, `UV_TOOL_DIR=<install>\uv-tools`, `TEMP`/`TMP`/`TMPDIR=<install>\tmp`, `UV_NO_CONFIG=1`, `HF_HOME=<install>\hf` (+ `HF_HUB_CACHE`/`HF_XET_CACHE` pinned under it so inherited values never win); `gui.build_env` sets the same HF vars. Repair copies (never moves) matching model dirs from the user's resolved old hub cache first.
   Uninstall: third mode beside Repair/Clean + `--uninstall`; removes install dir, `%APPDATA%\stt-faster`, both shortcuts, and the `HKCU\…\CurrentVersion\Uninstall\stt-faster` key (`winreg`, no admin) that install writes for Apps & features (`UninstallString = <install>\.venv\Scripts\stt-faster-setup.exe --uninstall`). A setup run from the install dir hops to a `%TEMP%\stt-faster-setup-*` copy (script + its Python runtime) first; that copy spawns a hidden `cmd` that deletes it after exit.
   Guard: `pytest tests/e2e/test_installer_e2e.py` (scratch `HOME` + `TMPDIR`, host models hardlinked into `~/.cache/huggingface`; network, ~30 s).
5. **Extras / diarization** panel + `--extras` installer mode.
   Rulings: setup writes `extras=diarization` to config after a successful `--extras`; `deps_command` adds `--extra cpu` whenever it is set (Repair keeps it); `--extras` pre-fetches `pyannote/speaker-diarization-community-1` with the token (2nd bar), 401/403 → error in setup, no relaunch; token file `%APPDATA%\stt-faster\hf_token`; ☐ Identify speakers shows iff pyannote imports + token saved, spinbox 2–10 default 2 (CLI minimum); diarization failure → re-run once `--no-diarize` + "Speakers skipped: <reason>" banner; no `stt-faster-setup.exe` in the venv → button disabled + hint; guard = unit tests, real run on the Windows eyeball list.
6. **GPU mode**: change #2b + `nvidia-smi` detection in the installer (the GUI's CPU retry shipped in slice 1). Spike done statically: the win wheel loads `cublas64_12.dll` and bundles cuDNN shim 9.1.0.70.
   Rulings: pin `nvidia-cublas-cu12` + `nvidia-cudnn-cu12==9.1.0.70` (win32); GPU iff driver ≥ 528.33 and VRAM ≥ 4 GB; install screen shows detection + "Use GPU" checkbox, headless `--cpu`; Repair's checkbox is editable and starts at the saved `device=`, pre-ticked on a GPU host when the installed app predates 1.2.0 (CPU-only v1.1.0 wrote `cpu` unasked; ruled 2026-09-26), Clean re-detects, a same-window retry uses the checkbox; first real GPU run → Windows eyeball list.
7. ~~Code signing~~ dropped 2026-09-26 → `archived/2026-09-25-smartscreen-code-signing.md`; replaced by the venv setup launcher (see *SmartScreen* ruling).

Backlog (unordered): VAD sliders (needs a `--vad` override that sets `vad_filter=True` via `config.set(...)` on the variant's config before `ServiceFactory.create_transcription_service`, `transcription_commands.py:250`; `vad_threshold` / `vad_parameters.min_silence_duration_ms` live in `preprocess/config.py:262-276`) · in-app update check · multilingual "Other" profile.

## Risks / verify-first

- **Native Windows has never been run.** ctranslate2, av and onnxruntime ship Windows wheels; confirm on first install. The Windows-specific `torchcodec` problem is already sidestepped (`pyannote_runner.py` feeds waveforms directly).
- **External binary:** variant 61 has no preprocess steps (`registry.py:252`), but `inspect_audio` always calls **`ffprobe`** (`preprocess/io.py:30,44`). The installer's ffmpeg zip must include `ffprobe.exe`, and the GUI prepends its `bin` dir to `PATH`.
- **CPU fallback banner:** both presets say `device="cuda"`. In CPU mode the GUI sets `STT_DEVICE=cpu` to skip the probe and the warning banner.
- **GPU DLL mismatch (unverified):** the Windows ctranslate2 wheel's CUDA major version is unknown, and a missing cuBLAS often fails only at the *first inference*, not at model load. That is why the GUI's CPU retry wraps the whole job instead of relying on `model_loader.py`'s load-time fallback. Slice 6's spike settles both points.
- **CLI details:** the command is `stt-faster transcribe process` (`cli/main.py:19`). Use `--variant`, never `-v`, which is bound to both `--variant` and `--verbose` (`transcription_commands.py:501,503`). The CLI default is `--diarize` (`:504-506`), so the GUI must pass `--no-diarize` unless Extras is set up.
- **Smart App Control** (company PC, 2026-09-26): the browser-launched exe runs, later launches of it are blocked; the uv launchers, uv Python and wheel DLLs run.

## Out of scope

Paid code-signing certificates · AMD/Intel GPUs · a fat frozen app · Mac/Linux installers.
