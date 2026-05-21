# Portable Windows Setup for stt-faster Transcription Bats

**Date:** 2026-05-21
**Status:** Approved design, not yet implemented.

## Goal

Let a user clone this repo on **any Windows computer**, run a single `setup.bat`, and then use the existing `scripts\windows\transcribe_*.bat` files to transcribe audio — with no editing of hardcoded paths.

On the current dev machine (which has WSL2 with the repo at `/home/kristjans/projects/stt-faster`), the bats keep working exactly as they do today, transparently using the WSL venv. On a fresh machine without WSL (or without the repo inside WSL), the same bats fall back to a native Windows `.venv` that `setup.bat` created.

## Background

### Current shape

- Repo lives at `/home/kristjans/projects/stt-faster/` inside WSL2.
- `scripts/windows/*.bat` are Windows batch files committed in the repo, but the user's workflow is to **copy** one into the audio folder (e.g. `C:\Users\PC\Downloads\transcribe\`) and double-click it.
- Inside each bat: `%~dp0` is the audio folder; the script translates that Windows path to a `/mnt/c/...` WSL path and runs `wsl -e bash -c "cd /home/kristjans/projects/stt-faster && .venv/bin/python scripts/transcribe_manager.py process …"`.
- The WSL repo path `/home/kristjans/projects/stt-faster` is **hardcoded** in every bat.
- `setup.bat` does not exist yet.

### Why this needs to change

The hardcoded WSL path is tied to one machine. To use these bats on another computer the user has to manually edit every file. The fix is a portable `setup.bat` (native Windows, no WSL assumption) and a small shared runtime helper that probes for WSL and falls back to the local Windows venv if WSL isn't a usable host.

### Reference precedent

`projects/edf-budget-planner/setup.bat` is the pattern to follow for the native-Windows installer: `git pull` → `py -3.x` probe → fresh `.venv` → `pip install -e .` → friendly "next steps" block.

## Chosen approach (with rationale)

| Decision | Chosen | Why |
|---|---|---|
| WSL vs native | **Native-only `setup.bat`; bats probe WSL at runtime** | User explicitly asked for it. Keeps `setup.bat` simple and predictable. Probe pattern preserves current-machine behavior. |
| Repo location | **Cloned to Windows disk** (e.g. `C:\projects\stt-faster`) | `setup.bat` runs from the cloned repo root. WSL is never touched by `setup.bat`. |
| Python version | **3.12** via `py -3.12` | Matches the WSL/Linux side (`pyproject.toml` pins `>=3.12,<3.13`). |
| Audio folder discovery | **Env var `STT_AUDIO_DIR`, falling back to the bat's own folder (`%~dp0`)** | Power users set the env var once; casual users keep the existing "copy bat into audio folder" workflow. |
| Model prefetch | **Prompt y/N at end of `setup.bat`** | Optional one-time download (~3 GB total for et-large + turbo). |
| Shared logic | **`scripts\windows\_runtime.bat`** | 10 bats need the same WSL-probe + path-translation + venv-selection logic. Inlining would duplicate ~30 lines across 10 files. |
| Model set | **Estonian (et-large/et-32) + English (turbo/distil)** | Covers all 7 `transcribe_*.bat` files. Different download patterns: Estonian uses CT2 subfolder only, English uses full snapshot. |
| Torch on Windows | **Verified not pulled** by transitive deps; `setup.bat` asserts torch-free as a guard rail | `faster-whisper` only requires torch via its `[conversion]` extra; ctranslate2 is the runtime path. |

## Design

### 1. `setup.bat` (new, repo root)

Native Windows installer. Mirrors `edf-budget-planner/setup.bat` plus a model-prefetch prompt and a torch-free guard.

Steps, in order:

1. `git pull` — warn and continue on failure (offline-friendly).
2. Probe `py -3.12 --version`. If missing → print install URL + `exit /b 1`.
3. Remove stale `.venv\` if present (error and abort if removal fails).
4. `py -3.12 -m venv .venv`.
5. `call .venv\Scripts\activate.bat`, then `pip install -e .`. Abort on failure.
6. **Torch guard**: `pip list | findstr /I "^torch "`. If found, print a loud `!!!`-bordered warning (does not abort — informational, so future deps drift is visible).
7. **Model prefetch prompt**: `Prefetch Whisper models for Estonian and English now? (~3 GB) [y/N]`.
   - On `y`: `set HF_HOME=%USERPROFILE%\.cache\hf`, then run `.venv\Scripts\python scripts\prefetch_models.py`.
   - On anything else: skip; first transcription run downloads lazily (current WSL behavior).
8. Print "Setup complete" block listing what to do next:
   - Optionally `set STT_AUDIO_DIR=<path>` (Windows env var).
   - Run any `scripts\windows\transcribe_*.bat` (either copied into the audio folder, or in-place when `STT_AUDIO_DIR` is set).
   - GPU note: native Windows runs CPU unless CUDA + cuDNN are manually installed; the bats print a highly-visible banner when fallback fires.
9. `pause`.

### 2. `scripts/prefetch_models.py` (new, ~30 lines)

A small Python helper that calls the existing model-resolver helpers (`_get_estonian_model_path`, `_get_cached_model_path`) for each model used by the transcribe bats. Pure download — no `WhisperModel(...)` construction, so it doesn't probe CUDA on a CPU-only box.

Models prefetched:

| Preset(s) using it | Model ID | Download shape |
|---|---|---|
| `et-large`, `et-32` | `TalTechNLP/whisper-large-v3-turbo-et-verbatim` | `allow_patterns=["ct2/*"]` (only the CT2 subfolder) |
| `turbo`, `distil` | `Systran/faster-distil-whisper-large-v3` | Full snapshot |

Lives under `scripts/` so it's reusable from WSL too (`./.venv/bin/python scripts/prefetch_models.py`).

### 3. `scripts/windows/_runtime.bat` (new shared helper, ~50 lines)

`call`-ed by every other bat in `scripts/windows/`. Owns these decisions in one place:

- **Repo root**: derived from `%~dp0..\..` (the helper lives at `scripts\windows\_runtime.bat`).
- **WSL probe**:
  - `where wsl` to check WSL is installed.
  - `wsl -e test -x /home/kristjans/projects/stt-faster/.venv/bin/python` to check the legacy WSL repo+venv is reachable.
  - Both pass → `STT_RUNTIME=wsl`. Otherwise → `STT_RUNTIME=native`. The WSL path stays **hardcoded** per spec — keeps the current machine working, fails cleanly on every other machine.
- **Audio folder resolution**:
  - If `STT_AUDIO_DIR` is set in the environment → use it.
  - Otherwise → use `STT_CALLER_DIR` (the calling bat sets this to `%~dp0` before calling the helper).
  - Trim trailing backslash.
- **WSL path translation** (only relevant when `STT_RUNTIME=wsl`): convert `C:\foo\bar` → `/mnt/c/foo/bar`, supporting C/D/E drive letters (matches today's bat logic, generalized).
- **Visible banner** echoed to stdout:
  ```
  [stt-faster] runtime: <wsl|native>
  [stt-faster] audio:   <resolved windows path>
  [stt-faster] WSL repo: <only printed when runtime=wsl>
  ```
  Same philosophy as the GPU-fallback banner: runtime selection is silent-but-load-bearing, so make it visible.

Variables exported to the calling bat:

| Variable | Meaning |
|---|---|
| `STT_RUNTIME` | `wsl` or `native` |
| `STT_REPO_WIN` | Absolute Windows path to repo root |
| `STT_WSL_REPO` | Hardcoded WSL repo path (for `cd` in WSL invocation) |
| `STT_AUDIO_DIR_RESOLVED` | Windows path to audio dir |
| `STT_AUDIO_DIR_WSL` | Same path translated to `/mnt/c/...` form |

### 4. Each `transcribe_*.bat` (modified, 7 files)

Each shrinks from ~95 lines to ~30 lines. Template:

```bat
@echo off
REM Audio Transcription - <LANG>, <PRESET> (variant <N>)
set "VARIANTS=44"

setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
call "%~dp0_runtime.bat"
if errorlevel 1 exit /b 1

REM (variant-list normalization — unchanged from today)

if "!STT_RUNTIME!"=="wsl" (
    wsl -e bash -c "export HF_HOME=\"$HOME/.cache/hf\" && export HF_HUB_CACHE=\"$HF_HOME/hub\" && cd !STT_WSL_REPO! && .venv/bin/python scripts/transcribe_manager.py process '!STT_AUDIO_DIR_WSL!' --language <et|en> --output-format both --variants '!VARIANTS_COMMA!'"
) else (
    set "HF_HOME=%USERPROFILE%\.cache\hf"
    set "HF_HUB_CACHE=%USERPROFILE%\.cache\hf\hub"
    pushd "!STT_REPO_WIN!"
    .venv\Scripts\python scripts\transcribe_manager.py process "!STT_AUDIO_DIR_RESOLVED!" --language <et|en> --output-format both --variants "!VARIANTS_COMMA!"
    popd
)
pause
```

Only the variant number, language flag, optional `--preset`, and banner text differ between the 7 bats.

### 5. Utility bats (modified)

| Bat | Current | After |
|---|---|---|
| `check_status.bat` | Single `wsl -e bash -c` invocation | `call _runtime.bat`, then dispatch the same Python invocation to WSL or native venv. No audio dir needed. |
| `report_only.bat` | `wsl -e bash -c "cd … && make variant-report"` | Replace `make variant-report` with direct `.venv/bin/python scripts/generate_variant_report.py` (works in both runtimes — `make` need not exist on Windows). |
| `compare_variants.bat` | 4 separate WSL invocations (version check, variant enumeration, transcription, report) | Each becomes the same dispatch block, with variant enumeration going through Python directly instead of via bash quoting. |

### 6. `scripts/windows/HOW_TO_USE.txt` and `README.md` (modified)

Rewrite the quick-start to cover:
- The `git clone` + `setup.bat` flow.
- Two ways to use the transcribe bats: (a) copy into audio folder (legacy), or (b) `set STT_AUDIO_DIR=…` and run from `scripts\windows\`.
- A note that WSL is auto-detected from the legacy hardcoded path; on machines without that path, native fires.
- A note about CPU-only on native Windows by default.

## File inventory

**New (3):**
- `setup.bat` (repo root, ~50 lines)
- `scripts/prefetch_models.py` (~30 lines)
- `scripts/windows/_runtime.bat` (~50 lines)

**Modified (12):**
- `scripts/windows/transcribe_estonian_Desk.bat`
- `scripts/windows/transcribe_estonian_Teams.bat`
- `scripts/windows/transcribe_estonian_Online.bat`
- `scripts/windows/transcribe_estonian_32bit_cpu.bat`
- `scripts/windows/transcribe_english_Desk.bat`
- `scripts/windows/transcribe_english_Teams.bat`
- `scripts/windows/transcribe_english_Online.bat`
- `scripts/windows/check_status.bat`
- `scripts/windows/report_only.bat`
- `scripts/windows/compare_variants.bat`
- `scripts/windows/HOW_TO_USE.txt`
- `scripts/windows/README.md`

**Unchanged:** all `backend/` Python code, `Makefile`, `pyproject.toml`. The Windows-vs-WSL distinction stops at the bat layer.

## Key decisions (numbered, append-only)

1. **Native-only `setup.bat`; WSL is never bootstrapped by setup.** User-specified. Simpler installer, predictable behavior.
2. **WSL probe in `_runtime.bat` uses the hardcoded path `/home/kristjans/projects/stt-faster`.** User-specified. On the current machine it passes (preserves today's behavior). On any other machine it fails and native fires. No need to make the WSL repo path configurable in this iteration.
3. **`STT_AUDIO_DIR` env var with `%~dp0` fallback.** User-specified. Best of both worlds: power-users get an env var, casual users keep the copy-to-folder muscle memory.
4. **Models prefetched: Estonian (`TalTechNLP/whisper-large-v3-turbo-et-verbatim`) and English (`Systran/faster-distil-whisper-large-v3`).** Covers all 7 transcribe bats. User explicitly requested both.
5. **Shared runtime helper instead of inlining.** 10 bats with identical probe+path logic would drift; one helper file is the right shape.
6. **Torch guard in `setup.bat`.** Verified torch is not in the dep tree (faster-whisper requires torch only under the `[conversion]` extra; ctranslate2 is the runtime). Guard is informational so future dep drift is visible.

## Out of scope

- Installing CUDA / cuDNN on native Windows. The bats print a highly-visible GPU-fallback banner (already shipped in `backend/model_loader.py`) when fallback fires.
- Detecting WSL distros other than the default (`wsl -d Ubuntu` etc.).
- Auto-discovering the WSL repo path (e.g. searching common locations under `/home/`).
- Persisting `STT_AUDIO_DIR` via `setx`. User sets it manually if they want.
- A separate Windows-only requirements file. `pyproject.toml` is platform-agnostic and verified torch-free.
- Migrating `runs.jsonl` history between WSL and native runtimes.

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| User clones to a path with spaces (`C:\Users\My Name\stt-faster`) | Low | All `pushd`/`call`/path expansions in `_runtime.bat` and the transcribe bats are double-quoted. Verify once during implementation. |
| GPU users on native Windows get CPU silently | Low | GPU-fallback banner already shipped. Doc note in HOW_TO_USE.txt. |
| `runs.jsonl` history splits across `~/.local/share/...` (WSL) and `%USERPROFILE%\.local\share\...` (native) on dual-runtime machines | Low | Acceptable per spec. Documented in HOW_TO_USE.txt. |
| Hardcoded WSL repo path becomes wrong on the user's dev machine if they move the repo | Low | Edit one constant in `_runtime.bat`. Document the location in HOW_TO_USE.txt. |
| `findstr /I "^torch "` torch guard false-negatives if pip uses different formatting | Low | Worst case: a real torch install slips through silently. Not a correctness bug, just a missed warning. |

## Verification

### On the current dev machine (WSL with the repo at the hardcoded path)

1. Run any `transcribe_*.bat`. Expect banner `[stt-faster] runtime: wsl`. Behavior identical to today.
2. Rename WSL `.venv` temporarily and re-run; expect banner `runtime: native` (only meaningful if `setup.bat` has also been run to populate the Windows `.venv`).

### On a fresh Windows machine (no WSL, or WSL without the repo)

1. `git clone https://…/stt-faster.git C:\projects\stt-faster`.
2. `cd C:\projects\stt-faster`.
3. `setup.bat` → answer `y` to the prefetch prompt.
4. `set STT_AUDIO_DIR=C:\Users\PC\Downloads\transcribe` (optional).
5. Run `scripts\windows\transcribe_estonian_Desk.bat`.
6. Expect banner `runtime: native`, GPU-fallback banner fires on first model load (no CUDA on the box), transcripts land under `<audio_dir>\processed\`.

### Acceptance criteria

- Zero edits to any committed file are needed to install on a fresh Windows machine.
- The user's current dev machine continues to use WSL unchanged.
- Both Estonian and English bats work in both runtimes.
- Torch is not installed on Windows.
