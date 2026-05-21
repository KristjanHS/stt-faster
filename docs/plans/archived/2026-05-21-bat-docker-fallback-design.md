# Windows `.bat` Docker fallback — design

**Date:** 2026-05-21
**Status:** SHIPPED (`3af8236` U1+U2+U3 + `73f67b6` Shape 3 positional fix).
**Author:** mybrain session w/ Claude

## §0 — Implementation note (added at archive time)

Shipped on `main` 2026-05-21 in two commits:
- `3af8236` — all three units (U1 `_runtime.bat`, U2 ten caller bats, U3
  `setup.bat` rewrite + `scripts/windows/README.md` + `HOW_TO_USE.txt`).
- `73f67b6` — Shape 3 fix: append `/workspace` positional to the
  `generate_variant_report.py` docker invocations in `report_only.bat`
  and `compare_variants.bat`. The original design template at §"Shape 3"
  inherited a gap from the WSL side, where the script's auto-detect
  fallback `Path("/mnt/c/Users/PC/Downloads/transcribe")` happens to
  exist on the dev machine. Inside the container that path is absent,
  so the script exited with "Error: Default directory not found" before
  reaching its outer-dir timestamp search. Fix: pass `/workspace`
  explicitly. Caught in code review.

§Architecture / §Implementation units operational facts are now reflected in
`scripts/windows/_runtime.bat`, the caller bats, `setup.bat`, and the
README/HOW_TO_USE docs. Decisions log + "Why not the alternatives"
rationale stays here as audit trail.

## Goal

When a Windows user double-clicks a `scripts/windows/*.bat` and WSL is not
available on their machine, fall back to running the work inside the existing
production Docker image instead of the current Windows-native `.venv`.

## Decisions (from clarify phase)

| Decision | Value |
|---|---|
| Runtime priority | WSL → Docker (native Windows venv dropped entirely) |
| GPU policy in Docker | CPU-only (acceptable to be slow on non-dev machines) |
| Image source | Pre-built locally via rewritten `setup.bat` (`docker build -t stt-faster:latest .`) |
| Bats in scope | All 10 bats under `scripts/windows/` (every caller of `_runtime.bat`) |
| No-runtime failure | Clear error referencing `setup.bat`, then `pause` + `exit /b 1` |

## Why not the alternatives

- **Three-tier (WSL → Docker → native venv).** Keeps the brittle Windows-venv
  path alive as a third fallback, defeating the simplification. User
  explicitly chose to drop native.
- **GPU in Docker.** Would require NVIDIA Container Toolkit, a cu130
  Dockerfile variant, and `--gpus` plumbing. Wrong for the "someone
  double-clicked this on a CPU laptop" scenario the fallback exists for.
- **Pull-from-registry image source.** No registry exists today; adding one
  is its own project and overkill for the current scale.
- **Consolidated `_exec.bat` helper (Approach B in the brainstorm).** The 10
  bats use four genuinely different invocation shapes
  (`transcribe_manager.py process`, `transcribe_manager.py status`,
  `generate_variant_report.py`, inline `python -c`). A shared helper would
  have to model all of them via env-var contracts and a `__AUDIO__`
  substitution token, concentrating complexity in one new file for an
  abstraction whose only payoff is "if a third runtime ever lands." The
  per-bat swap (Approach A) keeps each `else` branch local and shape-aware.

## Architecture

The existing two-stage flow is preserved:

```
scripts/windows/*.bat
   │
   ├── call _runtime.bat   ← probes runtime, sets STT_RUNTIME=wsl|docker|none
   │                          plus path variables (STT_AUDIO_DIR_RESOLVED, etc.)
   │
   └── if STT_RUNTIME==wsl ( wsl -e bash -c "..."           )
       else                ( docker run --rm -v ... image ... )
```

The only behavioral change: the `else` branch is now Docker instead of native
`.venv\Scripts\python`, and `_runtime.bat` can return `none` (handled by
existing `if errorlevel 1 (pause & exit /b 1)` in callers).

## Implementation units

Single PR. Three units sequenced for review clarity but all must land
together — U1 and U2 are coupled (U2's `else` body assumes U1 sets
`STT_RUNTIME=docker`).

### U1 — `scripts/windows/_runtime.bat`

- Add Docker probe between WSL probe and failure case:
  ```bat
  if /i "%STT_RUNTIME%"=="none" (
      where docker >nul 2>nul && docker image inspect stt-faster:latest >nul 2>nul && set "STT_RUNTIME=docker"
  )
  ```
  - `docker image inspect` (not `docker info`) — daemon-running AND
    image-built must both hold. A daemon-only probe would pass and then
    `docker run` would fail with a worse error.
- Change initial default `set "STT_RUNTIME=native"` → `set "STT_RUNTIME=none"`.
- Add explicit failure block after both probes:
  ```bat
  if /i "%STT_RUNTIME%"=="none" (
      echo [stt-faster] ERROR: no runtime available.
      echo [stt-faster] Install Docker Desktop and run setup.bat to build the image.
      exit /b 1
  )
  ```
- Remove `STT_REPO_WIN` derivation (`for %%I in ("%~dp0..\..") do ...`) — only
  ever used by the now-dead native branch.
- Banner update: `if /i "%STT_RUNTIME%"=="wsl" echo [stt-faster] WSL repo: ...`
  stays. No equivalent Docker-side line needed (image name is invariant).

### U2 — 10 caller bats (`scripts/windows/*.bat` minus `_runtime.bat`)

Each `else` body is replaced. Four shapes total; mapping below.

**Shape 1 — `process` (Estonian/English transcribe bats + the process call inside `compare_variants.bat`):**
```bat
) else (
    docker run --rm ^
      -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
      -v "%USERPROFILE%\.cache\hf:/home/appuser/.cache/hf" ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      stt-faster:latest process /workspace --language et --output-format txt --diarize --num-speakers 2 --variants "!VARIANTS_COMMA!"
)
```
Audio dir is mounted to `/workspace` (also the image `WORKDIR`), so the
positional becomes `/workspace`. `HF_HOME` is already set inside the image
(`ENV HF_HOME=/home/appuser/.cache/hf`, Dockerfile line 100), so the `-v`
suffices — no `-e HF_HOME=...` needed. The state-DB mount keeps `status`
calls in U2's check_status case seeing the same DB.

Used by: `transcribe_estonian_Desk.bat`, `transcribe_estonian_Teams.bat`,
`transcribe_estonian_Online.bat`, `transcribe_estonian_32bit_cpu.bat`,
`transcribe_english_Desk.bat`, `transcribe_english_Teams.bat`,
`transcribe_english_Online.bat`, plus the `process` call inside
`compare_variants.bat`. (Per-bat args differ — `--preset`, `--language`,
`--no-diarize`, `--variants` — keep each bat's existing args.)

**Shape 2 — `status` (`check_status.bat`):**
```bat
) else (
    docker run --rm ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      stt-faster:latest status
)
```
Only the state-DB mount matters; no audio dir, no HF cache.

**Shape 3 — `generate_variant_report.py` (`report_only.bat` and the report call inside `compare_variants.bat`):**
```bat
) else (
    docker run --rm ^
      -v "!STT_AUDIO_DIR_RESOLVED!:/workspace" ^
      -v "%USERPROFILE%\.local\share\stt-faster:/home/appuser/.local/share/stt-faster" ^
      --entrypoint python ^
      stt-faster:latest /app/scripts/generate_variant_report.py --far-speaker-range 252-291 --silence-range 19-61
)
```
Script path must be **absolute** (`/app/scripts/...`). The container
`WORKDIR` is `/workspace` (Dockerfile line 124), where audio is mounted —
a relative `scripts/...` would resolve there and fail to import.

**Shape 4 — inline `python -c "..."` (the first call in `compare_variants.bat`):**
```bat
) else (
    docker run --rm --entrypoint python stt-faster:latest -c "!VARIANT_PY!" > "!TEMP_VARIANTS_FILE!" 2>&1
)
```
No mounts needed (introspection only). cmd.exe redirects the `docker run`
stdout to `!TEMP_VARIANTS_FILE!` exactly as it redirected the native
`.venv\Scripts\python -c "..."` invocation.

### U3 — `setup.bat` rewrite + README sweep

Replace `setup.bat` entirely (no Python-venv steps remain). New file:

```bat
@echo off
REM stt-faster - Docker setup. Builds the production image used as the fallback
REM runtime when WSL is unavailable. WSL users do not need to run this; the
REM transcribe bats auto-detect WSL and use it directly.

git pull
if errorlevel 1 echo WARNING: git pull failed. Continuing with the local checkout.

echo === stt-faster - Docker setup ===

where docker >nul 2>nul
if errorlevel 1 (
    echo ERROR: docker not found on PATH.
    echo Install Docker Desktop from https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

docker info >nul 2>nul
if errorlevel 1 (
    echo ERROR: docker daemon is not reachable. Start Docker Desktop and re-run.
    pause
    exit /b 1
)

echo Building stt-faster:latest (this can take 5-15 minutes on first run)...
docker build -t stt-faster:latest "%~dp0."
if errorlevel 1 (
    echo ERROR: docker build failed.
    pause
    exit /b 1
)

REM Pre-create mount target dirs so docker run does not create them as root-owned.
if not exist "%USERPROFILE%\.cache\hf" mkdir "%USERPROFILE%\.cache\hf"
if not exist "%USERPROFILE%\.local\share\stt-faster" mkdir "%USERPROFILE%\.local\share\stt-faster"

echo.
echo === Setup complete ===
echo Next: double-click a transcribe_*.bat (or call it with STT_AUDIO_DIR set).
echo Models download lazily on first transcription run (~3 GB into %USERPROFILE%\.cache\hf).
pause
```

README sweep:
- Quick-start section: any "install Python 3.12 / run setup.bat to build .venv"
  language becomes "install Docker Desktop / run setup.bat to build the
  image."
- The setup.bat closing Notes block in the old file (`Native Windows runs
  CPU-only unless...`, `The bats auto-detect WSL...`) — the equivalent
  guidance is now baked into the new setup.bat output and is no longer
  needed in README.

## Out of scope

- Auto-starting Docker Desktop if installed-but-stopped (considered, rejected
  as too fragile — daemon startup is 30+ seconds and `start` path varies).
- GPU access via NVIDIA Container Toolkit (CPU-only is the decision).
- Publishing a registry image — separate project if it ever matters.
- Touching the WSL-side `_runtime.bat` probe (`wsl -e test -x ...`) — works
  fine, no reason to change.
- `select_variant.bat` / `.stt-variant.local` plumbing — the Docker fallback
  is CPU-only by decision, so it doesn't read the variant. The file
  remains for the WSL/dev side via `run_uv.sh`.

## Re-work audit

Three implementation units, each touches distinct files (U1: 1 file, U2: 10
files, U3: 1 file + README). No file is written then rewritten across
units. Pass.

## Decisions log

1. **2026-05-21** — Approach A (per-bat swap) over Approach B (consolidated
   `_exec.bat`). Reason: four distinct invocation shapes make B's
   substitution scheme carry more complexity than the per-bat `else` swap
   it would replace.
2. **2026-05-21** — Probe with `docker image inspect`, not `docker info`.
   Reason: image presence is load-bearing; a daemon-only probe would let
   `docker run` fail with a worse downstream error.
3. **2026-05-21** — Drop native Windows venv entirely (not three-tier
   fallback). Reason: user explicit; eliminates `setup.bat`'s Python-venv
   path, `STT_REPO_WIN`, and the `.venv\Scripts\python` else branches.
4. **2026-05-21** — Shape 3/4 require `/app/scripts/...` absolute paths,
   not relative. Reason: `WORKDIR /workspace` in the image means relative
   paths resolve against the audio mount, not the source tree.
