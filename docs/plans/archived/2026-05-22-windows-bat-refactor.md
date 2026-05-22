# Windows .bat refactor — de-dup + adopt docker auto-build everywhere

**Status:** Proposed (2026-05-22). Not yet started.
**Scope:** `scripts/windows/*.bat` only. `setup.bat` (repo root) stays as-is (see decision 6).

## Goal

1. De-duplicate ~30 lines of boilerplate shared across 7 transcribe bats.
2. Lift the docker-build robustness from `transcribe_english_Online_docker.bat` (auto-build prompt, HF env passthrough, diarize/HF_TOKEN fail-fast) so every bat's docker code-path gets it.

## Files in scope

- `scripts/windows/_runtime.bat` — existing shared helper. Gets extended.
- 7 regular transcribe bats: `transcribe_english_{Online,Desk,Teams}.bat`, `transcribe_estonian_{Online,Desk,Teams,32bit_cpu}.bat`.
- `transcribe_english_Online_docker.bat` — becomes a thin wrapper.
- Utility bats (already use `_runtime.bat`): `check_recent.bat`, `compare_variants.bat`, `report_only.bat` — unchanged code, but benefit from auto-build via `_runtime.bat`.
- `setup.bat` (repo root) — **out of scope** (see decision 6).
- Docs: `scripts/windows/README.md`, `scripts/windows/HOW_TO_USE.txt`.

## Design decisions

1. **Variant-normalization** → new `_variants.bat` (not `_runtime.bat`). Utility bats don't need it; `compare_variants.bat` builds its variants from Python anyway. Keep `_runtime.bat` orthogonal.
2. **Banner / footer** → new `_banner.bat` + `_footer.bat`. Per-bat strings differ in only 2-3 places; collapse to two `call`s.
3. **Docker invocation** → new `_docker_run.bat` owns mounts, `DOCKER_ENV_ARGS`, fail-fast guard, and the `docker run` line. Caller sets `STT_CLI_TAIL` (e.g. `--preset turbo --language en --output-format txt`).
4. **Auto-build prompt** → moves into `_runtime.bat`. Every utility bat that currently hits "no runtime available" benefits. Add opt-out `STT_NO_AUTOBUILD=1` for unattended runs.
5. **`transcribe_english_Online_docker.bat`** → keep as a thin wrapper that sets `STT_FORCE_DOCKER=1` + `STT_TITLE_SUFFIX=[DOCKER FORCED]`. `_runtime.bat` checks `STT_FORCE_DOCKER` and skips the WSL probe.
6. **`setup.bat`** → stays independent. It does `git pull` + cache-dir mkdir + *always-rebuild* (different intent from "build if missing"). Forcing it through the new helper would invert that. One duplicated `docker build` line is the price.
7. **External refs** → none (per Plan-agent grep). Internal refactor only; no renames.
8. **CMD parser constraints** (mandatory in every new helper header comment):
   - `@echo off`; helpers assume caller did `setlocal enabledelayedexpansion`; helpers do NOT `setlocal` themselves (would hide outputs from caller — same convention as current `_runtime.bat`).
   - Auto-build flow uses flat `goto :label`. **Never** nested if/else with caret-escaped echoes — the existing docker bat's comments explain why.
   - Variable existence checks use `if defined VAR if not "!VAR!"==""` to handle both unset and empty-defined cases.
   - Delayed expansion propagates across `call` to another `.bat` — helpers can use `!VAR!` set by caller.

## Step-by-step

Steps 1-5 are independent helper files; old bats keep working through step 5. Step 6 must follow steps 1-5. Step 7 follows 6. Step 8 last.

### Step 1 — Extend `_runtime.bat` (auto-build + force-docker)
- Add inputs: `STT_FORCE_DOCKER` (skip WSL probe if `1`), `STT_NO_AUTOBUILD`.
- Add output: `STT_REPO_GUESS` (from `%~dp0..\..`).
- After docker probe, if image missing AND `!STT_NO_AUTOBUILD!`!="1": run the goto-based auto-build flow from `transcribe_english_Online_docker.bat:41-83` (verbatim port; `STT_REPO_GUESS` replaces the inline path).
- Update the "no runtime" error to fire only after WSL failed AND docker failed AND (no Dockerfile OR declined OR build failed).

**User verification:** `docker rmi stt-faster:latest`, then run any transcribe bat → prompt appears; N exits with rc=1; Y builds and continues.

### Step 2 — Create `_variants.bat`
- Contract: caller sets `VARIANTS` (space-separated). Helper sets `VARIANTS_COMMA` + `VARIANT_COUNT`.
- Body: the loop currently at lines 17-27 of any transcribe bat.

**User verification:** edit one bat to `set "VARIANTS=1 36 44"`, run → banner shows `Variants: 1,36,44 (3 variants)`.

### Step 3 — Create `_banner.bat` and `_footer.bat`
- `_banner.bat`: caller sets `STT_TITLE`, `STT_MODEL`, `STT_LANG`, optional `STT_TITLE_SUFFIX`. Reads `VARIANTS_COMMA`/`VARIANT_COUNT`. Prints the 4-content-line header (title, model, language, variant(s)) wrapped in two `=` separators + trailing blank.
- `_footer.bat`: reads `STT_AUDIO_DIR_RESOLVED`. Prints "Processing Complete!" + Processed/Failed paths + `pause`.

**User verification:** capture console output of one bat before/after — byte-identical.

### Step 4 — Create `_docker_run.bat`
- Reads `STT_AUDIO_DIR_RESOLVED`, `STT_DIARIZE_ARGS`, `VARIANTS_COMMA`, `STT_CLI_TAIL`, `DIARIZE`.
- Composes `DOCKER_ENV_ARGS` from `HF_TOKEN` + `HF_XET_HIGH_PERFORMANCE` (only if defined).
- Runs diarize/HF_TOKEN fail-fast guard (`transcribe_english_Online_docker.bat:102-110`).
- Executes `docker run` with mounts + env args + `process /workspace !STT_CLI_TAIL! !STT_DIARIZE_ARGS! --variants "!VARIANTS_COMMA!"`.
- Exits with the `docker run` errorlevel.

**User verification:** with `DIARIZE=1` and `HF_TOKEN` unset, bat errors out cleanly before any docker call.

### Step 5 — Create `_transcribe.bat` (dispatch body)
- Owns the wsl/docker branch dispatch (lines 37-45 of current transcribe bats).
- Reads `STT_RUNTIME`, `STT_CLI_TAIL`, standard `STT_*` + `VARIANTS_COMMA`.
- WSL branch: invokes `wsl -e bash -c "..."` with the inline `HF_HOME` exports.
- Docker branch: delegates to `_docker_run.bat`.
- Single env var `STT_CLI_TAIL` for both branches (identical strings today — saves a knob).

### Step 6 — Rewrite the 7 regular transcribe bats
Each shrinks to ~15 lines:
```bat
@echo off
set "VARIANTS=52"
set "DIARIZE="
set "NUM_SPEAKERS=2"
setlocal enabledelayedexpansion
set "STT_CALLER_DIR=%~dp0"
set "STT_TITLE=Audio Transcription - ENGLISH (Online)"
set "STT_MODEL=Systran faster-distil-whisper-large-v3 (turbo)"
set "STT_LANG=English"
set "STT_CLI_TAIL=--preset turbo --language en --output-format txt"
call "%~dp0_runtime.bat"  || ( pause & exit /b 1 )
call "%~dp0_variants.bat"
call "%~dp0_banner.bat"
call "%~dp0_transcribe.bat"
call "%~dp0_footer.bat"
```
Apply to all 7. Differ only in the 4 `set` lines for title/model/lang/CLI tail (+ `VARIANTS`/`DIARIZE`/`NUM_SPEAKERS` knobs).

**Do one bat at a time and have the user verify** end-to-end on real audio.

**User verification:** for each rewritten bat — run on real audio; compare `run_meta.json` `cli_args` with the prior run.

### Step 7 — Rewrite `transcribe_english_Online_docker.bat` as thin wrapper
Same template as step 6, plus:
- `set "STT_FORCE_DOCKER=1"` before the `_runtime.bat` call.
- `set "STT_TITLE_SUFFIX=[DOCKER FORCED]"`.

**User verification:** run on a machine where WSL is healthy → banner shows `runtime: docker (forced)`, docker pipeline executes.

### Step 8 — Docs
Update `scripts/windows/README.md` and `scripts/windows/HOW_TO_USE.txt`: list the new helpers, add an "Adding a new transcribe variant" section showing the 4-line knob template.

## Risks / open questions

- **Helper proliferation**: 5 new `_*.bat` files for ~50 lines of dedup. Win is structural: an 8th variant becomes a 4-line change. If the user prefers fewer files, fold `_variants` + `_banner` + `_footer` into one `_transcribe_helpers.bat` exposing labeled entry points via `call :label`. Default: 5 files (labeled-entry-point bats are harder to debug).
- **`_runtime.bat` contract grows**: now owns auto-build. If `STT_REPO_GUESS` is wrong (bat copied next to audio), the existing `:no_repo` branch handles it — same UX as today's docker-forced bat.
- **WSL HF env passthrough asymmetry**: the docker branch passes `HF_TOKEN`; the WSL branch inherits the WSL user's env. Persists by design after refactor — worth a one-line comment in `_transcribe.bat`.

## Critical files

- `scripts/windows/_runtime.bat` (extended in step 1)
- `scripts/windows/transcribe_english_Online_docker.bat` (source for the lift; thin wrapper in step 7)
- `scripts/windows/transcribe_english_Online.bat` (template for the rewrite)
- `scripts/windows/transcribe_estonian_32bit_cpu.bat` (only one with a different preset shape — `--preset et-32`)
- `scripts/windows/compare_variants.bat` (uses `_runtime.bat`; benefits from step 1 auto-build but otherwise unchanged)
