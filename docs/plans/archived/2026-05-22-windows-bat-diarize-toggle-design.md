# Windows .bat diarize toggle — design

**Date:** 2026-05-22
**Status:** Approved, ready to implement.
**Scope:** `scripts/windows/_runtime.bat` + the 7 `transcribe_*.bat` files. No backend changes.

## Goal

Make diarization runtime-aware and per-bat configurable:

- **Docker branch defaults OFF** (new behavior — diarization is slow on the Docker/CPU path on most users' machines, and pyannote model warmup adds noticeable startup time).
- **WSL branch defaults ON** (unchanged from today — the dev machine has GPU + cuDNN preload wired in c667c3f).
- **Per-bat override** via a knob at the top of each .bat — same place users already edit `VARIANTS=44`.

## Non-goals

- Changing any backend / CLI flag semantics.
- Centralizing the `VARIANTS=` knob (out of scope, but the same pattern would apply later).
- Touching `compare_variants.bat`, `check_recent.bat`, `report_only.bat` — none of them diarize today.

## Design

### Knobs (each `transcribe_*.bat`, near the existing `VARIANTS=44`)

```bat
set "VARIANTS=44"
set "DIARIZE="          REM "" = runtime default (wsl=on, docker=off); "0" = off; "1" = on
set "NUM_SPEAKERS=2"    REM only used when diarize resolves to on; "" = auto-detect
```

Same three lines copied identically into all 7 bats. Users twiddle them in place.

### Resolution (added to `_runtime.bat`)

`_runtime.bat` already exports `STT_RUNTIME`, `STT_WSL_REPO`, `STT_AUDIO_DIR_RESOLVED`, `STT_AUDIO_DIR_WSL`. Add one more output: `STT_DIARIZE_ARGS` — the fully composed flag string that drops into both runtime command lines.

Resolution rules (in order):

| `DIARIZE` value | `STT_RUNTIME` | Resulting `STT_DIARIZE_ARGS` |
|-----------------|---------------|------------------------------|
| `1`             | any           | `--diarize` + `--num-speakers !NUM_SPEAKERS!` (if non-empty) |
| `0`             | any           | *(empty)* |
| unset / empty   | `wsl`         | `--diarize` + `--num-speakers !NUM_SPEAKERS!` (if non-empty) |
| unset / empty   | `docker`      | *(empty)* |

Implementation sketch (inserted after the runtime-probe block, before the banner):

```bat
REM ---- Diarize resolution: caller knobs + runtime-aware default ----
set "STT_DIARIZE_ARGS="
set "STT_DIARIZE_ON=0"
if /i "%DIARIZE%"=="1" set "STT_DIARIZE_ON=1"
if not defined DIARIZE if /i "%STT_RUNTIME%"=="wsl" set "STT_DIARIZE_ON=1"
if defined DIARIZE if "%DIARIZE%"=="" if /i "%STT_RUNTIME%"=="wsl" set "STT_DIARIZE_ON=1"
if /i "%DIARIZE%"=="0" set "STT_DIARIZE_ON=0"

if "%STT_DIARIZE_ON%"=="1" (
    set "STT_DIARIZE_ARGS=--diarize"
    if defined NUM_SPEAKERS if not "%NUM_SPEAKERS%"=="" set "STT_DIARIZE_ARGS=--diarize --num-speakers %NUM_SPEAKERS%"
)
```

Note: the `defined X && "X"==""` double-check covers both `set "DIARIZE="` (defined but empty) and not-setting-it-at-all. CMD's `if not defined` returns false for `set "X="`, so we need both branches.

### Banner line

`_runtime.bat` currently prints:
```
[stt-faster] runtime: wsl
[stt-faster] audio:   D:\recordings\2026-05-22
[stt-faster] WSL repo: /home/kristjans/projects/stt-faster
```

Add one line, after the existing block:
```
[stt-faster] diarize: on (--num-speakers 2)
```
or
```
[stt-faster] diarize: off
```

Confirms at a glance which branch the user is on without scrolling — same philosophy as the existing GPU-fallback banner.

### Per-bat edit

For each of the 7 `transcribe_*.bat` files, two changes:

1. Insert the two knob lines (`DIARIZE=`, `NUM_SPEAKERS=2`) directly after `set "VARIANTS=...`.
2. In both runtime branches (WSL line at L36 and Docker line at L42 in `transcribe_estonian_Desk.bat`; same shape in the others), replace the hardcoded `--diarize --num-speakers 2` with `!STT_DIARIZE_ARGS!`.

WSL branch becomes:
```bat
wsl -e bash -c "... .venv/bin/python scripts/transcribe_manager.py process '!STT_AUDIO_DIR_WSL!' --language et --output-format txt !STT_DIARIZE_ARGS! --variants '!VARIANTS_COMMA!'"
```

Docker branch:
```bat
docker run --rm ^
  ...
  stt-faster:latest process /workspace --language et --output-format txt !STT_DIARIZE_ARGS! --variants "!VARIANTS_COMMA!"
```

When `STT_DIARIZE_ARGS` expands empty, CMD collapses the double space cleanly and the CLI sees no `--diarize`. No quoting concerns (no spaces inside the value when present).

## Key decisions (and why)

1. **Resolution in `_runtime.bat`, not per-bat.** The helper already owns runtime detection; adding diarize resolution there keeps the new "wsl-on / docker-off" rule in one place. Future changes (add `--min-speakers`, change defaults) are a one-file edit instead of 7-way copy-paste. Cost: ~12 lines + one new contract entry at the top of `_runtime.bat`.

2. **Sentinel `""` for "runtime default", explicit `0`/`1` for force.** Matches how CMD users typically toggle (`set "DIARIZE=1"`), keeps the unset case meaningful (don't think about it → get sensible per-runtime behavior). Truthy strings like `on`/`off` were considered but `1`/`0` are unambiguous and shorter.

3. **`NUM_SPEAKERS` empty → auto-detect.** Lets a user diarize without pre-committing to a count (`set "NUM_SPEAKERS="`). Default stays `2` because that's what every current bat hardcodes.

4. **Single `STT_DIARIZE_ARGS` output, not separate `STT_DIARIZE_FLAG` + `STT_NUM_SPEAKERS_FLAG`.** Caller sites stay clean — one expansion drops the whole block in. Compose-time logic stays in the helper where it belongs.

5. **No environment-variable override path.** A parent shell setting `DIARIZE=1` will be wiped by `set "DIARIZE="` at the top of the .bat. Matches the `VARIANTS=` pattern. If env-var override is wanted later, change `set "DIARIZE="` to `if not defined DIARIZE set "DIARIZE="` — trivial future tweak, not needed for v1.

## Caller-contract update (top of `_runtime.bat`)

Append to the existing block comment:

```
REM Caller may also set (optional):
REM   DIARIZE          - "" / unset = runtime default; "0" = force off; "1" = force on
REM   NUM_SPEAKERS     - speaker count for diarization; "" = auto-detect
REM
REM Additional output:
REM   STT_DIARIZE_ARGS - composed CLI args ("--diarize --num-speakers 2", or empty)
```

## Test plan

Manual only — these are Windows-side bats with no test harness. Verify on the dev box (the only machine that has both runtimes available):

1. **WSL default path** — run `transcribe_estonian_Desk.bat` unchanged knobs. Banner shows `diarize: on (--num-speakers 2)`. Output TXT has speaker labels. (Regression check — must match today's behavior.)
2. **Docker default path** — kill WSL probe (e.g., temporarily rename `.venv`), rerun. Banner shows `diarize: off`. Docker `run` line has no `--diarize`. Output TXT has no speaker labels.
3. **Docker forced on** — set `DIARIZE=1` at top, rerun. Banner shows `diarize: on (--num-speakers 2)`. Docker run line includes both flags.
4. **WSL forced off** — set `DIARIZE=0`, rerun. Banner shows `diarize: off`. No `--diarize` in the wsl command.
5. **Auto-detect speakers** — set `DIARIZE=1` and `NUM_SPEAKERS=`, rerun. Banner shows `diarize: on` (no parenthetical). CLI args: `--diarize` only, no `--num-speakers`.

If 1 + 2 pass, the rest are wiring confidence — the resolution logic is small enough to read by eye.

## Implementation order

1. `_runtime.bat` — add caller-contract comment, resolution block, banner line.
2. `transcribe_estonian_Desk.bat` — apply the two-edit pattern (knobs + `!STT_DIARIZE_ARGS!`). Verify by eye that L36/L42 still parse.
3. Manual smoke test 1 + 2 above on the dev box.
4. Apply the same pattern to the other 6 bats once smoke-tested.
5. Commit as one conventional commit: `feat(windows): make diarize toggleable per-bat, default off on docker`.

## Risks

- **Empty-arg expansion edge cases** — if some CMD quirk turns a stray space into a positional arg, the CLI could misparse. Mitigation: smoke test 2 explicitly checks the docker-off case (the new behavior most exposed to this).
- **`if not defined X` semantics** — CMD's `if not defined` returns false when `set "X="` is in effect. The resolution block handles both cases explicitly. Verified against existing patterns in `_runtime.bat` (e.g., `if defined STT_AUDIO_DIR`).
- **Banner drift** — if the banner format changes elsewhere (GPU fallback, etc.), the new diarize line should stay aligned with the same prefix. Minor; not blocking.
