# Carry-over fixes — diarize plumbing + num_speakers validator + test_transcribe pyright

**Created**: 2026-05-25. Promoted from "Carry-over follow-ups" section of `memory/project_state.md`.

## Scope

Four small unrelated cleanups that have been sitting on the carry-over list since the Stage 2A pyannote review and the CUDA-extras D4 docs sweep. All four are independently safe; bundled here as one `/impag` run because each is too small to own its own plan doc and they touch related (but disjoint) regions of the diarize call chain.

## Tasks

### T1 — Tighten `num_speakers` validator (I2)

**File**: `backend/run_config.py:88-89`.

**Change**: bump the `num_speakers < 1` lower bound to `< 2`. pyannote diarization is undefined for `num_speakers=1`; the validator should reject it at config-build time rather than letting pyannote produce silent garbage at runtime.

```python
# was:
if self.num_speakers < 1:
    raise ValueError(f"num_speakers must be >= 1, got {self.num_speakers}")
# becomes:
if self.num_speakers < 2:
    raise ValueError(f"num_speakers must be >= 2, got {self.num_speakers}")
```

**Tests**: grep `tests/` for `num_speakers=1` / `num_speakers=0` constructions. Any test that relied on `1` as a legal value must change to `2` (or to `0`/`-1` if the test was probing the validator). Re-run `tests/unit/test_run_config.py` (or whichever file owns the validator test).

### T2 — Plumb `diarize` + `num_speakers` through `execute_variant` (I1)

**Files**: `backend/variants/executor.py`, `scripts/compare_transcription_variants.py`.

**Current state**: `execute_variant` (line 160) accepts `variant, audio_path, preset, language, *, context`. It internally dispatches to one of three helpers (`transcribe_with_baseline_params`, `transcribe_with_minimal_params`, `_transcribe_with_config`) — all three accept `diarize: bool = False`, `num_speakers: int = 2`, `diarize_runner: Any = None` as keyword args but `execute_variant` never forwards them, so the variants flow silently always disables diarize.

**Change**:
1. Add `diarize: bool = False`, `num_speakers: int = 2`, `diarize_runner: Any = None` to `execute_variant`'s signature (keyword-only, after `context`).
2. Forward all three to each of the three branch helpers (`transcribe_with_baseline_params`, `transcribe_with_minimal_params`, `_transcribe_with_config`).
3. In `scripts/compare_transcription_variants.py`, add `--diarize` / `--no-diarize` and `--num-speakers N` argparse flags (mirror the style in `scripts/transcribe_manager.py`). Default `--no-diarize` (matches `compare_variants.bat`'s current behavior — keep the existing default behavior unchanged).
4. Pass the parsed `args.diarize` + `args.num_speakers` into the `execute_variant(...)` call at `scripts/compare_transcription_variants.py:194`.

**Tests**: `tests/unit/test_variants.py` (lines 162, 188) tests `execute_variant`'s return format — should still pass. Add one test asserting the diarize kwarg is forwarded (mock `_transcribe_with_config` and assert `diarize=True` shows up in the call).

### T3 — Honor diarize in `transcribe_and_save` + `__main__` (I4)

**File**: `backend/transcribe.py`.

**Current state**:
- `transcribe_and_save` (line 865) signature lacks `diarize`/`num_speakers`/`diarize_runner`. It delegates to `transcribe_to_text` / `transcribe_to_json`, which also lack those params (they're at lines 824 / 839 — check signatures during implementation).
- `__main__` block (line 937) hardcodes `transcribe_to_text(...)` with no diarize.

**Change**:
1. Walk the call chain `transcribe_and_save` → `transcribe_to_text` / `transcribe_to_json` → `transcribe`. Add `diarize: bool = False`, `num_speakers: int = 2`, `diarize_runner: Any = None` (keyword-only) at every layer that currently strips them. Forward through.
2. `__main__`: leave the hardcoded dev-shim behavior as-is, but accept `STT_DIARIZE` + `STT_NUM_SPEAKERS` env vars (the Windows .bat layer already exports these — see `scripts/windows/_runtime.bat`). Pattern: `diarize = os.getenv("STT_DIARIZE", "0") == "1"`, `num_speakers = int(os.getenv("STT_NUM_SPEAKERS", "2"))`. Pass into `transcribe_to_text`. This is a dev convenience, not a real CLI entry point — `scripts/transcribe_manager.py` is the real one.

**Tests**: `tests/unit/test_transcribe.py` already covers `transcribe_and_save`; add one test asserting `diarize=True` flows from `transcribe_and_save` into the underlying `transcribe_fn` mock. The `__main__` shim is not covered (and shouldn't be).

### T4 — Fix 17 pyright errors in `tests/unit/test_transcribe.py`

**File**: `tests/unit/test_transcribe.py`.

**Categories** (from `.venv/bin/python -m pyright tests/unit/test_transcribe.py`):
- ~13 errors: `FakePreprocessResult` (line 103) is structurally compatible with `PreprocessResult` but not declared as a subclass, so passing a `lambda path, cfg: FakePreprocessResult(...)` to `preprocess_runner` fails the type check. Fix: declare `FakePreprocessResult(PreprocessResult)` or rebuild the fake using the real dataclass.
- 2 errors at lines 466, 482: `Path` passed where `str` expected for `json_path` in `transcribe_to_json`. Fix: `str(json_path)` at the call site.
- 4 errors at lines 54, 162, 181, 198, 279, 292, 306, 317: investigate during implementation — likely related to mock typing or `transcribe_fn` signature drift.

**Approach**: read the file once, fix all 17 in one pass. Don't break the behavior the tests exercise — only the type annotations.

## Out of scope

- The cuDNN system apt swap (carry-over but user-owned destructive op).
- Windows .bat smoke tests (no automated coverage).
- The doc-pin audit (`ctranslate2==4.4.0` drift on other surfaces beyond CUDA-extras D4) — separate scope, will re-list as carry-over after this plan ships.

## Test plan

After all four tasks land:
- `make pyright` clean (verifies T4 + no regressions).
- `make unit` clean (verifies T1/T2/T3 behavior).
- Quick integration spot-check is optional; this is a parameter-plumbing patch with no model-side behavior change.

## Stage / sizing

Single stage. 4 tasks, 4 files, ~30-60 LOC net diff. Estimated 1 session.
