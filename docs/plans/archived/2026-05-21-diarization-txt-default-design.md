# Diarization + TXT-default + always-timestamped output — Design

**Status**: Approved (Deep mode /mybrain session, 2026-05-21).
**Type**: Single-doc design + implementation plan (per `.claude/rules/plan-hygiene.md`: ≥3 files / ≥2 sessions → single plan doc, no multi-stage split).

## §0 Goals

Produce a readable, speaker-attributed, time-anchored transcript as the default output of every day-to-day Windows launcher, without breaking the variant-comparison workflow that consumes JSON.

Three user-stated asks, bundled because they converge on the same output surface:

1. Generate only `.txt` by default (no `.json` in production workflows).
2. `.txt` output always contains timestamps.
3. Use diarization with `2` speakers as the default; user-overridable.

**Count correction (post-review)**: there are 8 bats invoking `transcribe_manager.py process`, not 10. Verified via `grep -l 'transcribe_manager.py process' scripts/windows/*.bat` (2026-05-21). 1 variant-sweep bat (`compare_variants.bat`) + 7 production transcribe bats. All sections of this doc use these counts.

## §1 Decisions log

Each numbered decision is load-bearing. Re-opening any of these requires a corresponding update to the implementation step list in §9.

1. **TXT line format**: `[mm:ss.ff --> mm:ss.ff] SPEAKER_NN: text`. One line per whisper segment. Range timestamps (not start-only, not SRT cues, not speaker-turn blocks).
2. **Diarization backend**: `pyannote.audio` with model `pyannote/speaker-diarization-3.1`. HF-gated; HF_TOKEN required.
3. **Default-flip mechanism**: bats explicitly pass all three defaults (`--output-format txt`, `--diarize`, `--num-speakers 2`). CLI defaults exist as *real* defaults — they take effect for direct CLI invocations and they would also take effect if a bat forgot to pass a flag. The bats redundantly state them for two reasons: (a) the bat is self-documenting (a user reading the bat sees the actual behaviour without consulting the CLI), (b) future CLI-default changes cannot silently change bat behaviour. So: "guardrail" = real default that takes effect when no override is passed, *not* "inert safety net". JSON code path is preserved (deleting it would break `scripts/generate_variant_report.py` and `scripts/generate_variant_diffs.py`).
4. **Segment×speaker reconciliation**: simple overlap-assign — each whisper segment gets the speaker whose pyannote turn covers the most of that segment's time. No segment splitting at speaker boundaries (future-add if needed).
5. **Deterministic speaker labelling**: `anchor_speaker_zero()` performs a **full chronological relabel** of *all* speakers — the first-appearing pyannote label becomes `SPEAKER_00`, the second-appearing becomes `SPEAKER_01`, and so on. (Not just SPEAKER_00 anchored with others retaining pyannote's labels — that would leave gaps like `{SPEAKER_00, SPEAKER_02, SPEAKER_05}` for ≥3 speakers.) Eliminates the "speakers swapped across re-runs" surprise and keeps the label set contiguous regardless of `--num-speakers` value.
6. **`--num-speakers N` overridable, default 2**. No auto-detection mode (out of scope).
7. **`--diarize / --no-diarize` flag, default on**. `--no-diarize` exists primarily for `compare_variants.bat` and debugging, where speaker labels add noise + cost.
8. **`compare_variants.bat` keeps `--output-format both` AND adds `--no-diarize`**. The variant-comparison workflow needs JSON (downstream report generators read it) and does not benefit from diarization on every variant.
9. **Architecture**: new `backend/diarize/` subsystem mirroring `backend/preprocess/` and `backend/variants/`. Not inline in `transcribe.py`; not a variant-system phase.
10. **Sequential model loading mechanism**: after whisper finishes producing segments, before instantiating the pyannote pipeline: `del model` on the whisper reference held by the runner, then `gc.collect()`, then (CUDA path only) `torch.cuda.empty_cache()`. The pyannote pipeline is created in a local scope inside `pyannote_runner.run_pyannote()` and goes out of scope at function return; the same `del` + `empty_cache()` triplet runs on exit. Local-scope GC alone is insufficient because PyTorch caches CUDA allocations until `empty_cache()` is called — without the explicit empty_cache, the VRAM bound claim does not hold. Both loads go through `model_loader.py`'s existing GPU→CPU fallback banner pattern.
11. **Re-work audit outcome (per /mybrain step 7.5, adapted to code)**: `backend/transcribe.py` is touched by exactly one step (step 4), combining the `annotate()` call site and the `transcribe_to_text()` format change. Tests for the new TXT format land in the same step (step 4), not in a separate test-only step.

12. **Bat classification rule** (resolves §5's earlier punt-row): a bat goes in the **variant-sweep profile** (`--output-format both` + `--no-diarize`) iff it currently passes `--output-format both` AND its purpose is variant comparison (consumed by `generate_variant_report.py` / `generate_variant_diffs.py`). All other bats invoking `transcribe_manager.py process` go in the **production profile** (`--output-format txt` + `--diarize` + `--num-speakers 2`). With the 2026-05-21 grep this resolves to: `compare_variants.bat` → variant-sweep; the 7 other bats → production. The rule, not the enumeration, is load-bearing — a future bat just gets classified by it.

## §2 Scope

**In**:
- `.txt` is the only file produced by 7 of 8 Windows launchers (`compare_variants.bat` excepted).
- `.txt` always contains range timestamps + speaker labels when diarization is on; timestamps only when `--no-diarize`.
- Diarization default-on; `pyannote.audio` backend; `num_speakers=2` default; `--num-speakers N` override.
- `.json` (when emitted via `--output-format both` or `--output-format json`) gains a `speaker` field per segment.
- `SPEAKER_00` deterministically anchored to the chronologically-first speaker.
- HF_TOKEN missing / model-license unaccepted → clear error pointing to `docs/diarization_setup.md`.

**Out**:
- Word-level alignment (would require WhisperX-style word timestamps).
- Splitting whisper segments at speaker boundaries.
- Auto speaker-count detection.
- Diarization as a variant-system axis (variant reports unchanged).
- New file formats (SRT, VTT).
- Removing the JSON code path or the `--output-format` flag.

## §3 User-visible behaviour

**CLI surface** (`scripts/transcribe_manager.py`):

```
.venv/bin/python scripts/transcribe_manager.py process /path \
    [--preset et-large] [--language et] \
    [--output-format txt|json|both]   # default: txt (unchanged)
    [--diarize | --no-diarize]        # NEW; default: --diarize
    [--num-speakers N]                # NEW; default: 2
    [--variant N | --variants N,M,...]
```

**TXT output (diarization on)**:

```
[00:00:00.40 --> 00:00:03.05] SPEAKER_00: Tere, kuidas läheb?
[00:00:03.10 --> 00:00:05.70] SPEAKER_01: Aitäh, hästi. Ja sinul?
[00:00:05.80 --> 00:00:09.10] SPEAKER_00: Mul on ka kõik hästi, alustame koosolekut.
```

**TXT output (diarization off, `--no-diarize`)**:

```
[00:00:00.40 --> 00:00:03.05] Tere, kuidas läheb?
[00:00:03.10 --> 00:00:05.70] Aitäh, hästi. Ja sinul?
```

Timestamps are emitted unconditionally — there is no CLI flag that turns them off. The decision lives in `transcribe_to_text()`'s loop (see §4 code snippet): timestamps always written; speaker prefix only when the segment dict has a `speaker` field. `--no-diarize` removes the `speaker` field upstream by not calling `annotate()`.

**JSON output (unchanged structure; new `speaker` field appears when diarization is on)**:

```json
{
  "audio": "filename.wav",
  "segments": [
    {"id": 0, "start": 0.40, "end": 3.05, "text": " Tere, kuidas läheb?", "speaker": "SPEAKER_00"},
    ...
  ]
}
```

When diarization is off, the `speaker` field is absent. Downstream JSON consumers that don't yet know about `speaker` are unaffected (additive field).

## §4 Architecture & data flow

### New module layout

```
backend/diarize/
├── __init__.py           # exports: annotate, SpeakerTurn, DiarizationConfigError, DiarizationRuntimeError
├── errors.py             # ~20 LOC: error classes
├── pyannote_runner.py    # ~80 LOC: load model, run inference, HF_TOKEN error mapping
└── pipeline.py           # ~120 LOC: SpeakerTurn dataclass, overlap_assign(), anchor_speaker_zero(), annotate()
```

### Key types (`backend/diarize/pipeline.py`)

```python
@dataclass(frozen=True)
class SpeakerTurn:
    start: float      # seconds
    end: float
    speaker: str      # "SPEAKER_00", "SPEAKER_01", ...

def annotate(
    segments: list[dict],
    audio_path: str,
    *,
    num_speakers: int = 2,
    runner: Callable[..., list[SpeakerTurn]] = run_pyannote,
) -> list[dict]:
    """Return segments with a 'speaker' field added via overlap-assign.
    Performs full chronological relabel (Decision §1.5).
    On zero turns, returns segments unchanged (no 'speaker' field).
    """

def overlap_assign(
    segments: list[dict],
    turns: list[SpeakerTurn],
) -> list[dict]:
    """For each segment, assign 'speaker' = the SpeakerTurn whose
    [start, end] overlaps the most of the segment's [start, end].
    Segments with zero overlap to any turn are returned without a 'speaker' field."""

def anchor_speaker_zero(
    segments: list[dict],
) -> list[dict]:
    """Renumber speaker labels by chronological appearance.
    The first segment with a 'speaker' field defines SPEAKER_00; the next
    unseen speaker label becomes SPEAKER_01; etc. Segments without a
    'speaker' field pass through unchanged."""
```

### Call sites

- `backend/transcribe.py::transcribe()` — after the segment list is built, before `_build_metrics_payload`:
  ```python
  if run_config.diarize:
      segments_payload = diarize.annotate(
          segments_payload, str(audio_path),
          num_speakers=run_config.num_speakers,
      )
  ```
- `backend/transcribe.py::transcribe_to_text()` — segment-write loop becomes:
  ```python
  for s in segments:
      ts = f"[{_fmt(s['start'])} --> {_fmt(s['end'])}]"
      speaker = f" {s['speaker']}:" if 'speaker' in s else ""
      text_file.write(f"{ts}{speaker} {s['text'].lstrip()}\n")
  ```
- `transcribe_to_json` — no code change; the new `speaker` field flows through automatically.

### Data flow (one file, one variant)

```
audio.wav
   │
   ▼
[preprocess] (existing variant pipeline)
   │
   ▼
[transcribe] → segments = [{start, end, text}, ...]   ← faster-whisper
   │
   ▼  (only if run_config.diarize)
[diarize.annotate]
   │
   ├── run_pyannote(audio_path, num_speakers=2) → [SpeakerTurn, ...]
   ├── overlap_assign(segments, turns) → segments with 'speaker' added
   └── anchor_speaker_zero(segments) → SPEAKER_00 is chronologically first
   │
   ▼
segments = [{start, end, text, speaker}, ...]
   │
   ├── transcribe_to_text   → audio.txt    [mm:ss.ff --> mm:ss.ff] SPEAKER_NN: text
   └── transcribe_to_json   → audio.json   {segments: [{..., speaker: "SPEAKER_NN"}]}
```

### Model loading discipline

Sequential, never concurrent (Decision §1.10):

1. Load whisper → run → release reference (existing).
2. Load pyannote → run → release reference (new `load_pyannote_pipeline()` helper in `model_loader.py`, mirroring `load_whisper_model()`).

Both reuse the existing GPU→CPU fallback banner pattern.

### HF token & model fetch

- `scripts/prefetch_models.py` gains `prefetch_pyannote()`. Same HF cache root.
- `pyannote_runner.py` catches `huggingface_hub.utils.HfHubHTTPError`; maps 401 → `DiarizationConfigError(missing HF_TOKEN)`, 403 → `DiarizationConfigError(model license not accepted)`. Both messages point to `docs/diarization_setup.md`.
- `setup.bat` / `_runtime.bat` (Stage 1 Windows setup) get one new line documenting that `HF_TOKEN` must be set (env var or `.env`).

## §5 Config wiring & bat changes

### CLI surface additions (`scripts/transcribe_manager.py`)

```
--diarize / --no-diarize        # action store_true/store_false, dest='diarize', default=True (guardrail)
--num-speakers N                # type=int, default=2 (guardrail)
```

### `RunConfig` additions (`backend/run_config.py`)

```python
diarize: bool = True
num_speakers: int = 2
```

Both serialised into the run-log JSONL via the existing `to_dict()` pattern (additive — no breakage).

### Bat changes — 8 files

Enumeration verified 2026-05-21 via `grep -l 'transcribe_manager.py process' scripts/windows/*.bat`. Classification per Decision §1.12.

| Bat | `--output-format` | Diarize flag | `--num-speakers` | Profile |
|-----|-------------------|--------------|------------------|---------|
| `transcribe_estonian_Online.bat` | **txt** (was `both`) | `--diarize` | `--num-speakers 2` | Production |
| `transcribe_estonian_Teams.bat` | **txt** (was `both`) | `--diarize` | `--num-speakers 2` | Production |
| `transcribe_estonian_Desk.bat` | **txt** (was `both`) | `--diarize` | `--num-speakers 2` | Production |
| `transcribe_estonian_32bit_cpu.bat` | **txt** (was `both`) | `--diarize` | `--num-speakers 2` | Production |
| `transcribe_english_Online.bat` | **txt** (was `both`) | `--diarize` | `--num-speakers 2` | Production |
| `transcribe_english_Teams.bat` | **txt** (was `both`) | `--diarize` | `--num-speakers 2` | Production |
| `transcribe_english_Desk.bat` | **txt** (was `both`) | `--diarize` | `--num-speakers 2` | Production |
| `compare_variants.bat` | **both** (unchanged) | `--no-diarize` | *(omitted — irrelevant when diarize off)* | Variant sweep |

7 production bats + 1 variant-sweep bat = 8 total. No "any other" punt row; if a new bat lands before this design ships, apply the Decision §1.12 rule and add it to the appropriate row.

**Principle**: defaults that matter live in the bats AND in the CLI defaults (the two agree, redundantly). The bats are self-documenting, and future CLI-default changes cannot silently change bat behaviour.

## §6 Testing strategy

Per `docs/testing_approach.md` and `~/.claude/rules/testing.md` value-gate: every test must justify itself; no speculative coverage.

### Unit (cheap, fast)

`tests/unit/test_diarize_pipeline.py`:

- `overlap_assign`: synthetic segments + turns → assert majority-overlap winner. 4 cases: clean alignment, mid-segment switch, segment outside any turn, segment spanning 3 turns.
- `anchor_speaker_zero`: first chronological speaker is `SPEAKER_01` in input → assert relabel to `SPEAKER_00`.
- `annotate` with zero turns: segments unchanged, no `speaker` field, no crash.

`tests/unit/test_diarize_pyannote_runner.py`:

- Mock `huggingface_hub.utils.HfHubHTTPError(401)` → raises `DiarizationConfigError` with HF_TOKEN guidance.
- Mock 403 → raises with model-accept URL.

`tests/unit/test_transcribe_processor.py` (updates):

- New TXT format with speaker: assert exact line shape `[mm:ss.ff --> mm:ss.ff] SPEAKER_NN: text\n`.
- New TXT format without speaker (back-compat): assert timestamps-only output works when `speaker` field absent.

### Integration (slow, real model, gated)

`tests/integration/test_diarize_e2e.py`:

- `@pytest.mark.integration`; skip cleanly if `HF_TOKEN` env var not set.
- Fixture: `tests/fixtures/audio/two_speakers_10s.wav` — short, license-clear, two distinct voices.
- Asserts: 2 distinct speakers found; `SPEAKER_00` is chronologically first; segments have `speaker` field; TXT output matches the documented line format.

### What we deliberately do not test

- CPU/GPU fallback path — exercised by existing `model_loader.py` coverage; the diarize runner reuses it.
- "Diarize-off produces identical output to pre-feature behaviour" snapshot — negative-coverage; the back-compat unit test above covers the format-without-speaker case.
- `--num-speakers > 2` — passthrough to pyannote; testing pyannote's behaviour is not our job. Documented but not snapshotted.

## §7 Errors, prefetch wiring, docs

### Error classes (`backend/diarize/errors.py`)

```python
class DiarizationConfigError(Exception):
    """Setup problem — HF_TOKEN missing, model license unaccepted, etc.
    Exit code 2 from CLI; message points to docs/diarization_setup.md.
    Aborts the batch (config-level, not file-level)."""

class DiarizationRuntimeError(Exception):
    """Pyannote ran but produced unusable output.
    Falls back to no-speaker .txt output and logs WARNING.
    Single-file scope; does not abort batch."""
```

`processor.py`'s per-file `try/except` routes failed files to `failed/`. `DiarizationConfigError` is a batch-level setup problem — not a per-file failure — so it must propagate *past* that try/except and abort the run. Implementation: re-raise `DiarizationConfigError` explicitly in `processor.py`'s exception handler instead of swallowing it into the `failed/` lifecycle.

### Prefetch (`scripts/prefetch_models.py`)

New `prefetch_pyannote()` function appended to the existing prefetch list. Uses `snapshot_download(repo_id="pyannote/speaker-diarization-3.1", ...)`. Same HF cache root. Errors map to `DiarizationConfigError` — so a fresh-machine setup surfaces the HF_TOKEN requirement once, up front, before any transcription runs.

### Docs

New `docs/diarization_setup.md`: (1) what diarization is, (2) one-time HF account + token setup, (3) one-time pyannote model accept, (4) verify with `python scripts/prefetch_models.py`, (5) troubleshooting (401, 403, GPU OOM, zero turns). Linked from `README.md` and from `DiarizationConfigError` message text.

Updates:
- `docs/Transcription_solution.md` — Output Formats § gets the new TXT line format and the `speaker` field.
- `README.md` — one-line mention pointing to the setup doc.
- `scripts/windows/HOW_TO_USE.txt` — one-line note that TXT is default with timestamps + speakers.

## §8 Pre-mortem & mitigations

| Failure mode | Likelihood | Impact | Mitigation (folded into design) |
|--------------|:----------:|:------:|---------------------------------|
| HF_TOKEN missing → cryptic 401 first run | High | High | `pyannote_runner.py` 401-handler → `DiarizationConfigError` with explicit guidance; also surfaced by `prefetch_pyannote()` at setup time. |
| pyannote model license unaccepted → 403 | High first-run | High | 403-handler → `DiarizationConfigError` pointing to model accept URL. |
| GPU OOM (whisper + pyannote both loaded) | Med | Med | Sequential model loading (Decision §1.10); GPU→CPU fallback via `model_loader.py`. |
| Pyannote returns zero turns (silence/music) | Low-med | Low | `annotate()` returns segments unchanged; log single WARNING; TXT emits timestamps only. |
| User forces `--num-speakers 2` on 3+ speaker call | Med | Med | Documented in CLI help + setup doc: 2 is *default*, not a *floor*. |
| pyannote API drift between minor releases | Low | Med | Pin pyannote version in `pyproject.toml`; verify against `requirements.txt` export workflow. |
| Speaker label swaps across re-runs of the same file | Med (without anchor) | Low | `anchor_speaker_zero()` post-processing — SPEAKER_00 is deterministically the chronologically-first speaker. |

## §9 Implementation step list & commit boundaries

### Step list (in order)

| # | Step | Files | Tests in same commit |
|---|------|-------|----------------------|
| 1 | Add `pyannote.audio` dep + prefetch | `pyproject.toml`, `requirements.txt` (regen via `make export-reqs`), `scripts/prefetch_models.py` | none — dep landing |
| 2 | Create `backend/diarize/` subsystem | `backend/diarize/{__init__,errors,pyannote_runner,pipeline}.py` | `tests/unit/test_diarize_pipeline.py`, `tests/unit/test_diarize_pyannote_runner.py` |
| 3 | Wire CLI + RunConfig + passthrough | `scripts/transcribe_manager.py`, `backend/run_config.py`, `backend/processor.py`, `backend/components.py` (required — `output_format` is plumbed through `components.py` extensively per existing code; `diarize`/`num_speakers` follow the same passthrough pattern) | RunConfig roundtrip + CLI argparse smoke |
| 4 | Integrate in transcribe + new TXT format (single Edit pass) | `backend/transcribe.py` | `tests/unit/test_transcribe_processor.py` updates: speaker-present + speaker-absent paths |
| 5 | Update 8 bats | 7 production bats (explicit defaults), `compare_variants.bat` (+ `--no-diarize`) | none |
| 6 | Integration e2e test | `tests/integration/test_diarize_e2e.py`, `tests/fixtures/audio/two_speakers_10s.wav` | the test itself |
| 7 | Docs | `docs/diarization_setup.md` (new), `docs/Transcription_solution.md`, `README.md`, `scripts/windows/HOW_TO_USE.txt` | none |

### Re-work audit (per /mybrain step 7.5, adapted to code)

Walked each step's concrete edits. Two collisions found and resolved (recorded as Decision §1.11):

1. `backend/transcribe.py` originally split across two steps (annotate-call + format change). Merged into step 4 — same Edit pass — to avoid two PostToolUse hook cycles dumping pyright noise mid-flight.
2. Tests for the new TXT format originally in a trailing test-only step; relocated to step 4 per "test updates land with the prod change they cover" (plan-hygiene rule, last paragraph).

No other collisions. Steps 5 and 7 touch independent surfaces; order interchangeable but docs-last is more conventional.

### Commit boundaries (Conventional Commits; primary branch `main`)

| Commit | Scope | Behaviour change |
|--------|-------|------------------|
| C1 | `feat(deps): add pyannote.audio for diarization` | None at runtime; prefetch surfaces HF_TOKEN requirement |
| C2 | `feat(diarize): pyannote-based speaker diarization subsystem` | None (subsystem unused) |
| C3 | `feat(cli): add --diarize/--no-diarize and --num-speakers flags` | None (flags wired but transcribe() does not call diarize yet) |
| C4 | `feat(transcribe): emit timestamps and speaker labels in txt output` | **First user-visible change**: TXT gains `[start --> end] SPEAKER_NN:` prefixes when `--diarize` is on; timestamps always present even when `--no-diarize` |
| C5 | `feat(scripts/windows): default to txt+diarize+num-speakers=2; preserve compare_variants both/no-diarize` | **Second user-visible change**: 7 production bats produce only `.txt` with diarization; `compare_variants.bat` keeps `both` + adds `--no-diarize` |
| C6 | `test(diarize): integration test on 2-speaker fixture` | None (test only) |
| C7 | `docs(diarize): setup guide + transcription solution updates` | None |

C4 and C5 are the two commits that flip user behaviour. Both revertable independently.

**Interim state between C4 and C5**: CLI default for `--diarize` is `True` (per Decision §1.3 — CLI defaults are real defaults). Bat invocations that don't yet pass `--diarize` therefore *do* get speakers, by inheritance from the CLI default. The TXT format gains both timestamps and speakers immediately at C4. C5 changes nothing for `--diarize`/speakers behaviour at the production bats; it only flips `--output-format` from `both` to `txt` for those 7 bats and adds the explicit flags so the bat self-documents. `compare_variants.bat` does change behaviour at C5 — it adds `--no-diarize` so the variant sweep no longer pays the diarization cost.

### File list summary

**New** (10):
- `backend/diarize/__init__.py`
- `backend/diarize/errors.py`
- `backend/diarize/pyannote_runner.py`
- `backend/diarize/pipeline.py`
- `tests/unit/test_diarize_pipeline.py`
- `tests/unit/test_diarize_pyannote_runner.py`
- `tests/integration/test_diarize_e2e.py`
- `tests/fixtures/audio/two_speakers_10s.wav`
- `docs/diarization_setup.md`
- `docs/plans/2026-05-21-diarization-txt-default-design.md` (this file)

**Modified** (~16):
- `pyproject.toml`, `requirements.txt`
- `scripts/prefetch_models.py`
- `backend/run_config.py`, `backend/processor.py`, `backend/components.py`, `backend/transcribe.py`, `backend/model_loader.py`
- `scripts/transcribe_manager.py`
- 8 `scripts/windows/*.bat` (7 production + `compare_variants.bat`)
- `tests/unit/test_transcribe_processor.py`
- `docs/Transcription_solution.md`, `README.md`, `scripts/windows/HOW_TO_USE.txt`

## §10 Open items

All decision-shaping questions resolved during the /mybrain session. Three categories:

### Resolved during design (no action needed)

- Bat enumeration: 8 bats confirmed via 2026-05-21 grep (see §5 and pre-§1 count correction note). Classification rule in Decision §1.12.

### Resolved at C1 (dependency-landing commit)

- **pyannote version pin**: must be chosen at C1. Default direction: pin to the latest stable pyannote.audio release that the model card for `pyannote/speaker-diarization-3.1` documents as compatible. C1 commit message should record the exact version pinned.
- **`HF_TOKEN` env var name**: verify against the installed pyannote version's source (`pyannote/audio/core/pipeline.py` typically reads it). If pyannote uses a different name (e.g. `HUGGINGFACE_HUB_TOKEN`), `pyannote_runner.py` should support both — read `HF_TOKEN` first, fall back to whatever pyannote reads. Document the chosen name in `docs/diarization_setup.md`.

### Resolved at C6 (integration-test commit)

- **Test fixture `two_speakers_10s.wav`**: source a 10-second 2-speaker license-clear sample. Acceptable sources: (a) record fresh with two voices, (b) extract a public-domain podcast clip, (c) use an existing fixture library (LibriSpeech, AMI, VoxConverse) and trim. License must be compatible with the project; document the source in `tests/fixtures/audio/README.md`.

No items remain that require user decision before C1 can start.
