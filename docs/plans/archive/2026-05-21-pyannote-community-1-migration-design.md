# Pyannote community-1 migration — design

**Date**: 2026-05-21
**Status**: Design — ready for implementation review
**Driver**: Quality / upgrade path off the now-legacy `pyannote/speaker-diarization-3.1`
**Supersedes**: dep-pin rationale in `pyproject.toml:37-44` (rewrite as part of this work)

## 1. Context

The current diarization stack pins `pyannote.audio==3.4.0` with `torchaudio>=2.2.0,<2.7.0` because:
- pyannote 3.4 imports `torchaudio.AudioMetaData`, removed in torchaudio 2.7.
- pyannote 4.x avoids that but declares `torchcodec>=0.7.0`, which needs FFmpeg ≥6 shared libraries at runtime — historically a portability hazard across Win + WSL + Docker.

Diarization currently uses model `pyannote/speaker-diarization-3.1`. On 2025-09-29 pyannote published `pyannote/speaker-diarization-community-1` as the open successor and tagged 3.1 as **legacy**. There is no `speaker-diarization-4.x` model; the version bump on pyannote.audio (3.x → 4.x) is a library refactor, separate from model versioning. `precision-2` exists but is paid/enterprise and out of scope.

### Quality gains, community-1 vs 3.1 (DER% from model card)

| Benchmark | 3.1 | community-1 | Δ rel |
|-----------|----:|------------:|------:|
| AliMeeting (ch1) | 24.5 | 20.3 | **−17%** |
| AMI (IHM) | 18.8 | 17.0 | −10% |
| MSDWild | 25.4 | 22.8 | −10% |
| DIHARD 3 | 21.4 | 20.2 | −6% |
| AISHELL-4 | 12.2 | 11.7 | −4% |
| VoxConverse | 11.2 | 11.2 | 0% |

No Estonian benchmark exists; the closest analogues for our two-speaker dialogue workload are AMI and AliMeeting, both of which show consistent gains. We accept the model-card numbers and validate qualitatively in real use post-merge.

## 2. Decision

**Adopt `pyannote/speaker-diarization-community-1` on `pyannote.audio>=4.0.4` using the tensor-input API, decoding audio in-process with PyAV.** This sidesteps torchcodec entirely.

### Why this works

Verified at `pyannote/pyannote-audio@4.0.4`, file `src/pyannote/audio/core/io.py`:

- Lines 42-52: `torchcodec` is imported inside `try/except Exception`. Failure emits a `warnings.warn(...)` and does **not** re-raise — pyannote.audio imports successfully without torchcodec.
- Line 49 of the warning text is the upstream-documented escape hatch: *"use audio preloaded in-memory as a {'waveform': (channel, time) torch.Tensor, 'sample_rate': int} dictionary"*.
- Lines 313-317 (`Audio.__call__` tensor branch) and 268-271 (`Audio.get_duration` tensor branch) route via `torchaudio.functional.resample` only — zero torchcodec references on the tensor path.
- The `AudioSamples` / `AudioStreamMetadata` references at lines 273, 320 are local-variable type annotations only; Python does not evaluate these at runtime.

This is not a shim or monkey-patch — it is the documented fallback API. PyAV (`av==16.0.1`, already in deps) bundles its own FFmpeg and works portably on Win + WSL + Docker, so audio decode is decoupled from the system's FFmpeg version.

`torchcodec>=0.7.0` will still be pulled as a transitive dep of pyannote.audio 4.x. The wheel installs without FFmpeg present; only its runtime-decode path needs FFmpeg shared libs, and our flow never reaches that path. Windows wheel availability for torchcodec 0.7+ must be confirmed during implementation (see §6).

## 3. Options considered (record for posterity)

| Option | Summary | Verdict |
|--------|---------|--------:|
| **A** Per-target FFmpeg-6 | Bump Debian base to trixie; require Ubuntu 24.04 on WSL; bundle ffmpeg.exe via `setup.bat` (~80-150 MB extra) | Fallback if B fails on Windows |
| **B** PyAV tensor input | Documented escape hatch; PyAV already in deps; no system FFmpeg dependency | **Chosen** |
| C  Isolated diarize runtime | Separate venv/sidecar with own deps | Overkill — B is simpler |
| D  Stay on 3.4, hand-build pipeline for community-1 | Re-implement pyannote 4.x clustering | High risk, low payoff |
| E  Switch diarizer entirely (NeMo, Reverb) | Escape pyannote treadmill | Deferred — revisit if community-1 disappoints |

## 4. Implementation plan

Single PR, ~6 files. No staging needed.

### 4.1 Dependencies (`pyproject.toml`, `requirements.txt`)

- `pyannote.audio==3.4.0` → `pyannote.audio==4.0.4` (match project's exact-pin style)
- `torchaudio>=2.2.0,<2.7.0` → `torchaudio>=2.8.0,<3.0.0` (pyannote 4.0.4 requires ≥2.8.0)
- Keep `av==16.0.1` — now load-bearing for diarization audio decode (was preprocessing-only)
- Rewrite comment block to explain tensor-input strategy; remove the FFmpeg-6 fear note
- Regenerate `requirements.txt` via `uv export` per existing recipe

### 4.2 Code (`backend/diarize/pyannote_runner.py`)

- `PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"` → `"pyannote/speaker-diarization-community-1"`
- Add private `_load_audio_tensor(path: str) -> tuple[Tensor, int]` using PyAV to decode → mono → return `(waveform, sample_rate)`. PyAV is already vendored; no new dep.
- Line 90: `pipeline(audio_path, num_speakers=num_speakers)` → `pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)`
- Update docstring (line 49) and error messages to reference community-1.
- Optional: filter the torchcodec import warning once at module load (it will fire on first import even with tensor input, because the import sits at module top in pyannote's io.py).

### 4.3 Prefetch (`scripts/prefetch_models.py`)

- `PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"` → `"pyannote/speaker-diarization-community-1"` (line 27)
- Update the cross-ref comment (lines 44-50) to drop the "no revision pin / lock SHA via 3.4.0" rationale — under 4.0.4 the same logic applies but to a different snapshot.

### 4.4 Setup doc (`docs/diarization_setup.md`)

- Replace `pyannote/speaker-diarization-3.1` references (4 sites at lines 3, 33, 96, 132).
- **Critical**: community-1 requires accepting a separate gated-model license; existing 3.1 acceptance does not carry over. Add explicit step "go to https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the contact-sharing terms".
- Note that HF_TOKEN-rejection (403) symptoms map to "need to accept community-1 terms, not 3.1".

### 4.5 Tests (`tests/unit/test_diarize_pyannote_runner.py`)

- Update model-name string assertion at line 66.
- Add a unit test for `_load_audio_tensor`: feed a small fixture wav, assert shape `(channel, time)`, dtype float32, sample_rate matches.
- Existing tests that inject a fake `runner` continue to pass unchanged.

### 4.6 Integration smoke

Run an existing real Estonian sample through the updated pipeline locally (requires HF_TOKEN + community-1 license acceptance). Confirm:
- No torchcodec runtime errors
- 2 speakers detected
- Output `.txt` looks qualitatively reasonable

## 5. Risks + rollback

### Risks

1. **torchcodec lacks a Windows wheel for Python 3.12** — would block `uv sync` on the portable Windows install. **Mitigation**: verify in §6 before merging. If true, fall back to Option A (bundle FFmpeg in `setup.bat`) and uninstall torchcodec via `--no-deps` + manual transitive install — messier but recoverable.
2. **community-1 quality regresses on Estonian** — model card has no Estonian numbers; we are extrapolating from AMI/AliMeeting. **Mitigation**: smoke-test on a real Estonian sample pre-merge (§4.6); if visibly worse, revert before tagging a release.
3. **torchcodec import warning is noisy** — fires once per process at module load. **Mitigation**: filter via `warnings.filterwarnings` in `pyannote_runner.py` before the pyannote import.
4. **HF account already accepted 3.1 but not community-1** — users get a confusing 403 after upgrading. **Mitigation**: error message in `pyannote_runner.py` includes the exact community-1 URL.

### Rollback

Revert is one commit:
- `pyproject.toml` (deps + comment) ← previous
- `backend/diarize/pyannote_runner.py` ← previous
- `scripts/prefetch_models.py` ← previous
- `docs/diarization_setup.md` ← previous

No data migration; HF cache holds both model snapshots side-by-side.

## 6. Pre-implementation verification

Two questions to resolve before opening the PR:

1. **Does `torchcodec>=0.7.0` have a Windows wheel for Python 3.12?** Run on the portable Windows env: `uv pip install --dry-run torchcodec>=0.7.0`. If it tries to build from source, this is a blocker for Option B and triggers Option A fallback.
2. **Does HF_TOKEN with community-1 license accepted actually authenticate?** One-off: accept the license, then run `huggingface-cli download pyannote/speaker-diarization-community-1` with the token.

If both pass: proceed with §4. If (1) fails: switch to Option A (bundle FFmpeg in Windows setup, bump Docker base, document WSL Ubuntu 24.04 requirement). If (2) fails: investigate before any code change.

## 7. Out of scope

- `pyannote/speaker-diarization-precision-2` (paid/enterprise)
- NeMo Sortformer, Reverb, speechbrain, diart (alternative diarizers — deferred)
- Estonian-labeled DER benchmark spike (intentionally skipped per decision in `/mybrain` session 2026-05-21: model-card gains across multiple diarization corpora are sufficient evidence)
- Option A's Windows FFmpeg bundling work (only built if §6.1 forces fallback)

## 8. Decision log

- **2026-05-21**: Quality gains across AliMeeting/AMI/MSDWild benchmarks accepted as sufficient evidence; Estonian-specific spike skipped to save ~1h. Risk accepted: post-merge qualitative validation in real use.
- **2026-05-21**: Option B (PyAV tensor input) chosen over Option A (per-target FFmpeg-6) after `pyannote/pyannote-audio@4.0.4/src/pyannote/audio/core/io.py` confirmed torchcodec is a soft import and the tensor-input API is upstream-documented.
- **2026-05-21**: Single-PR scope chosen over phased rollout — change touches 6 files with no inter-file ordering hazard and a one-commit revert. Recorded per `.claude/rules/plan-hygiene.md` write-gate (≤6 files, 1 session).
