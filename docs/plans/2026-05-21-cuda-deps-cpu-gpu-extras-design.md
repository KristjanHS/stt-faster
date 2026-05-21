# CUDA dependencies — CPU/GPU split via uv extras

**Status**: Design — ready for implementation
**Date**: 2026-05-21
**Owner**: Kristjan
**Supersedes**: none
**Related**: `2026-05-21-portable-windows-setup-design.md` (shares `_runtime.bat` infrastructure),
`docs/diarization_setup.md` (HF_TOKEN setup that pairs with this)

## 1. Problem

After the pyannote.audio diarization subsystem (commits `90912d6` → `0515257`) shipped,
every Linux install — including CPU-only Docker prod images and CI runners — pulls
~3–5 GB of CUDA libraries transitively:

`nvidia-cublas`, `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime`,
`nvidia-cudnn-cu13`, `nvidia-cufft`, `nvidia-cufile`, `nvidia-curand`,
`nvidia-cusolver`, `nvidia-cusparse`, `nvidia-cusparselt-cu13`, `nvidia-nccl-cu13`,
`nvidia-nvjitlink`, `nvidia-nvshmem-cu13`, `nvidia-nvtx`, `cuda-bindings`,
`cuda-pathfinder`, `cuda-toolkit`, `triton`.

None of these are required for the diarization solution to run — `pyannote_runner.py`
treats CUDA as optional (`torch.cuda.is_available()` check, line 38) and the project
already has a documented GPU→CPU fallback pattern.

Root cause: `pyproject.toml` declares a `pytorch-cpu` index with `explicit = true`,
but has no `[tool.uv.sources]` binding for `torch` / `torchaudio`. With `explicit =
true` and no source binding, the index is dead — torch resolves from PyPI default,
which lists all the CUDA libs as required deps on `sys_platform == 'linux'`. The
`--no-emit-package torch` flag in `make export-reqs` only hides the `torch` line in
the exported `requirements.txt`; transitive CUDA deps remain.

## 2. Goals

- **Default `uv sync` produces a CPU-only install** with no `nvidia-*` / `cuda-*` /
  `triton` packages.
- **GPU users opt in once per machine**; subsequent syncs auto-pick GPU without
  retyping flags.
- **Single `uv.lock`** — no parallel lockfiles to drift.
- **Docker images** built CPU by default; GPU image via a single build-arg.
- **CI** stays CPU; remains green; install gets faster (no ~3 GB CUDA download).
- Mechanism is **extensible** to additional GPU targets (cu121, ROCm, MPS) without
  redesign.

## 3. Non-goals

- AMD ROCm, Apple MPS, Intel XPU — extensible but not built.
- Multiple CUDA versions (cu121, cu124 alongside cu130) — extensible but not built.
- Auto-detect GPU hardware at install time (rejected during ideation — silent
  miscalibration risk outweighs UX win).
- Runtime variant switching inside a single venv — switching requires re-sync.
- GPU runners in CI.

## 4. Approach (chosen)

**uv conflicting extras + persistent local variant file.**

Both `cpu` and `cu130` extras are defined and mutually exclusive via
`[tool.uv.conflicts]`. `[tool.uv.sources]` binds `torch` and `torchaudio` to the
appropriate index per extra. `torch` and `torchaudio` move **out of base
dependencies** into both extras — this is the only way to prevent transitive
torch resolution (via pyannote.audio) from pulling CUDA wheels when no extra is
active.

A gitignored `.stt-variant.local` file at repo root holds the per-machine choice
(`cpu` or `cu130`). The existing wrapper infrastructure (`run_uv.sh`,
`_runtime.bat`, Make targets) reads this file via a small helper and always passes
the corresponding `--extra` flag to `uv sync`. Bare `uv sync` becomes a power-user
path (documented).

### 4.1 Rejected alternatives

| Approach | Why rejected |
|---|---|
| Two parallel venvs (`.venv` + `.venv-gpu`), two lockfiles | Maintenance overhead (two lockfiles to bump in sync for security updates); duplicates disk on machines that build both. uv extras give the same isolation without the duplication. |
| Auto-detect via `nvidia-smi` / `wmic` probe | Silent miscalibration: can't test CPU path on a GPU box, CI runner GPU passthrough misclassifies, Docker base-image variation drifts. Failures are quiet (wrong wheel). |
| Env-var-only (`STT_VARIANT=gpu uv sync`) | Doesn't persist across shell sessions without `.bashrc` edits; harder to discover from `git status`. |

## 5. Design

### 5.1 `pyproject.toml`

> **Baseline updated 2026-05-21**: post the community-1 migration (commit `3e0af69`),
> the snippet below was rewritten against `pyannote.audio==4.0.4` + `torch/torchaudio>=2.8.0,<3.0.0`.
> The pre-migration values (`pyannote.audio==3.4.0` + `torch/torchaudio>=2.4.0,<2.7.0`)
> are no longer the starting state.

```toml
[project]
dependencies = [
    # ... existing entries (ctranslate2, faster-whisper, etc) ...
    "pyannote.audio==4.0.4",
    # torch + torchaudio moved to extras — see [project.optional-dependencies]
]

[project.optional-dependencies]
cpu = [
    "torch>=2.8.0,<3.0.0",
    "torchaudio>=2.8.0,<3.0.0",
]
cu130 = [
    "torch>=2.8.0,<3.0.0",
    "torchaudio>=2.8.0,<3.0.0",
]

[tool.uv]
conflicts = [
    [ { extra = "cpu" }, { extra = "cu130" } ],
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu",   extra = "cpu"   },
    { index = "pytorch-cu130", extra = "cu130" },
]
torchaudio = [
    { index = "pytorch-cpu",   extra = "cpu"   },
    { index = "pytorch-cu130", extra = "cu130" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url  = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url  = "https://download.pytorch.org/whl/cu130"
explicit = true
```

The `torchaudio<2.7` upper bound remains as documented today (pyannote 3.4 imports
`torchaudio.AudioMetaData`, removed in 2.7).

### 5.2 Variant file: `.stt-variant.local`

- Location: repo root.
- Format: single line — either `cpu` or `cu130`.
- Gitignored. Owned by the user (per machine), not by the repo.
- Absent → treated as `cpu`.

### 5.3 Resolver helper: `scripts/select_variant.sh` and `scripts/select_variant.bat`

Reads `.stt-variant.local` if present, validates against `{cpu, cu130}`, emits the
value on stdout, exits non-zero with a clear message on unknown value:

```
unknown variant 'cuda' in .stt-variant.local — expected 'cpu' or 'cu130'
```

### 5.4 Wrapper integration

**`run_uv.sh`** — currently runs `uv sync`. New behaviour:

```sh
VARIANT="$(scripts/select_variant.sh)"
echo "[stt-faster] variant=${VARIANT} (set via 'make use-cpu' / 'make use-gpu' to switch)"
uv sync --extra "$VARIANT" "$@"
```

**`_runtime.bat`** (portable Windows runtime probe — shipped in commits `ec5c929`
+ `ebb9015`) — same pattern via the `.bat` resolver. Adds a new pre-sync step;
does not undo the existing bat-rewire work.

### 5.5 Makefile targets

```make
sync:        ; ./run_uv.sh
use-cpu:     ; echo cpu   > .stt-variant.local && $(MAKE) sync
use-gpu:     ; echo cu130 > .stt-variant.local && $(MAKE) sync
show-variant:; @cat .stt-variant.local 2>/dev/null || echo "(default: cpu)"
```

### 5.6 Docker

```dockerfile
ARG STT_VARIANT=cpu
RUN echo "$STT_VARIANT" > .stt-variant.local && ./run_uv.sh
```

Prod image builds CPU-only by default. A GPU image is produced via
`docker build --build-arg STT_VARIANT=cu130`. No runtime detection — the choice is
baked into the image.

### 5.7 `make export-reqs` split

```make
export-reqs: export-reqs-cpu export-reqs-cu130

export-reqs-cpu:
    uv export --no-hashes --group test --locked --no-emit-project \
              --extra cpu --format requirements-txt > requirements.txt

export-reqs-cu130:
    uv export --no-hashes --group test --locked --no-emit-project \
              --extra cu130 --format requirements-txt > requirements-gpu.txt
```

- `--no-emit-package torch` is **dropped** — the CPU export needs no special
  exclusion (CPU wheels have no nvidia transitive deps).
- `requirements.txt` is the canonical CPU export (matches default install).
- `requirements-gpu.txt` is the GPU sidecar, generated alongside.

### 5.8 CI

`.github/workflows/*.yml` `uv sync` invocations pass `--extra cpu` explicitly. CI
does not rely on `.stt-variant.local`. A new lightweight assertion test verifies
the CPU runner cannot import `nvidia_cublas` — catches accidental regressions to
PyPI-default torch.

### 5.9 `pip-audit`

`make pip-audit` continues to audit `requirements.txt` (CPU). An additional
`pip-audit-gpu` target audits `requirements-gpu.txt`. Both wired into the
pre-push hook chain.

### 5.10 `pytest` `gpu` marker

The existing `gpu` marker keeps its meaning ("requires GPU hardware"). Tests with
the marker are filtered out by default. A collection-time skip with a clear
reason is added when `torch.cuda.is_available()` is False, so importing GPU
tests on a CPU venv produces a readable skip rather than a cryptic import error.

### 5.11 `model_loader` GPU→CPU fallback

Unchanged. The extras decide *what's installed*; the runtime fallback handles the
orthogonal case where CUDA is installed but unavailable at runtime (no `--gpus`
flag in docker, driver mismatch, etc).

## 6. Rollout

| Stage | Scope | Verification |
|---|---|---|
| **D1: deps mechanism** | pyproject.toml extras + sources + conflicts; regen `uv.lock`; `.gitignore` adds `.stt-variant.local`; `scripts/select_variant.{sh,bat}` | `uv sync --extra cpu` lockfile inspection: zero `nvidia-*` / `cuda-*` / `triton`. `uv sync --extra cu130` lockfile inspection: contains them. Disk-footprint check (CPU venv should be <2 GB, GPU venv 5–7 GB). |
| **D2: wrappers** | `run_uv.sh`, `_runtime.bat`, Makefile targets (`use-cpu`/`use-gpu`/`show-variant`/`sync`) | `make use-cpu && make sync` on a clean venv → CPU install. `make use-gpu && make sync` on the GPU box → CUDA install. |
| **D3: CI + Docker** | `.github/workflows/*.yml` add `--extra cpu`; Dockerfile gains `ARG STT_VARIANT=cpu`; `make export-reqs` splits CPU/GPU; new `pip-audit-gpu` target | Green CI on a sample PR. Docker build with default arg succeeds; image size drops by ~3–5 GB versus current. |
| **D4: docs + cleanup** | Update `pyproject.toml:9` comment ("GPU-verified stack"), `README.md` install section, `docs/Transcription_solution.md` GPU note, `docs/diarization_setup.md`. Remove the dead `# for GPU: torch = { index = "pytorch-cu130"}` comment now that it's real. | `grep -r 'pytorch-cu130\|GPU-verified\|install torch separately' docs/ README.md pyproject.toml Makefile` returns only intentional references. |

Each stage is a single PR. D1 is the high-risk stage and lands first on a
throwaway-tested branch.

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|:---:|:---:|---|
| uv `conflicts` + per-extra `sources` resolution misbehaves with transitive torch (via pyannote) | Med | High | D1 throwaway-branch validation: regen `uv.lock`, diff resolved torch source per extra, confirm CPU branch has no `nvidia-*`. Fallback: platform-marker pattern (`sys_platform`-gated indices). |
| `torchaudio>=2.4,<2.7` CPU wheel not on pytorch-cpu index for some Python 3.12 patch version | Low | Med | Browse `https://download.pytorch.org/whl/cpu/torchaudio/` during D1. If gap exists, pin a known-good version pair instead of a range. |
| pyannote / speechbrain / torch-audiomentations import CUDA symbols at import time on CPU | Low | High | D1 e2e run on CPU venv. Already mitigated by `pyannote_runner.py`'s lazy-import pattern. |
| CI silently regresses to PyPI default (forgot to pass `--extra cpu`) | Med | Med | New unit assertion: `nvidia_cublas` must not be importable on the CPU CI runner. |
| User edits `.stt-variant.local` to a typo (e.g. `cuda`) and gets confusing errors | Med | Low | `select_variant.{sh,bat}` validates and emits a clear error. |
| GPU box gets `.stt-variant.local` committed accidentally | Low | Low | `.gitignore` entry; `make use-gpu` writes to the gitignored path only. |

## 8. Decisions log

1. **Selection mechanism**: uv extras + `[tool.uv.conflicts]`. *Rejected:* separate
   venvs, env-var-only, hardware auto-detect.
2. **Default behavior**: wrapper layer passes `--extra cpu` when no
   `.stt-variant.local` is set. Bare `uv sync` is a power-user path, documented.
3. **Lockfile model**: single `uv.lock` with both resolutions (uv's native
   conflicts behaviour). *Rejected:* parallel lockfiles per variant.
4. **Persistence**: gitignored `.stt-variant.local`, per-machine. One-time
   `make use-gpu` on the GPU host. *Rejected:* env-var-only (doesn't persist
   across shells without profile edits), auto-detection (silent miscalibration).
5. **GPU target**: `cu130` only for now. Extras structure leaves room for more
   without redesign.
6. **Requirements export**: two files — `requirements.txt` (CPU, canonical) +
   `requirements-gpu.txt` (GPU sidecar). Both audited by `pip-audit`.
7. **`torch` and `torchaudio` move out of base deps** into both `cpu` and `cu130`
   extras. *Rationale:* the only way to keep base `uv sync` from pulling
   transitive CUDA via pyannote.
8. **Bare `uv sync` (no extra) is undefined behaviour** for casual use. Users hit
   it only via wrappers (which always set an extra). Direct users get a CUDA-y
   resolution from PyPI default, same as today — and we document that in the
   contributor guide.

## 8a. Corrections during D1 implementation (2026-05-21)

D1 landed with two corrections to the design as originally written. Both surfaced
during the throwaway-branch validation (§7 risk row 1's mitigation) — exactly
the kind of catch that gate exists for.

1. **GPU target: `cu128` → `cu126`.** The `pytorch-cu128` index publishes torch
   only from 2.7.0 upward; combined with the load-bearing `torchaudio<2.7`
   constraint (pyannote 3.4 imports `torchaudio.AudioMetaData`, removed in 2.7),
   `cu128` was unsatisfiable. `cu126` is the only modern CUDA index that
   publishes a torch wheel in the `>=2.4,<2.7` range (torch 2.6.0+cu126). cu126
   is also more driver-backward-compatible than cu128, so this is a strict
   improvement for GPU users. All references in this doc, scripts, and
   pyproject have been updated.
2. **Base `sympy==1.14.0` → `sympy>=1.13.1,<2.0`.** Torch 2.6.0 (both `+cpu`
   and `+cu126` wheels) pins `sympy==1.13.1`. The old base pin came from the
   PyPI-default torch 2.4.1 resolution that the design is replacing. Range
   allows uv to pick 1.13.1 under either extra without freezing the no-extra
   PyPI path on 1.14.0.

Validation outcome with these corrections:
- `rm -rf .venv && uv sync --extra cpu`: 1.7 GB total (design predicted <2 GB).
  Zero `nvidia-*` / `cuda-*` / `triton` packages.
- 149/149 unit tests pass against `torch==2.6.0+cpu`.
- `uv.lock` lists `torch==2.6.0+cu126` from `https://download.pytorch.org/whl/cu126`
  for the GPU extra (full GPU install deferred — no GPU hardware in this session;
  D2 wrapper validation on the GPU box will exercise it).

## 8b. GPU target bumped cu126 → cu130 (2026-05-21)

Subsequent to §8a, the GPU extra was bumped to `cu130` with `torch>=2.11.0,<3.0.0`
(and matching torchaudio bound). Rationale:

- The `torchaudio<2.7` constraint from §8a was retired by the pyannote
  community-1 migration (now on `pyannote.audio==4.0.4`, tensor-input API), so
  the old "stuck in `>=2.4,<2.7`" reasoning no longer applies.
- `download.pytorch.org/whl/cu130` publishes `torch==2.11.0+cu130` and
  `torchaudio==2.11.0+cu130` for cp312 linux + win. No darwin (CUDA never on
  macOS anyway).
- Newer CUDA toolkit on GPU hosts is now the common case; cu130 keeps us aligned
  with what fresh driver installs ship.
- Mechanism is unchanged from §8a — extras, conflicts, indices, and variant
  selector all swing as a single edit. CPU extra remains `>=2.8.0,<3.0.0`
  (no reason to gate cpu users on cu130's torch floor).

## 8c. Corrections during D3 implementation (2026-05-21)

D3 landed with two intentional deviations from the design as originally
written. Recording them here so anyone returning to §5 doesn't write the
pre-shipped form into a new file:

1. **Dockerfiles use `--extra ${STT_VARIANT}` directly, not the
   `.stt-variant.local`-via-`run_uv.sh` indirection.** §5.6 prescribed
   `RUN echo "$STT_VARIANT" > .stt-variant.local && ./run_uv.sh`. Both
   `Dockerfile` and `docker/app.Dockerfile` instead declare
   `ARG STT_VARIANT=cpu` and pass `--extra "${STT_VARIANT}"` to every
   `uv sync` call. The two forms are functionally equivalent because
   `run_uv.sh` is itself just a thin wrapper around
   `uv sync --extra "$VARIANT"`. The direct form is cleaner inside a
   Docker layer: no temp variant file shipped into the image, no
   `run_uv.sh` copy step, `--locked` stays explicit. If a third
   Dockerfile is ever added, follow the shipped form.
2. **CPU-isolation canary uses `importlib.metadata.distribution`, not
   `import nvidia_cublas`.** §5.8 specified asserting that
   `nvidia_cublas` "must not be importable". On cu13 wheels there is
   no importable `nvidia.cublas` Python module — the wheel ships only
   native `.so` files under `nvidia/cu13/lib/`. A `find_spec`-based
   check would silently always pass and be useless as a regression
   canary. `tests/unit/test_no_transitive_cuda.py` instead probes
   `importlib.metadata.distribution("nvidia-cublas")`, which uses the
   wheel's installed dist-info — the only signal that survives the
   cu12 → cu13 wheel-layout change.

## 9. Out of scope (with explicit hooks for later)

- **AMD ROCm**: add a `rocm` extra and `pytorch-rocm` index entry. No design change.
- **Apple MPS**: torch already exposes MPS via the standard CPU wheel on macOS;
  no extra needed. Just verify the CPU extra resolves correctly on darwin.
- **Multiple CUDA targets** (`cu121`, `cu124`): add more extras to the conflicts
  list and corresponding `[tool.uv.sources]` entries. Lockfile grows; mechanism
  unchanged.
- **Auto-detect at install**: if reconsidered later, layer it on top — the helper
  script becomes `select_variant_or_probe.sh` and writes to `.stt-variant.local`.
- **GPU CI runners**: add a workflow that syncs with `--extra cu130` on a GPU
  runner, runs the `gpu`-marked tests. No design change.
