# Keyless, offline speaker diarization

**Status:** ready — Stage 1 next.

"Identify speakers" should need no Hugging Face account or token. Model weights are fetched once at install/extras time from an ungated, revision-pinned mirror, and at transcription time they load from a local dir only.

## Rulings (owner, 2026-09-26 — verbatim)

- Goal: "Identify speakers" works with no Hugging Face account/token; the app works fully OFFLINE after installation (owner emphasis — model download happens at install/extras time, never at transcription time).
- Backend: keep pyannote `speaker-diarization-community-1` (pyannote.audio==4.0.4), loaded via `Pipeline.from_pretrained(<local dir>)` — pyannote officially supports local-dir loading with no token.
- Weights source: the existing ungated mirror `pyannote-community/speaker-diarization-community-1`, pinned to revision `8a527374977391da736e0daaef26855d949d9685` (files: config.yaml, segmentation/pytorch_model.bin, embedding/{README.md,pytorch_model.bin}, plda/{README.md,plda.npz,xvec_transform.npz}; config.yaml references `$model/segmentation` etc. relative to the dir). Licence CC-BY-4.0 → ship attribution.
- Out of scope: sherpa-onnx, pyannoteAI precision-2 cloud tier (app must stay offline), removing torch.
- Open hedge to carry as `[AUDIT]` (not blocking code): the plda/ weights' licence is unstated in its README (BUT Speech@FIT VBx); owner to confirm before a public release.

## Pinned constants (checked against the HF tree API at the pinned revision, 2026-09-26)

| Name | Value |
|---|---|
| `DIARIZATION_REPO` | `pyannote-community/speaker-diarization-community-1` |
| `DIARIZATION_REVISION` | `8a527374977391da736e0daaef26855d949d9685` |
| `DIARIZATION_MODEL_DIR_ENV` | `STT_DIARIZATION_MODEL_DIR` |
| `DIARIZATION_INCLUDE` | `config.yaml`, `README.md`, `segmentation/*`, `embedding/*`, `plda/*` (drops the 861 KB `diarization.gif`; about 33 MB total) |
| sha256 `segmentation/pytorch_model.bin` (5 906 507 B) | `7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e` |
| sha256 `embedding/pytorch_model.bin` (26 646 242 B) | `6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929` |
| sha256 `plda/plda.npz` (133 852 B) | `9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255` |
| sha256 `plda/xvec_transform.npz` (134 376 B) | `325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f` |

The pinned `config.yaml` (444 B) was checked: `$model/segmentation`, `$model/embedding`, `$model/plda`, `VBxClustering`, `dependencies: pyannote.audio: 4.0.0`.

Bandit B615 only accepts a literal 40-hex revision **inside** each `snapshot_download()` call, not a constant (see the comment at `scripts/prefetch_models.py:45-50`). So every call spells out the SHA with `# pragma: allowlist secret`, and a unit test asserts that each literal equals `DIARIZATION_REVISION`.

## Refactor-first assessment: skip, the seam is already clean

- `annotate(..., runner=None)` (`backend/diarize/pipeline.py:73-103`) already accepts an injected runner.
- `run_pyannote(..., env=os.environ, import_pipeline=_import_pipeline_class)` (`pyannote_runner.py:204-212`) already takes its env and its pyannote import as parameters. The tests stub `Pipeline` through `_stub_pipeline_import` (`tests/unit/test_diarize_pyannote_runner.py:31`), with no monkeypatch.
- The installer's model fetch is already a data-driven `ModelSpec` → `model_command` → `fetch_model` path (`installer/setup_gui.py:92,476,1321`).
- Stage 1 adds one more defaulted parameter (`resolve_model`). No prep refactor is needed.
- Constraint: `installer/setup_gui.py` imports only the stdlib (it ships as a frozen exe), so it cannot import `backend.*`. The pinned constants are therefore duplicated in the installer, and a drift test guards them (Stage 2).

## Stage 1: runner loads from a resolved local dir, never the network

Focus: new `backend/diarize/model.py`, `backend/diarize/pyannote_runner.py`, `backend/diarize/errors.py` (docstring).

- `model.py` holds the pinned constants above and `resolve_model_dir(env=os.environ, *, cached_snapshot=_cached_snapshot) -> Path`. Resolution order:
  1. `env["STT_DIARIZATION_MODEL_DIR"]` if set. It must contain `config.yaml`; otherwise raise `DiarizationConfigError` naming the variable.
  2. The HF-cache snapshot. `_cached_snapshot(cache_dir)` calls `snapshot_download(DIARIZATION_REPO, revision="8a52…", local_files_only=True, cache_dir=cache_dir)`, which makes no HTTP calls. `cache_dir` is taken from `env["HF_HUB_CACHE"]`, else `env["HF_HOME"]/hub`, else `None` (the library default). It is passed explicitly because huggingface_hub reads its constants at import time, and the GUI sets these env vars on the subprocess.
  3. If neither yields a dir, raise `DiarizationConfigError("Speaker model not installed — run `make diarization-model` (dev) or Transcribe → Extras → Install (Windows); see docs/diarization_setup.md")`. This message reaches the GUI banner through `_DIARIZATION_FAILURE` (`gui.py:57-59`).
  - Existence check only: `config.yaml` plus the four weight files. Hashing belongs to download time (Stage 2).
- `pyannote_runner.py`:
  - Delete `_read_hf_token` (:139-140), the token guard (:224-228), the `HfHubHTTPError` 401/403 mapping (:241-255) and `PYANNOTE_MODEL` (:43).
  - Add a parameter `resolve_model: Callable[[Mapping[str, str]], Path] = resolve_model_dir`. Load with `pipeline_cls.from_pretrained(str(resolve_model(env)))`, with no `token` argument.
  - Reword the `None`-return error (:259-262) to name the dir.
  - Update the module docstring (:1) and the `run_pyannote` docstring.
- Offline guarantee (the "HF_HUB_OFFLINE-equivalent"): the load path gets a local dir, and resolution uses `local_files_only=True`. A process-wide `HF_HUB_OFFLINE` would be read too late here, because `backend/transcribe.py:38` imports huggingface_hub first. The runtime falsifier for this is in Stage 5.
- Tests:
  - New `tests/unit/test_diarize_model.py`, covering resolution order: env dir wins; env dir without `config.yaml` → error; injected `cached_snapshot` used when env is unset; `HF_HUB_CACHE` / `HF_HOME` → `cache_dir` mapping; neither → error text names both fixes. Also a test that parses the source with `ast` and asserts every `revision=` literal equals `DIARIZATION_REVISION`.
  - `tests/unit/test_diarize_pyannote_runner.py`: replace the three HF_TOKEN tests (:40-80) with: missing model → `DiarizationConfigError`; `from_pretrained` receives the resolved path string and no `token` kwarg (the stub records the call).
  - `tests/integration/test_diarize_with_pyannote.py`: the skip condition becomes "`resolve_model_dir()` raises" instead of "HF_TOKEN unset".
- Done when: the unit tests are green, and `grep -rn "HF_TOKEN\|HUGGINGFACE_HUB_TOKEN" backend/diarize` returns nothing.

## Stage 2: pinned, tokenless, verified download (dev CLI + installer)

Focus: `backend/diarize/model.py`, `scripts/prefetch_models.py`, `Makefile`, `installer/setup_gui.py`.

- `model.py` gets `fetch_model(*, downloader=snapshot_download, cache_dir=None) -> Path`:
  - Calls `downloader(DIARIZATION_REPO, revision="8a52…", allow_patterns=DIARIZATION_INCLUDE, token=False, cache_dir=cache_dir)`. `token=False` means a stored login is never sent.
  - Then calls `verify_snapshot(path)`, which checks the sha256 of the four weight files. On a mismatch it deletes the snapshot dir and raises `DiarizationConfigError` ("corrupt download — rerun").
  - Idempotent: a cached snapshot costs no bytes, and the hashes are re-checked on every call.
- `scripts/prefetch_models.py`: `prefetch_pyannote` becomes a call to `fetch_model()`. Drop the token/401/403 block (:52-80) and the old gated SHA `3533c8…`. Add `--diarization-only`.
- `Makefile`: new target `diarization-model:` → `.venv/bin/python scripts/prefetch_models.py --diarization-only`.
- Installer (`installer/setup_gui.py`):
  - `ModelSpec` gains `revision: str = "main"`.
  - `DIARIZATION_MODEL = ModelSpec(DIARIZATION_REPO, DIARIZATION_INCLUDE, DIARIZATION_REVISION)` (:102), with a comment that drops the "gated" wording.
  - `model_command` (:476) appends `--revision <spec.revision>`.
  - `expected_model_size` (:254) builds its tree URL from `spec.revision` instead of `main`.
  - New stdlib `verify_snapshot(dir, DIARIZATION_SHA256)` is called in `fetch_diarization` (:1380) after `fetch_model`. The dir is `model_cache_dir(hub, repo)/"snapshots"/revision`, the same layout that `hf download --revision` writes.
- Tests:
  - `tests/unit/test_diarize_model.py`: `fetch_model` with an injected downloader writing fixture files (good hashes pass; bad hash → dir removed + error; `token=False` and the revision are passed).
  - `tests/unit/test_setup_gui.py`: `model_command` includes `--revision`; the size URL uses the revision; `verify_snapshot` pass/fail.
  - Drift test: installer `DIARIZATION_MODEL` fields and sha table == `backend.diarize.model` constants.
- Done when: `make diarization-model` on a clean `HF_HUB_CACHE=$(mktemp -d)` downloads about 33 MB with no `HF_TOKEN` set, and a second run downloads 0 bytes.

## Stage 3: remove the token UX, migrate existing installs, pin the GUI offline

Focus: `backend/gui.py`, `installer/setup_gui.py`.

- GUI:
  - Delete `AppPaths.token_file` (:93-95), `read_hf_token` (:158-162), the token `Entry` (:618), `env["HF_TOKEN"]` (:286-287) and `hf_token` in `run_job` (:460).
  - `save_and_install(paths, *, launch, tempdir)` drops the token. `EXTRAS_HINT` (:50-54) becomes: "Speaker identification downloads a ~33 MB model once — no account needed. Model: pyannote community-1 (CC-BY-4.0)."
  - The button label becomes "Install speaker detection".
  - `diarization_available(env, *, find_spec=..., resolve=resolve_model_dir)` returns true iff the pyannote module is present and `resolve(env)` succeeds. `env` is the **same env `build_env` hands the CLI**, so the GUI and the runtime can never disagree.
  - `build_env` sets `HF_HUB_OFFLINE=1` when the installed `hf_home` dir exists (:284 branch). This makes every hub lookup, Whisper's `snapshot_download` included, cache-only at transcription time. Dev checkouts are untouched.
- Installer:
  - Delete `HF_AUTH_ERROR`/`HF_AUTH_HELP` (:107-113), `HF_TOKEN_PROBE_URL` + `hf_token_status` (:274-288), `read_hf_token` (:667-675), `Installer.hf_token`/`token_status`, the token guard in `run()` (:1425-1429), and the `token=` parameter of `fetch_model`.
  - `fetch_diarization` becomes fetch + verify.
- Migration:
  - Every `run()` (base, repair and extras) best-effort unlinks `config_file.parent/"hf_token"` and rmtrees `hf/hub/models--pyannote--speaker-diarization-community-1` (the old gated cache).
  - Base/repair `tasks()` adds the diarization model task when `read_config(...)["extras"] == "diarization"` (the same signal `deps_command` uses at :417). Without this, an upgraded speaker install would lose its checkbox.
- Tests:
  - `tests/unit/test_gui.py`: availability via tmp snapshot dirs through the env; `build_env` has no `HF_TOKEN` and has `HF_HUB_OFFLINE` only when installed; `save_and_install` launches `--extras` with no file written.
  - `tests/unit/test_setup_gui.py`: rewrite or delete the ~36 token references (:28-29, :694-839); new tests for "extras with no token file succeeds", "migration removes token file + legacy cache", and "repair with extras=diarization schedules the model task".
- Done when: `grep -rn "hf_token\|HF_TOKEN\|HF_AUTH" backend/gui.py installer/` returns nothing, and the unit tests are green.

## Stage 4: docs, attribution, remaining token surfaces

Focus: `docs/diarization_setup.md`, `docs/Transcription_solution.md` (:12, :111), `README.md` (:26-29 privacy, :63 link text), new `NOTICE`, `scripts/transcribe_manager.py:103`, `scripts/windows/_docker_run.bat` (:29-50), `scripts/windows/_transcribe.bat` (:13-22 comments), `tests/e2e/test_production_container.py` (:12, :385-392).

- `diarization_setup.md` is rewritten around one install step (`make diarization-model` / Extras), `STT_DIARIZATION_MODEL_DIR` for air-gapped copies (copy the snapshot dir), and a troubleshooting entry for "Speaker model not installed". The 401/403 sections are removed.
- README privacy: the installer fetches the diarization model from huggingface.co (the ungated mirror); the app downloads nothing for diarization.
- `NOTICE` gives CC-BY-4.0 attribution: model name, authors (pyannote / Hervé Bredin, pyannoteAI), licence link, mirror repo + pinned revision, "unmodified". A `[AUDIT]` line records the plda/VBx licence as owner-unconfirmed. The downloaded `README.md` model card also stays beside the weights.
- The docker bat drops the HF_TOKEN diarize fail-fast, which is also stale (it says "3.1"). The container must find the model: `[AUDIT]` check whether `_docker_run.bat` mounts an HF cache. If it doesn't, bake the snapshot into the image at build time via `fetch_model()` and set `STT_DIARIZATION_MODEL_DIR` in the Dockerfile.
- The e2e skip changes to "model not resolvable".
- Done when: `grep -rln "HF_TOKEN\|hf.co/settings/tokens\|accept the licen" README.md docs/*.md scripts backend installer` lists only intentional non-diarization hits (Whisper rate-limit notes and `tests/conftest.py` are left as they are).

## Stage 5: full verify, network-off e2e, review

1. `.venv/bin/python -m pytest tests/unit tests/integration -q --no-cov 2>&1 | tail -20`, then `make pyright`, `make ruff-fix`, `make pre-commit` (Bandit B615 must stay clean).
2. Network-off e2e (Linux/WSL):
   - Setup: `unset HF_TOKEN HUGGINGFACE_HUB_TOKEN; rm -f ~/.cache/huggingface/token` (or a throwaway `HF_HOME`), `make diarization-model`, and a Whisper prefetch.
   - Run: `unshare -rn .venv/bin/python scripts/transcribe_manager.py process <dir with tests/fixtures/audio/two_speakers_10s.wav> --diarize --num-speakers 2`. `unshare -rn` gives a user+net namespace with loopback only; fall back to `docker run --network none` if userns is blocked.
   - Expected: `.txt` lines with `SPEAKER_00`/`SPEAKER_01`, and no connection errors in the log.
   - **Falsifier (must go non-zero):** the same command with `STT_DIARIZATION_MODEL_DIR=/nonexistent` must fail that file with "Speaker model not installed".
   - `[AUDIT]` RNNoise: `backend/preprocess/steps/ffmpeg_pipeline.py:20-37` downloads `sh.rnnn` at run time when it is missing. If the netns run hits it, that breaks the owner's offline ruling. Record it as a follow-up for the installer to prefetch it; it is outside this plan's code.
3. Fresh `Agent(subagent_type="code-reviewer")` over the Stage 1-4 diff. Triage findings by defect class, then commit fixes.
4. Owner-owed Windows eyeballs (ranked):
   1. Fresh install → ▸ Extras: no token field; "Install speaker detection" → the setup row "Model: speaker-diarization-community-1" reaches 100 % (about 33 MB); Transcribe reopens with ☐ Identify speakers.
   2. Upgrade an existing token install (run setup repair): `%APPDATA%\stt-faster\hf_token` and `<install>\hf\hub\models--pyannote--speaker-diarization-community-1` are gone, `models--pyannote-community--…\snapshots\8a52…` is present, and the checkbox is still shown.
   3. Airplane mode → transcribe a two-speaker file with Identify speakers → speaker labels appear and there is no "Speakers skipped" banner.
   4. Delete the `models--pyannote-community--…` folder and relaunch → the checkbox is hidden and Extras offers the install again.
   5. The Extras hint shows the CC-BY-4.0 line, and `NOTICE` is present in the installed app dir.

## Open hedges

- `[AUDIT]` plda/ licence (from the Rulings). The owner must confirm it before a public release. It does not block the code.
- `[AUDIT]` The Stage 3 `HF_HUB_OFFLINE=1` assumes the installed Whisper snapshots have `refs/main` cached, which `hf download` writes. Stage 5 item 3 on Windows proves it.
- `[AUDIT]` The Docker model path and the RNNoise run-time download (Stages 4 and 5).
