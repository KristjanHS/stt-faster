# stt-faster

[![Latest release](https://img.shields.io/github/v/release/KristjanHS/stt-faster?sort=semver&label=latest%20release)](https://github.com/KristjanHS/stt-faster/releases/latest)
[![Release date](https://img.shields.io/github/release-date/KristjanHS/stt-faster?label=released)](https://github.com/KristjanHS/stt-faster/releases/latest)
[![Downloads](https://img.shields.io/github/downloads/KristjanHS/stt-faster/total?label=downloads)](https://github.com/KristjanHS/stt-faster/releases)

**[⬇ Download Transcribe-Setup.exe (latest, Windows)](https://github.com/KristjanHS/stt-faster/releases/latest/download/Transcribe-Setup.exe)** · [What's new](https://github.com/KristjanHS/stt-faster/releases/latest) · [All releases](https://github.com/KristjanHS/stt-faster/releases)

Turns speech recordings into text on your own computer — Estonian by default, English too.
 _(Developers: jump to [For developers](#for-developers).)_

## Windows app (no technical setup)

Download [Transcribe-Setup.exe](https://github.com/KristjanHS/stt-faster/releases/latest/download/Transcribe-Setup.exe) and double-click it — no admin rights needed. The app is not code-signed, so Windows warns twice:

1. Browser says the file *isn't commonly downloaded*: click **Keep** (Edge: **… → Keep → Show more → Keep anyway**).
2. Blue *Windows protected your PC* screen: click **More info → Run anyway**.

Open it straight from the browser. Afterwards, update or remove the app with Start ▸ Transcribe ▸ **Repair** / **Uninstall**: Smart App Control may block opening the downloaded file again.

**Privacy:** This program will not transfer any information to other networked systems unless specifically requested by the user or the person installing or operating it. Only the installer downloads, and only what it needs:

- Installer: github.com (uv, ffmpeg, the app source from the latest release), pypi.org + files.pythonhosted.org (Python packages), download.pytorch.org (torch), huggingface.co (the Whisper speech models and the speaker model from its ungated, revision-pinned mirror; no account or token) and raw.githubusercontent.com (the RNNoise noise-reduction model).
- App: no network calls while transcribing; every model loads from disk. No telemetry (pyannote's usage metrics are switched off), no update checks. Model attributions: [NOTICE](NOTICE).

## License

MIT License - see [LICENSE](LICENSE) file.

---

# For developers

_Everything below is for building, testing or running stt-faster from source — not needed to use the [Windows app](#windows-app-no-technical-setup)._

## Requirements

- uv (https://astral.sh/uv) or pip
 - Docker & Docker Compose
 - Linux/WSL2
 - Optional: Act CLI for local CI runs (https://github.com/nektos/act).

---

## Dev setup

- Create .venv and install dev/test toolchain (editable install, **CPU torch wheels by default**): `./run_uv.sh`
- GPU users: `make use-gpu` once per machine (writes `.stt-variant.local`), then `./run_uv.sh`. `make use-cpu` to switch back; `make show-variant` to check. Mechanism: uv extras `cpu` / `cu130` — see `docs/Transcription_solution.md` § "CPU / GPU install variants".
- Run pre-commit via uv (uses the venv): `uv run pre-commit run --all-files`
- Quick integration test run: `make integration-local`
- Start Docker services (if needed): `docker compose -f docker/docker-compose.yml up -d --build`

- Fallback to pip/venv:
Generate requirements.txt based on uv.lock: `make export-reqs` (writes `requirements.txt` for CPU and `requirements-gpu.txt` for the `cu130` extra).

## Command-line transcription

Batch audio transcription with Estonian (default) and English models, with speaker diarization on by default. [Technical details →](docs/Transcription_solution.md) · [Speaker model setup →](docs/diarization_setup.md)

### Quick Start

**Windows (WSL)**: Copy a batch file from `scripts/windows/` (e.g. `transcribe_estonian_Desk.bat`, `transcribe_estonian_Teams.bat`, `transcribe_english_Desk.bat`) into your audio folder and double-click. See `scripts/windows/README.md` for the full list and which variant each file targets.

**Command line**:
```bash
# Process folder (Estonian default)
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio

# Show recent runs
.venv/bin/stt-faster db recent

# Use different model
.venv/bin/python scripts/transcribe_manager.py process /path/to/audio --preset large8gb
```

**Model presets**: `et-large` (Estonian, default), `large8gb` (English/multi, best accuracy), `turbo` (fast), `distil` (fastest)

**Output**: Files moved to `processed/` subfolder with `.txt` transcripts (range timestamps + speaker labels by default). Use `--output-format both` for `.json` alongside, or `--no-diarize` to skip speaker labels. Failed files in `failed/` subfolder.

**Troubleshooting**: For WSL paths use `/mnt/c/Users/...`. Delete `~/.local/share/stt-faster/runs.jsonl` to reset run history.

### Docker Usage (No Local Installation)

Run stt-faster in Docker without installing Python or dependencies:

```bash
# Build production image (one time); fetch the speaker + RNNoise models on the host
make docker-build-prod
make diarization-model rnnoise-model

# Process audio files using Docker wrapper
./scripts/transcribe-docker process /path/to/audio --preset turbo

# Or run directly
docker run --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/hf:/home/appuser/.cache/hf \
  stt-faster:latest process /workspace/audio --preset turbo
```

The image carries no models: pass the speaker and RNNoise models as read-only mounts, see [docs/diarization_setup.md § Docker](docs/diarization_setup.md#docker).

**Features:**
- ✅ No Python installation required
- ✅ Models cached in `~/.cache/hf` (persisted across runs)
- ✅ Transcription state in `~/.local/share/stt-faster`
- ✅ Runs as non-root user with your UID/GID

**See:** [docker/README.md](docker/README.md) for full Docker documentation.

## Local CI with Act
- Act is supported for local CI parity. See `docs/AI_instructions.md` for setup and usage.

---

## AI Coding Agent Configs

- Gemini: `gemini.md` quickstart; settings in `.gemini/config.yaml` and `.gemini/settings.json` for Gemini CLI and Gemini Code Assist.
- Codex CLI: `.codex/` (use these configs in `~/.codex/`)  and `AGENTS.md` operational rules and guardrails for agents in this repo.
- Cursor: `.cursor/` and `.cursor/rules/*.mdc` to guide Cursor behavior; `.cursorignore` for noise filtering.
- Shared docs: `docs/` contains agent-focused references like `AI_instructions.md`, `CODEX_RULES.md`, and `testing_approach.md`.

---

## Checks & Automations

- Pre-commit: Ruff (lint+format), Pyright (backend), Yamlfmt, Actionlint, Hadolint, Bandit, Detect-secrets. Run: `uv run pre-commit run --all-files`. Config: `.pre-commit-config.yaml` (+ `.secrets.baseline`).
- Pre-push: Ruff format/check, Yamlfmt, Pyright, Unit tests; optional Semgrep + CodeQL via Act. Enable with `make setup-hooks`. Toggle via env: `SKIP_LINT=1 SKIP_PYRIGHT=1 SKIP_TESTS=1 SKIP_LOCAL_SEC_SCANS=0`.
- GitHub CI (+local CI: Act cli):
  - `python-lint-test.yml`: Lint, Unit tests, Pyright. Integration/E2E run under Act (schedule/manual).
  - `meta-linters.yml`: Actionlint, Yamlfmt, Hadolint on relevant changes.
  - `semgrep.yml`, `codeql.yml`: Security scans on PR/schedule/manual.
  - `trivy_pip-audit.yml`: pip-audit + Trivy on dep/Docker changes and schedule.
- Tests & coverage: `tests/unit`, `tests/integration`, `tests/e2e`. Fast path example: `make unit`. Coverage HTML: `reports/coverage`.
- Logging: App logging in `backend/config.py` (level via `LOG_LEVEL`, Rich when TTY; optional file rotation via `APP_LOG_DIR`). Script logging helpers in `scripts/common.sh`. Unit tests cover both.

 - Use Makefile targets for common checks; see `Makefile` and `docs/AI_instructions.md` for details.

### Pre-commit in restricted environments

Run `make pre-commit`, which pins caches locally via `UV_CACHE_DIR=./.uv-cache` and `PRE_COMMIT_HOME=./.pre-commit-cache`.
If outbound network is unavailable and required wheels are not already cached, uv may fail (e.g., fetching `filelock`). Populate caches once in a networked environment or manually place the needed wheels under `./.uv-cache` to reuse offline.

---

**CI/Act Environment Alignment**
- **Problem (historical):** Using a separate venv name (like `.venv-ci`) under act could drift from tools expecting `.venv`, leading to missing imports (e.g., `dotenv`, `rich`).
- **Current Solution:**
  - **Single Pyright config:** `pyrightconfig.json` no longer sets `venvPath`/`venv`; Makefile passes `--pythonpath` so Pyright analyzes against the active interpreter.
  - **Standardize on uv defaults:** We no longer set `UV_PROJECT_ENVIRONMENT`; uv creates/uses the in-project `.venv` by default.
  - **Safe checkout under act:** Workflows set `clean: false` for `actions/checkout` so `--bind` doesn’t remove local files.
  - Result: Consistent type checking and tests across local, CI, and act without hard-coding venv names into Pyright.
- **Where to look:**
  - `Makefile` targets listed above; interpreter selection and `--pythonpath` wiring.
  - `.github/workflows/python-lint-test.yml` environment no longer forces a venv name.
