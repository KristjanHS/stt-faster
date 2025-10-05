# stt-faster

 repo for speech to text using faster-whisper HF model

---

## Prerequisites
- uv (https://astral.sh/uv) or pip
 - Docker & Docker Compose
 - Linux/WSL2
 - Optional: Act CLI for local CI runs (https://github.com/nektos/act).

---

## Quick Start

# Create .venv and install dev/test toolchain (editable install): `./run_uv.sh`
# Run pre-commit via uv (uses the venv): `uv run pre-commit run --all-files`
# Quick integration test run: `make integration-local`
# Start Docker services (if needed): `docker compose -f docker/docker-compose.yml up -d --build`

# Fallback to pip/venv: 
- Generate requirements.txt based on uv.lock: `make export-reqs`

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
- Tests & coverage: `tests/unit`, `tests/integration`, `tests/e2e`. Fast path example: `.venv/bin/python -m pytest tests/unit -q`. Coverage HTML: `reports/coverage`.
- Logging: App logging in `backend/config.py` (level via `LOG_LEVEL`, Rich when TTY; optional file rotation via `APP_LOG_DIR`). Script logging helpers in `scripts/common.sh`. Unit tests cover both.

 - Use Makefile targets for common checks; see `Makefile` and `docs/AI_instructions.md` for details.

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


## License

MIT License - see [LICENSE](LICENSE) file.
