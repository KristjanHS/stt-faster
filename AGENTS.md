## Purpose  
How to work safely and correctly in this repo. Follow unless a human overrides.

- Important refs: `docs/AI_instructions.md`, `docs/testing_approach.md` (use `@<relative path>` with tools).  

## Repo profile  
- Root: `stt-faster`; Python 3.12 via `.venv`; faster-whisper; Docker Compose.  
- Key dirs: `backend/`, `scripts/`, `tests/` (unit/integration/e2e/ui), `docker/`, `reports/`.

## Core rules  
- Run from repo root; set `workdir` explicitly; use workspace-relative paths.  
- Use `.venv/bin/python`; never set `PYTHONPATH`.  
- No secrets in git (`.secrets.baseline` enforced).  
- Prefer local tools; check `which` before Docker/CI wrappers.  
- Avoid `docker compose down -v` except for test-only stacks.  
- Pipe pager-prone commands with `| cat` (use `bash -o pipefail -c 'cmd | cat'`).  
- Git: Conventional Commits, default to `dev`, use `git --no-pager log`.  
- When plan-only, gather context, output a numbered plan, stop; once approved, store in `docs_AI_coder/plan.md`.

## Commands & tasks (stop on first failure)  
- Unit tests: `.venv/bin/python -m pytest tests/unit -q`  
- Integration: `.venv/bin/python -m pytest tests/integration -q`  
- Pre-commit: `uv run pre-commit run --all-files`  
- If UI/log outputs change, update snapshots/expectations instead of disabling checks.  
- After edits, run builds/tests; on failure, share logs before retrying (max three attempts).  
- Integration markers: default full `make integration`; AI agents may set `INTEGRATION_MARKERS="not gpu"` if no GPU.  
- Suites: unit (fast, sockets blocked, `-n auto`), integration (one real component, network allowed), e2e
  (Docker Compose), UI (Playwright `--no-cov`); unit tests must avoid real network I/O.  
- Run pytest as module (e.g., `.venv/bin.python -m pytest tests/`).  
- Use Make targets when available: `make pyright|ruff-fix|ruff-format|unit|integration-local|pre-commit|pre-push`.

## Code & deps  
- Lines ≤120; small functions; explicit error handling; add/update tests with behavior changes.  
- Ruff for lint/format (`ruff check . --fix`, `ruff format .`); Pyright for types; avoid `print`—use logging.  
- Logs live under `logs/` with descriptive names; structured logging and proper levels.  
- LangChain: secrets via env vars, robust error handling, logging around API calls.  
- Imports/deps: for `ModuleNotFoundError`, ensure editable install (`pip install -e .`); prefer extras in
  `pyproject.toml` with thin requirements wrappers.  

## Safe edits & docs  
- For risky refactors/schemas, propose a Tasklist plan first; do not alter Dockerfiles/Compose/security configs without
  risks/mitigations.  
- Fixing tests: state expected vs actual, then decide whether to change test or code.  
- JSON schemas: run `jsonlint` before editing; validate after changes.  
- Editing `.cursor/rules`: study existing files; keep header fields (`description`, `type`, `globs`, `alwaysApply`),
  valid YAML, and bullet content.  
- No summary/overview files (e.g., `TASK_SUMMARY.md`); keep end summaries in chat unless explicitly asked.  

## Etiquette  
- Use relative paths and explicit working dirs for tools.  
- When unsure about security or data handling, stop and ask.
