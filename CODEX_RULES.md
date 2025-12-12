**Purpose**
- Rules for Codex so the assistant and contributors follow the same conventions when working via Codex.

**Scope**
- Applies to Codex sessions in this repo. Content is adapted from files under `.cursor/rules/` and aligned with our repo’s tooling and the Codex CLI execution model.

**General**
- Run commands from the repo root.
- Prefer local tools over Docker for dev; use Docker where required for parity or stack bring‑up.
- Use `ripgrep` (`rg`) for searches; read files in chunks (≤250 lines) to avoid truncation.
- Avoid destructive actions without explicit user request (e.g., volume deletion, force resets).

**Linting, Formatting, Types**
- Ruff for linting/formatting: `ruff check . --fix`, `ruff format .`.
- No `print` in app/tests (Ruff T201). Use `logging`.
- Pyright for types: `pyright .` (use `# type: ignore[...]` sparingly with justification).

**Testing**
- See `docs/testing_approach.md` for test suites, commands,
  environment configuration, and recommended patterns.

**Terminal and Python Execution**
- Use the project venv Python: `.venv/bin/python`.
- Run pytest as a module to avoid import issues: `.venv/bin/python -m pytest ...`.
- Never set `PYTHONPATH`.
- For commands that might page or hide output (e.g., `git log`, `git diff`), force full output in non‑interactive logs, e.g., `bash -o pipefail -c 'git --no-pager log | cat'`.
- Prefer Makefile targets for common checks (`make pyright`, `make ruff-fix`, `make unit`, `make pre-commit`).

**Imports and Dependencies**
- If `ModuleNotFoundError`: ensure editable install: `uv pip install -e .` (or `uv sync`).
- Prefer optional deps in `pyproject.toml`

**Docker Volumes Safety**
- Never remove production volumes. Do not use `-v` on non‑test stacks.
- Prefer `docker compose down` without `-v` unless clearly operating on test‑only resources.

**JSON Schemas**
- Validate JSON syntax before edits (`jsonlint` or equivalent) and ensure schema validity when modifying `*.schema.json` or `schemas/**/*.json`.

**LangChain**
- Use env vars for API keys and secrets.
- Add robust error handling and logging around API calls.

**Logging**
- Create log files under `logs/` only.
- Use structured logging and appropriate levels.

**Git Usage**
- Use Conventional Commits.
- Use a dev branch for new work unless directed otherwise.
- Avoid interactive pagers in logs: `git --no-pager ...`.

**Local vs CI Tools**
- Prefer locally installed tools for speed; match CI only when parity is required.
- For Act usage and caching details, see `docs/AI_instructions.md`.

**After Edits**
- When appropriate, run tests to catch regressions. 
- If failure persists after fix, stop and surface logs to decide whether tests or code are incorrect (describe the test’s intent vs actual behavior before changing either).

**Planning with Codex**
- For multi‑step or ambiguous tasks, use Codex’s plan tool to outline concise steps and keep exactly one step in progress.
- Do not perform destructive actions as part of planning. Execute only after plan confirmation when significant.

**Key Project Docs**
- `docs/AI_instructions.md`
- `docs/testing_approach.md`

**Notes**
- This document consolidates `.cursor/rules/*` semantics for Codex CLI. See `.cursor/rules/` for the original Cursor‑specific rule files.
