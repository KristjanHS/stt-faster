# CLAUDE.md — Project Instructions for Claude Code

## What This Is

Batch speech-to-text pipeline built on `faster-whisper` (Estonian default, English/multi via model presets). Backend in Python 3.12, Docker-packaged, with a frontend, scripts/, and a unit/integration/e2e test split.

<!-- L1=this file (always), L2=docs/ (on demand via trigger table), L3=.claude/rules/ (path-gated) -->

## Reference Index (encounter a problem? check here)

| Trigger | Read |
|---------|------|
| Quick-start, `./run_uv.sh`, Makefile targets (`make unit`/`pyright`/`ruff-fix`/`pre-commit`/`integration-local`/`docker-build-prod`/`export-reqs`), pre-push env toggles | `README.md` (+ `Makefile` for the full target list) |
| Pre-commit framework, GitHub Actions workflows, local CI with Act, pre-push Act sequence | `docs/AI_instructions.md` |
| Test layout (unit / integration / e2e), commands, env vars, patterns | `docs/testing_approach.md` |
| Transcription pipeline, faster-whisper models, presets, output layout | `docs/Transcription_solution.md` |
| Pipeline stages (preprocess → variants → transcribe → postprocess) | `docs/stt_stages.md` |
| Codex/agent operational rules + Codex-specific guardrails | `AGENTS.md`, `CODEX_RULES.md` |
| Cursor-specific rules (rule per `.mdc` file) | `.cursor/rules/` |

## Critical Rules

1. **Use `.venv/bin/python` directly. Never set `PYTHONPATH`.** Editable install (`uv pip install -e .`) handles import paths.
2. **`ModuleNotFoundError` ⇒ re-run editable install** (`uv pip install -e .` or `uv sync`) before investigating — stale installs are the dominant cause.
3. **No `print` in app or test code.** Use `logging` (Ruff T201 enforces this). Log files only under `logs/`; structured logging with proper levels.
4. **Never `docker compose down -v` on non-test stacks** — removes volumes and any persisted state. Plain `down` only, unless the compose file is explicitly test-only.
5. **Run pre-commit + tests after edits.** If a failure persists after a fix, stop and surface logs before retrying (max three attempts). State expected vs. actual before deciding whether to change the test or the code.
6. **No summary docs.** Don't create `TASK_SUMMARY.md`-style trailing artifacts. End summaries belong in chat / the commit message unless explicitly requested. Plans (when approved) go in `docs_AI_coder/plan.md`.
7. **Conventional Commits.** Primary branch is `main`.
8. **Secrets stay out of git.** `.secrets.baseline` (detect-secrets) gates this — don't bypass.

## Large Files

Thresholds + intent-declaration rule in `~/.claude/rules/reading-large-files.md` (auto-loads on `.py`/`.md` reads). Quick listing of current large files: `wc -l backend/**/*.py tests/**/*.py | sort -rn | head`.

## Tooling Quickref

- Lint/format: `make ruff-fix`, `make ruff-format` (or `ruff check . --fix`, `ruff format .`)
- Types: `make pyright`
- Tests: `make unit`, `make integration`, `make integration-local`, `make e2e`
- Pre-commit: `make pre-commit` (uses local-pinned `.uv-cache` / `.pre-commit-cache` for restricted environments)
- Pre-push hook install: `make setup-hooks`
- Env install: `./run_uv.sh`

When running `pytest tests/ -q` over the full suite, pass `timeout: 300000` to the Bash tool up front — it exceeds the default two-minute window. Targeted sub-suites don't need the override.

## Rules Index (path-gated — auto-loaded when touching matching files)

| Rule file | Triggers on | Key content |
|-----------|-------------|-------------|
| `.claude/rules/plan-hygiene.md` | `docs/**`, `docs_AI_coder/**`, `CLAUDE.md`, `.claude/rules/**` | Write-gate (no plan doc for ≤2-file/1-session work), archival rules, multi-stage gating, feature-removal grep |
| `.claude/rules/rule-authoring.md` | `.claude/rules/**` | Copy shipped-fix form verbatim when adding rule examples — design-doc forms often carry the pre-fix shape |
