# Declare phony targets (grouped for readability)
# Meta
.PHONY: help
# Setup
.PHONY: setup-hooks setup-uv export-reqs
# Lint / Type Check
.PHONY: ruff-format ruff-fix yamlfmt pyright pre-commit
# Tests
.PHONY: unit integration e2e
# Docker
.PHONY: docker-back docker-unit
# Security / CI linters
.PHONY: pip-audit semgrep actionlint
# CI helpers and Git
.PHONY: uv-sync-test pre-push

# Use bash with strict flags for recipes
SHELL := bash
.SHELLFLAGS := -euo pipefail -c

# Stable project/session handling
LOG_DIR := logs

# Configurable pyright config path (default to repo config)
PYRIGHT_CONFIG ?= ./pyrightconfig.json

help:
	@echo "Available targets:"
	@echo "  -- Setup --"
	@echo "  setup-hooks        - Configure Git hooks path"
	@echo "  setup-uv           - Create venv and sync dev/test via uv"
	@echo "  export-reqs        - Export requirements.txt from uv.lock"
	@echo ""
	@echo "  -- Lint & Type Check --"
	@echo "  ruff-format        - Auto-format code with Ruff"
	@echo "  ruff-fix           - Run Ruff lint with autofix"
	@echo "  yamlfmt            - Validate YAML formatting via pre-commit"
	@echo "  pyright            - Run Pyright type checking"
	@echo "  pre-commit         - Run all pre-commit hooks on all files"
	@echo ""
	@echo "  -- Tests --"
	@echo "  unit         - Run unit tests (local) and write reports"
	@echo "  integration  - Run integration tests (uv preferred)"
	@echo "  e2e          - Run e2e tests (local) and write reports"
	@echo ""
	@echo "  -- Docker --"
	@echo "  docker-back  - Build and start services in background"
	@echo "  docker-unit  - Run unit tests inside app container"
	@echo ""
	@echo "  -- Security / CI linters --"
	@echo "  pip-audit          - Export from uv.lock and audit prod/dev+test deps"
	@echo "  semgrep      - Run Semgrep locally via uvx (no metrics)"
	@echo "  actionlint         - Lint GitHub workflows using actionlint in Docker"
	@echo ""
	@echo "  -- CI helpers & Git --"
	@echo "  uv-sync-test       - uv sync test group (frozen) + pip check"
	@echo "  pre-push           - Run pre-push checks with all SKIP=0"

setup-hooks:
	@echo "Configuring Git hooks path..."
	@git config core.hooksPath scripts/git-hooks
	@echo "Done."

# uv-based setup
setup-uv:
	@./run_uv.sh

# Run local integration tests; prefer uv if available
integration:
	@if command -v uv >/dev/null 2>&1; then \
		uv run -m pytest tests/integration -q ${PYTEST_ARGS}; \
	else \
		echo "uv not found. Either install uv (https://astral.sh/uv) and run './run_uv.sh', or ensure your venv is set up then run '.venv/bin/python -m pytest tests/integration -q ${PYTEST_ARGS}'"; \
		exit 1; \
	fi

# Export a pip-compatible requirements.txt from uv.lock, except torch and editable project.
# With pip, torch installation must be done separately, eg:
#    pip install torch==1.8.0 --index-url https://download.pytorch.org/whl/cpu
#    pip install torch==1.7.1 --index-url https://download.pytorch.org/whl/cu128
export-reqs:
	@echo ">> Exporting requirements.txt from uv.lock (incl dev/test groups)"
	uv export --no-hashes --group test --locked --no-emit-project --no-emit-package torch --format requirements-txt > requirements.txt

# --- CI helper targets (used by workflows) -----------------------------------

# audits the already existing venv
pip-audit: export-reqs
	@echo ">> Auditing dependencies (based on requirements.txt)"
	uvx --from pip-audit pip-audit -r requirements.txt

uv-sync-test:
	uv sync --group test --frozen
	uv pip check

# New canonical unit test target
unit:
	mkdir -p reports
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python -m pytest tests/unit -n auto --maxfail=1 -q --html reports/unit.html --self-contained-html ${PYTEST_ARGS}; \
	else \
		uv run -m pytest tests/unit -n auto --maxfail=1 -q --html reports/unit.html --self-contained-html ${PYTEST_ARGS}; \
	fi

# E2E test target
e2e:
	mkdir -p reports
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python -m pytest tests/e2e -n auto --maxfail=1 -q --html reports/e2e.html --self-contained-html ${PYTEST_ARGS}; \
	else \
		uv run -m pytest tests/e2e -n auto --maxfail=1 -q --html reports/e2e.html --self-contained-html ${PYTEST_ARGS}; \
	fi


pyright:
	@# Determine interpreter path: prefer .venv, then system python
		@PY_INTERP=""; \
		if [ -x .venv/bin/python ]; then \
		PY_INTERP=".venv/bin/python"; \
	elif command -v python3 >/dev/null 2>&1; then \
		PY_INTERP=$$(command -v python3); \
	else \
		PY_INTERP=$$(command -v python); \
	fi; \
		if [ -x .venv/bin/pyright ]; then \
		.venv/bin/pyright --pythonpath "$$PY_INTERP" --project $(PYRIGHT_CONFIG); \
	else \
		uvx pyright --pythonpath "$$PY_INTERP" --project $(PYRIGHT_CONFIG); \
	fi

yamlfmt:
	# Ensure dev + test groups are present so later test steps still work
	uv sync --group dev --group test --frozen
	uv run pre-commit run yamlfmt -a

# Ruff targets (use uv-run to avoid global installs)
ruff-format:
	@if [ -x .venv/bin/ruff ]; then \
		.venv/bin/ruff format .; \
	else \
		uv run ruff format .; \
	fi

ruff-fix:
	@if [ -x .venv/bin/ruff ]; then \
		.venv/bin/ruff check --fix .; \
	else \
		uv run ruff check --fix .; \
	fi

# Run full pre-commit suite (dev deps required)
pre-commit:
	# Keep test deps installed to avoid breaking local test runs after this target
	uv sync --group dev --group test --frozen
	uv run pre-commit run --all-files

# Run the same checks as the Git pre-push hook, forcing all SKIP flags to 0
pre-push:
	SKIP_LOCAL_SEC_SCANS=0 SKIP_LINT=0 SKIP_PYRIGHT=0 SKIP_TESTS=0 scripts/git-hooks/pre-push

# Lint GitHub Actions workflows locally using official container
actionlint:
	@docker run --rm \
		--user "$(shell id -u):$(shell id -g)" \
		-v "$(CURDIR)":/repo \
		-w /repo \
		rhysd/actionlint:latest -color && echo "Actionlint: no issues found"

# Run Semgrep locally using uvx, mirroring the local workflow
semgrep:
	@if command -v uv >/dev/null 2>&1; then \
		uvx --from semgrep semgrep ci \
		  --config auto \
		  --metrics off \
		  --sarif \
		  --output semgrep_local.sarif; \
		echo "Semgrep SARIF written to semgrep_local.sarif"; \
		if command -v jq >/dev/null 2>&1; then \
		  COUNT=$$(jq '[.runs[0].results[]] | length' semgrep_local.sarif 2>/dev/null || echo 0); \
		  echo "Semgrep findings: $${COUNT} (see semgrep_local.sarif)"; \
		else \
		  COUNT=$$(grep -o '"ruleId"' -c semgrep_local.sarif 2>/dev/null || echo 0); \
		  echo "Semgrep findings: $${COUNT} (approx; no jq)"; \
		fi; \
	else \
		echo "uv not found. Install uv: https://astral.sh/uv"; \
		exit 1; \
	fi

# Build and start docker services in background
docker-back:
	docker compose -f docker/docker-compose.yml up -d --build

# Run unit tests inside the app container (venv-safe)
docker-unit:
	@# Ensure the app service is running before exec
	@if ! docker compose -f docker/docker-compose.yml ps -q app | xargs -r docker inspect -f '{{.State.Running}}' 2>/dev/null | grep -q true; then \
	  echo "app service is not running. Start it with 'make docker-back'."; \
	  exit 1; \
	fi
	mkdir -p reports
	docker compose -f docker/docker-compose.yml exec -T app \
	  /opt/venv/bin/python -m pytest tests/unit -vv --maxfail=1 \
	  --html reports/unit.html --self-contained-html ${PYTEST_ARGS}
