# Declare phony targets (grouped for readability)
# Meta
.PHONY: help
# Setup
.PHONY: setup-hooks setup-uv export-reqs install-act setup-act new-project-bootstrap new-project-cleanup new-project-post new-project-git-setup
# Lint / Type Check
.PHONY: ruff-format ruff-fix yamlfmt pyright pre-commit
# Tests
.PHONY: unit integration e2e verify-variants
# Audio
.PHONY: preprocess-audio
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
INTEGRATION_MARKERS ?=

# Configurable pyright config path (default to repo config)
PYRIGHT_CONFIG ?= ./pyrightconfig.json

help:
	@echo "Available targets:"
	@echo "  -- Setup --"
	@echo "  setup-hooks        - Configure Git hooks path"
	@echo "  setup-uv           - Create venv and sync dev/test via uv"
	@echo "  new-project-bootstrap - Copy this repo to ~/projects/<SLUG> (requires SLUG=...)"
	@echo "  new-project-cleanup   - In the copied repo: strip app code and rename to <SLUG> (requires SLUG=...)"
	@echo "  new-project-post      - In the copied repo: install tooling + quick unit tests"
	@echo "  export-reqs        - Export requirements.txt from uv.lock"
	@echo "  install-act        - Install Act CLI for local CI runs"
	@echo "  setup-act          - Install and verify Act + Docker setup"
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
	@echo "  e2e          - Run e2e tests (sequential, Docker-based) and write reports"
	@echo "  verify-variants - Verify transcription parameters for all enabled variants"
	@echo ""
	@echo "  -- Audio --"
	@echo "  preprocess-audio  - Run preprocessing on tests/test.mp3 (AUDIO=..., OUT=... to override)"
	@echo ""
	@echo "  -- Docker (Production) --"
	@echo "  docker-build-prod    - Build production Docker image for end users"
	@echo "  docker-run-prod      - Run production Docker container (shows help)"
	@echo ""
	@echo "  -- Docker (Dev/Test) --"
	@echo "  docker-back          - Build and start dev/test services in background"
	@echo "  docker-unit          - Run unit tests inside dev container"
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

new-project-bootstrap:
	@if [ -z "$(SLUG)" ]; then \
		echo "Usage: make new-project-bootstrap SLUG=<project-slug>"; \
		exit 1; \
	fi
	./scripts/new_project_bootstrap.sh "$(SLUG)"

new-project-cleanup:
	./scripts/new_project_cleanup.sh $(if $(SLUG),$(SLUG),)

new-project-post:
	@echo "Setting up tooling (uv) and running a quick unit test sweep..."
	./run_uv.sh
	uv sync --group dev --group test --frozen || uv sync --group dev --group test
	uv run pre-commit install --install-hooks || true
	@echo "Git init/remote setup is now manual. Run scripts/new_project_git_setup.py yourself if you want automation."
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python -m pytest tests/unit -q; \
	else \
		echo "No .venv found; run ./run_uv.sh first"; \
		exit 1; \
	fi

new-project-git-setup:
	@echo "Initializing git + origin/branch via scripts/new_project_git_setup.py..."
	.venv/bin/python ./scripts/new_project_git_setup.py

# Install Act CLI for local CI runs
install-act:
	@echo ">> Installing Act CLI..."
	@if command -v act >/dev/null 2>&1; then \
		echo "✅ act is already installed (version: $$(act --version | head -1))"; \
	elif command -v brew >/dev/null 2>&1; then \
		echo "Installing act via Homebrew..."; \
		brew install act; \
		echo "✅ act installed successfully"; \
	else \
		echo "Installing act via official installer..."; \
		curl -fsSL https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash; \
		echo "✅ act installed successfully"; \
	fi
	@act --version

# Setup Act: install + verify Docker availability
setup-act: install-act
	@echo ""
	@echo ">> Verifying Docker availability..."
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "⚠️  Docker command not found. Install Docker Desktop for WSL2 integration."; \
		echo "   See: https://docs.docker.com/desktop/wsl/"; \
		exit 1; \
	elif ! docker info >/dev/null 2>&1; then \
		echo "⚠️  Docker is installed but not running. Start Docker Desktop to use act."; \
		echo "   After starting Docker, verify with: docker info"; \
		exit 1; \
	else \
		echo "✅ Docker is available and running"; \
	fi
	@echo ""
	@echo "✅ Act setup complete!"
	@echo ""
	@echo "Try these commands:"
	@echo "  act -l                                    # List all workflows"
	@echo "  act -W .github/workflows/python-lint-test.yml -j unit_tests"
	@echo "  make pre-push                              # Run with all checks enabled"

# Run local integration tests; prefer uv if available
integration:
	@if command -v uv >/dev/null 2>&1; then \
		uv run -m pytest tests/integration -vv $(if $(INTEGRATION_MARKERS),-m "$(INTEGRATION_MARKERS)",) ${PYTEST_ARGS}; \
	else \
		echo "uv not found. Either install uv (https://astral.sh/uv) and run './run_uv.sh', or ensure your venv is set up then run '.venv/bin/python -m pytest tests/integration -vv ${PYTEST_ARGS}'"; \
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
		.venv/bin/python -m pytest tests/unit -n auto --maxfail=1 -vv --html reports/unit.html --self-contained-html ${PYTEST_ARGS}; \
	else \
		uv run -m pytest tests/unit -n auto --maxfail=1 -vv --html reports/unit.html --self-contained-html ${PYTEST_ARGS}; \
	fi

# E2E test target (run without parallelization due to shared Docker container)
e2e:
	mkdir -p reports
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python -m pytest tests/e2e --maxfail=1 -vv --html reports/e2e.html --self-contained-html ${PYTEST_ARGS}; \
	else \
		uv run -m pytest tests/e2e --maxfail=1 -vv --html reports/e2e.html --self-contained-html ${PYTEST_ARGS}; \
	fi

# Verify variant transcription parameters
verify-variants:
	@echo "Verifying transcription parameters for all enabled variants..."
	@if [ -x .venv/bin/python ]; then \
		.venv/bin/python scripts/variant_checks/verify_all_variants.py; \
	else \
		uv run python scripts/variant_checks/verify_all_variants.py; \
	fi
	@echo "Variant verification completed."

preprocess-audio:
	@set -euo pipefail; \
	if [ ! -x .venv/bin/python ]; then \
	  echo "Missing .venv/bin/python. Run ./run_uv.sh first."; \
	  exit 1; \
	fi; \
	SRC=$(if $(AUDIO),$(AUDIO),tests/test.mp3); \
	OUT=$(if $(OUT),$(OUT),reports/preprocessed.wav); \
	echo "Preprocessing $$SRC -> $$OUT"; \
	STT_PREPROCESS_ENABLED=1 .venv/bin/python scripts/run_preprocess.py --input "$$SRC" --output "$$OUT"


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
	UV_CACHE_DIR=./.uv-cache PRE_COMMIT_HOME=./.pre-commit-cache uv sync --group dev --group test --frozen
	UV_CACHE_DIR=./.uv-cache PRE_COMMIT_HOME=./.pre-commit-cache uv run pre-commit run --all-files

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

# --- Production Docker (for end users) ---
.PHONY: docker-build-prod docker-run-prod

docker-build-prod:
	@echo ">> Building production Docker image..."
	docker build -t stt-faster:latest .
	@echo "✅ Production image built: stt-faster:latest"
	@echo "   Usage: ./scripts/transcribe-docker process /path/to/audio --preset turbo"

docker-run-prod:
	@echo ">> Running production Docker container..."
	@if ! docker image inspect stt-faster:latest >/dev/null 2>&1; then \
		echo "⚠️  Image not found. Building..."; \
		$(MAKE) docker-build-prod; \
	fi
	./scripts/transcribe-docker --help

# --- Dev/Test Docker (for developers and CI) ---
.PHONY: docker-back docker-unit

# Build and start dev/test docker services in background
docker-back:
	docker compose -f docker/docker-compose.yml up -d --build

# Run unit tests inside the dev app container (venv-safe)
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
