# AI Agent Instructions

Action-first cheatsheet for automations.

### Core Policies

- **No PYTHONPATH**: Use editable install and `-m` execution
- **Absolute imports**: `from backend.config import HF_CACHE_DIR`
- **Module execution**: `.venv/bin/python -m backend.config`
- **Renovate validation**: `npx --package renovate renovate-config-validator renovate.json`

### Project Structure

- **Key Directories**: `backend/`, `tests/`, `docker/`, `scripts/`

## Testing Strategy

For test suite organization, quick commands, and overall guidance,
refer to `docs/testing_approach.md`. The notes below focus on
agent-specific integration details and health checks.

### Environment Configuration

**Configuration**: Centralized in `pyproject.toml`

### Real Model Testing Patterns

  **Model Configuration Testing:**
- Use `monkeypatch.setenv()` to override environment variables
- Test custom model repositories and commit hashes
- Reload configuration modules to pick up changes
- Validate configuration loading and parsing

### Model Cache Management

**Session-scoped caching**: Models loaded once per session and reused across tests
**Environment-based paths**: `HF_HOME` controls Hugging Face model cache storage
**Automatic cleanup**: Cache directories cleared after test sessions

**Cache Management Best Practices:**
- Ensure clean state between tests with cache reset
- Test operations with actual cached models

### Integration Test Best Practices
1. **Use pytest markers** instead of manual service checking
2. **Leverage the `integration` fixture** for service management
3. **Keep tests simple** - focus on one service requirement at a time
4. **Use descriptive test names** that indicate service requirements
5. **Handle service unavailability gracefully** - tests should skip with clear messages
6. **Use `monkeypatch`** for focused mocking instead of complex fixture chains

# CI/CD & Release Management

## Overview

This project uses GitHub Actions for CI/CD and security scanning, with automated release processes for promoting changes from `dev` to `main`.

## Available Workflows

### 1. CodeQL Analysis (`codeql.yml`)
- **Purpose**: Static code analysis for security vulnerabilities
- **Triggers**: Push to main, PR to main, weekly schedule, manual
- **Duration**: Up to 6 hours
- **Environment**: Runs on GitHub and under Act. Under Act, upload is skipped and results are summarized (informational only).

### 2. Semgrep Security Analysis (`semgrep.yml`)
- **Purpose**: Pattern-based security scanning
- **Triggers**: Push to main, PR to main, manual
- **Duration**: Up to 30 minutes
- **Environment**: Runs on GitHub and under Act. Under Act, SARIF upload is skipped; findings are still printed and saved.

### 3. CI Pipeline (`python-lint-test.yml`)
- **Purpose**: Linting, testing, and type checking
- **Triggers**: PR to main/dev, manual, weekly schedule
- **Jobs**: `lint` → `fast_tests` (depends on `lint`), `pyright`, `docker_smoke_tests`
- **Duration**: 5-15 minutes for fast tests

#### Pre-push local sequence (Act-powered)
- Order and behavior when you `git push` locally:
  1) **Pyright**: type checking (blocking)
  2) **Lint**: `ruff check` and `ruff format --check` (blocking)
  3) **Fast tests**: pytest fast suite (blocking)
  4) **Semgrep**: security scan (blocking)
  5) **CodeQL**: security analysis (informational only locally; does not block)

- Mapping to jobs/workflows:
  - Pyright → `python-lint-test.yml` job `pyright` (invoked via workflow_dispatch under Act)
  - Lint → `python-lint-test.yml` job `lint`
  - Fast tests → `python-lint-test.yml` job `fast_tests` (needs `lint`)
  - Semgrep → `semgrep.yml` job `semgrep`
  - CodeQL → `codeql.yml` job `analyze`

## Local Development with Act CLI

### Installation
```bash
# Linux/macOS
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

```

### Project Configuration
The `.actrc` file uses optimized runner images and mounts a persistent uv cache to speed up repeated runs.

- Persists uv cache across runs: `--container-options "--volume=${HOME}/.cache/uv:/uv-cache"`
- Sets `UV_CACHE_DIR=/uv-cache` and `UV_LINK_MODE=copy` for compatibility
- Auto-cleans runner containers: `--rm`

### Usage
```bash
# List workflows
act -l

# Run CI locally
act workflow_dispatch -W .github/workflows/python-lint-test.yml

# Run specific workflows
act workflow_dispatch -W .github/workflows/codeql.yml
act workflow_dispatch -W .github/workflows/semgrep.yml

# Run specific jobs
act workflow_dispatch -j lint
act workflow_dispatch -j pyright
act workflow_dispatch -j fast_tests
```

## Pre-commit Framework

This project uses a comprehensive pre-commit framework for code quality, formatting, and security scanning.

### Quick Setup
```bash
# Configure project hooks to use repo-managed scripts
make setup-hooks

# Run all hooks manually (uv preferred)
uv run pre-commit run --all-files
```

### Included Tools

- **Ruff**: Python linting and formatting
- **YAMLfmt**: YAML file formatting
- **Actionlint**: GitHub Actions validation
- **Hadolint**: Dockerfile validation
- **Bandit**: Python security scanning
- **Pyright**: Static type checking
- **Detect-secrets**: Secrets detection with baseline tracking

### Usage

**Automatic**: Hooks run on every `git commit`:
**Manual**: Run all hooks or specific ones:
```bash
pre-commit run --all-files
pre-commit run ruff
pre-commit run detect-secrets
```

### Configuration Files
- `.pre-commit-config.yaml`: Main configuration
- `.secrets.baseline`: Known secrets tracking
- `scripts/git-hooks/pre-commit`: Git hook script

### Troubleshooting
```bash
# Update hooks
pre-commit autoupdate

# Fix common issues
ruff check . --fix

```

### Requirements
- Local environment via uv or .venv
- Clean working tree (no uncommitted changes)

## Testing Workflows

### Manual Testing
1. GitHub UI: Actions tab → Select workflow → "Run workflow"
2. Act CLI: `act workflow_dispatch -W .github/workflows/[workflow].yml` (CI workflows only)
