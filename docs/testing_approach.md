# Testing Approach

A high-level guide to testing with real local models and external services.

## Quick Start

```bash
# Run unit tests (fast, isolated)
.venv/bin/python -m pytest tests/unit/ -v

# Run integration tests (with real models)
.venv/bin/python -m pytest tests/integration/ -v

# Run GPU tests (requires GPU hardware)
.venv/bin/python -m pytest -m gpu -v

# Skip GPU tests
.venv/bin/python -m pytest -m "not gpu" -v
```

## Core Principles

1. **Test Isolation**: Each test runs independently without state leakage
2. **Real Model Testing**: Use real local models for ML validation, mock external services
3. **Readability**: Clear setup → action → assertion phases in tests
4. **pytest-Native**: Prefer pytest features over unittest patterns
5. **DRY Fixtures**: Shared test setup in `conftest.py` files, not duplicated in tests

## Test Organization

```
tests/
├── conftest.py              # Root fixtures (all tests)
├── unit/
│   ├── conftest.py         # Unit test fixtures + network blocking
│   └── test_*.py           # Fast, isolated unit tests
├── integration/
│   ├── conftest.py         # Integration fixtures (CLI, GPU)
│   └── test_*.py           # Real models + mocked services
└── e2e/
    ├── conftest.py         # E2E-specific fixtures
    └── test_*.py           # Full system validation
```

## Test Fixtures

### Fixture Hierarchy

Fixtures are organized hierarchically through `conftest.py` files:

**Root Level** (`tests/conftest.py`) - Available to all tests:
- `project_root`: Session-scoped path to project root
- `pytest_sessionstart`: Ensures report directories exist, validates Python interpreter

**Unit Tests** (`tests/unit/conftest.py`) - Unit tests only:
- `_disable_network_for_unit_tests`: Auto-blocks network access (pytest-socket)
- `temp_db`: Temporary `TranscriptionDatabase` with cleanup
- `temp_folder`: Temporary folder with automatic cleanup

**Integration Tests** (`tests/integration/conftest.py`) - Integration tests only:
- `cli_test_folder`: Temporary folder with empty audio file placeholders
- `real_audio_test_folder`: Temporary folder with actual test.mp3
- `cli_test_db`: Temporary database path for CLI testing
- `gpu_available`: Session-scoped GPU/CUDA availability check
- `ctranslate2_device_count`: Session-scoped CUDA device count from ctranslate2

### Fixture Best Practices

**When to create a fixture** (add to appropriate `conftest.py`):
- Multiple tests need the same setup
- Setup is complex or resource-intensive
- State needs isolation between tests
- Environment checks required (e.g., GPU availability)

**When NOT to create a fixture** (keep setup in test):
- Only one test needs it
- Setup is trivial (< 3 lines)
- Setup is test-specific and not reusable

**Fixture naming conventions**:
- Descriptive names: `cli_test_folder` not `folder1`
- Underscore prefix for auto-use: `_disable_network_for_unit_tests`
- Type hints: `-> Path`, `-> bool`

**Fixture scopes**:
- `session`: Expensive setup reused across all tests (e.g., `gpu_available`)
- `function` (default): Fresh state for each test (e.g., `temp_db`)
- `module`: Shared within a test module

## Environment Configuration

**Service URLs**: Use explicit environment variables for service endpoints
- `OLLAMA_URL` → Ollama service endpoint (default: `http://localhost:11434`)

**Docker/Compose**: Services set URLs for intra-container communication
- `OLLAMA_URL=http://ollama:11434`

**Configuration**: Service URLs can be overridden via environment variables

## Key Testing Concepts

### Service-Specific Integration Tests

Tests declare service dependencies using pytest markers and use the `integration` fixture for automatic environment handling.

## Testing Strategy Guidelines

### When to Use Real Models vs Mocks

| Component Type | Recommended Approach | Reasoning |
|---|---|---|---|
| **ML Models** (SentenceTransformer, CrossEncoder) | **Real Local Models** | Need to validate actual model behavior, performance, and component interactions |
| **External Services** (Ollama) | **Mock** | Network calls slow tests, external dependencies unreliable |
| **Database Operations** | **Mock** | Test data isolation, avoid external dependencies |
| **File I/O Operations** | **Mock** | Filesystem operations can be slow and unreliable |
| **Network APIs** | **Mock** | External API calls introduce latency and unreliability |
| **Configuration Systems** | **Real** | Need to test actual config loading and environment variables |

## Test Markers

Use pytest markers to selectively run tests:

```python
@pytest.mark.slow        # Tests taking >30s (model downloads, processing)
@pytest.mark.gpu         # Tests requiring GPU/CUDA hardware
@pytest.mark.docker      # Tests requiring Docker daemon
```

**Examples**:
```bash
pytest -m slow                    # Only slow tests
pytest -m "not slow"              # Skip slow tests
pytest -m "gpu and not slow"      # Fast GPU tests only
pytest -m "not gpu"               # Skip GPU tests (useful in CI)
```

## GPU Testing

GPU environment tests (`tests/integration/test_gpu_diagnostics.py`) validate system setup:

**What they check** (environment validation, NOT business logic):
- NVIDIA drivers installed (nvidia-smi works)
- CUDA/cuDNN libraries loadable
- ctranslate2 detects GPU devices
- faster-whisper can load models on GPU

**When useful**:
- New server deployment validation
- Developer GPU setup troubleshooting
- CI/CD with GPU runners

**When NOT useful**:
- Standard CI (CPU-only) - tests skip automatically
- Rapid development cycles - not needed every run
- Testing transcription logic - use CLI tests with real audio instead

**Note**: These are environment checks, not integration tests of GPU transcription functionality.

## Best Practices

### General
1. **Test Isolation**: Each test is independent, no shared state
2. **Cache Management**: Use fixtures that handle model caching properly
3. **Timeout Handling**: Set reasonable timeouts for model loading
4. **Performance**: Real models are slower than mocks - design accordingly
5. **Resource Management**: Ensure proper cleanup between tests
6. **Offline Support**: Tests work without internet connectivity

### Fixtures
1. **DRY Principle**: Define fixtures once in `conftest.py`, reuse everywhere
2. **Type Hints**: All fixtures use type hints for IDE support
3. **Scope Appropriately**: Session for expensive setup, function for isolation
4. **Auto-use Sparingly**: Only when truly needed (e.g., network blocking)

### Debugging
```bash
# Show fixture setup/teardown
pytest --setup-show tests/unit/test_transcribe.py

# List available fixtures
pytest --fixtures tests/integration/

# Verbose output with logging
pytest -vv --log-cli-level=DEBUG
```

---

**Note**: This document provides a high-level overview for human readers. For detailed agent-focused tips and CI notes, see `docs/AI_instructions.md`.
