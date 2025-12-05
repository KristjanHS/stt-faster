# Testing Approach

A high-level guide to testing with clear separation between unit, integration, and E2E tests.

## Quick Start

```bash
# Run unit tests (fast, isolated, no network)
.venv/bin/python -m pytest tests/unit/ -v

# Run integration tests (mocked external dependencies)
.venv/bin/python -m pytest tests/integration/ -v

# Run E2E tests (real models, real transcription)
.venv/bin/python -m pytest tests/e2e/ -v

# Skip slow tests (useful during development)
.venv/bin/python -m pytest -m "not slow" -v

# Skip GPU tests
.venv/bin/python -m pytest -m "not gpu" -v
```

## Test Hierarchy

| Test Level | Speed | Mocking Strategy | When to Use |
|-----------|-------|------------------|-------------|
| **Unit** (`tests/unit/`) | Fast (<5s) | Mock everything external | Test single components in isolation |
| **Integration** (`tests/integration/`) | Medium (~30s) | Mock only external APIs/models | Test component interactions |
| **E2E** (`tests/e2e/`) | Slow (minutes) | No mocks, real everything | Validate end-to-end workflows |

## Core Principles

1. **Test Isolation**: Each test runs independently without state leakage
2. **Clear Boundaries**: Unit tests mock external calls; integration tests mock external systems; E2E tests use real everything
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

**Configuration**: Environment variables can be used to override default behaviors

## Key Testing Concepts

### Service-Specific Integration Tests

Tests declare service dependencies using pytest markers and use the `integration` fixture for automatic environment handling.

## Testing Strategy Guidelines

### What to Mock at Each Level

| Component | Unit Tests | Integration Tests | E2E Tests |
|-----------|-----------|-------------------|-----------|
| **Model Loading** (`pick_model`) | ✅ Mock | ✅ Mock | ❌ Real |
| **Transcription Logic** | ✅ Mock | ❌ Real (with mocked model) | ❌ Real |
| **Database Operations** | ❌ Real (temp DB) | ❌ Real (temp DB) | ❌ Real (temp DB) |
| **File Operations** | ❌ Real (temp files) | ❌ Real (temp files) | ❌ Real (temp files) |
| **HuggingFace API** | ✅ Mock | ✅ Mock | ❌ Real |
| **Network Calls** | ✅ Blocked (pytest-socket) | ✅ Mock | ❌ Real |

### Example Mocking Patterns

**Unit Tests** - Mock external function calls:
```python
@patch("backend.transcribe.pick_model")
def test_process_file(mock_model):
    mock_model.return_value = MagicMock(...)
```

**Integration Tests** - Mock model loading, test logic flow:
```python
@patch("backend.transcribe.pick_model")
def test_cli_to_database_flow(mock_model):
    # Tests real CLI → Processor → Database integration
    # Only mocks the model loading
```

**E2E Tests** - No mocks, real models:
```python
@pytest.mark.slow
def test_real_transcription():
    # Real CLI subprocess call
    # Real model download/loading
    # Real transcription
```

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
