# Testing Approach

A high-level guide to testing with clear separation between unit, integration, and E2E tests.

## Test Hierarchy

| Test Level | Speed | Mocking Strategy | When to Use | Folder |
|-----------|-------|------------------|-------------|--------|
| **Unit** | Fast (<5s) | Mock everything external | Test single components in isolation | `tests/unit/` |
| **Integration** | Medium (~30s) | Mock only external APIs/models | Test component interactions | `tests/integration/` |
| **E2E** | Slow (minutes) | No mocks, real everything | Validate end-to-end workflows | `tests/e2e/` |

## Core Principles

1. **Test Isolation**: Each test runs independently without state leakage
2. **Clear Boundaries**: Folder-based organization by test type (unit/integration/e2e) - **use folders, not markers** for test type classification
3. **pytest-Native**: Prefer pytest features and fixtures over unittest patterns
4. **DRY Fixtures**: Shared setup in hierarchical `conftest.py` files. Never use monkeypatching in permanent tests.
5. **Behavior Over Implementation**: Test what the code does, not how it does it. Focus on business logic and user-facing behavior.

## Test Fixtures

Fixtures are organized hierarchically through `conftest.py` files:
- **Root** (`tests/conftest.py`): Available to all tests
- **Unit** (`tests/unit/conftest.py`): Auto-blocks network, provides temp resources
- **Integration** (`tests/integration/conftest.py`): GPU checks, model caching, CLI test helpers

**Key fixture scopes**:
- `session`: Expensive setup reused across all tests (model downloads, GPU checks)
- `function` (default): Fresh state for each test (temp databases, temp folders)

## Environment Configuration

**Model Caching:** Set `USE_CACHED_MODEL=true` (default) to use cached tiny Whisper model (~75MB) or `false` for mock-only mode (offline/faster).

## Docker E2E Execution

- Docker-backed end-to-end suites live under `tests/e2e/` and are executed by default when running `make e2e` (or `.venv/bin/python -m pytest tests/e2e`). There is no longer a `RUN_DOCKER_E2E` gate—Docker tests are the default path.
- GPU diagnostics are now part of the default E2E execution; if you need to skip them (e.g., on CPU-only runners) use pytest markers such as `-m 'not gpu'`.

## Mocking Strategy

| Component | Unit Tests | Integration Tests | E2E Tests |
|-----------|-----------|-------------------|-----------|
| **Model Loading** | ✅ Mock | ✅ Mock | ❌ Real |
| **Transcription Logic** | ✅ Mock | ❌ Real | ❌ Real |
| **Database/File Operations** | ❌ Real (temp) | ❌ Real (temp) | ❌ Real |
| **Network Calls** | ✅ Blocked | ✅ Mock | ❌ Real |

## Test Markers

**Important**: Test type (unit/integration/e2e) is determined by folder location, **not markers**. Do not use `@pytest.mark.integration` or similar test-type markers.

Markers indicate cross-cutting concerns (use across all test folders):
- `@pytest.mark.slow` - Tests >30s (model downloads, processing)
- `@pytest.mark.network` - Requires internet (HuggingFace API)
- `@pytest.mark.gpu` - Requires GPU/CUDA hardware
- `@pytest.mark.docker` - Requires Docker daemon

## GPU Testing

GPU tests validate system setup and actual GPU functionality (behavior tests):
- NVIDIA drivers and CUDA/cuDNN libraries
- ctranslate2 GPU device detection
- faster-whisper model loading on GPU
- **Actual transcription on GPU** (not just model loading)

**Use for**: Server deployment validation, GPU setup troubleshooting, verifying GPU transcription works  
**Skip when**: CPU-only CI (auto-skips), rapid development, testing transcription logic

## Best Practices

1. **Test Isolation**: Each test independent, no shared state
2. **Fixture Scoping**: Session for expensive setup, function for fresh state
3. **Offline Support**: Use `USE_CACHED_MODEL=false` or pre-download model
4. **Performance**: Real models slower than mocks - choose appropriately
5. **Behavior-Focused Testing**: Test business logic and user-facing behavior, not implementation details
6. **Folder-Based Organization**: Use `tests/unit/`, `tests/integration/`, `tests/e2e/` folders to indicate test type - do not use test-type markers
7. **Avoid Trivial Tests**: Don't test file existence, help text, version strings, or implementation details - test what the code does, not how

## What NOT to Test

These are examples of low-value tests that should be avoided:

- **File existence checks** - Would fail at build/import time anyway
- **Help text validation** - Documentation, not critical functionality
- **Version string checks** - No behavior validation
- **Import checks** - Would fail at runtime if broken
- **Implementation details** - Test behavior, not specific IDs, paths, or internal structure
- **Trivial defaults** - Test that defaults work, not every single default value
- **Code inspection** - Don't test if strings exist in source files

---

**Note**: This document provides a high-level overview for human readers. For detailed agent-focused tips and CI notes, see `docs/AI_instructions.md`.
