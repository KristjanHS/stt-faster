# Testing Approach

A high-level guide to testing with real local models and external services.

## Quick Start

### Running Tests locally
- Run `.venv/bin/python -m pytest tests/integration/ -v`

## Core Principles

1. **Test Isolation**: Each test runs independently without state leakage
2. **Real Model Testing**: Use real local models for ML validation, mock external services
3. **Readability**: Clear setup → action → assertion phases in tests
4. **pytest-Native**: Prefer pytest features over unittest patterns

## Test Organization

- `tests/unit/` - Fast, isolated unit tests
- `tests/integration/` - Real models + mocked services
- `tests/e2e/` - Full system validation

## Test Utilities - to avoid code duplication.

Shared test utilities are centralized in `tests/conftest.py`
Unit test utilities: `tests/unit/conftest.py`
Integration test utilities: `tests/integration/conftest.py`
E2E test utilitie: `tests/e2e/conftest.py`

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

## Best Practices

1. **Cache Management**: Always use fixtures that handle model caching properly
2. **Timeout Handling**: Set reasonable timeouts for model loading operations
3. **Performance Expectations**: Real models are slower than mocks - account for this in test design
4. **Resource Management**: Ensure models are properly cleaned up between tests
5. **Offline Support**: Tests should work in environments without internet connectivity

---

**Note**: This document provides a high-level overview for human readers. For detailed agent-focused tips and CI notes, see `docs/AI_instructions.md`.
