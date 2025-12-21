# E2E Tests Audit Report

## Summary

This audit identifies low-value, redundant, or misplaced e2e tests that should be rewritten, moved to integration tests, or deleted.

## Test Files Overview

1. **test_production_container.py** (555 lines) - Production Docker container tests
2. **test_real_transcription.py** (146 lines) - Real transcription with actual models
3. **test_gpu_diagnostics.py** (152 lines) - GPU diagnostics and GPU transcription
4. **test_docker_container.py** (231 lines) - Docker Compose dev container tests
5. **test_cli_commands_real_model.py** (47 lines) - CLI with real model

---

## 🔴 HIGH PRIORITY: Delete or Rewrite

### 1. test_real_transcription.py - REDUNDANT TESTS

**Issue**: Two tests that do essentially the same thing.

- `test_transcribe_real_mp3_file` (lines 26-92)
- `test_transcribe_accepts_preset_argument` (lines 94-146)

**Analysis**:
- Both tests transcribe `test.mp3` with `et-large` preset
- Both verify JSON output exists and contains segments
- Both check language is Estonian
- The second test's "preset argument" validation is trivial (just checking no error messages)

**Recommendation**: 
- **DELETE** `test_transcribe_accepts_preset_argument` - it's redundant
- **KEEP** `test_transcribe_real_mp3_file` but simplify it (remove duplicate assertions)

---

### 2. test_cli_commands_real_model.py - REDUNDANT

**Issue**: Single test that duplicates functionality already covered.

- `test_process_command_with_cached_model_e2e` (lines 19-46)

**Analysis**:
- Does the same thing as `test_real_transcription.py::test_transcribe_real_mp3_file`
- Uses `tiny_whisper_model` fixture instead of downloading model
- Tests CLI with real model, but this is already covered by:
  - `test_real_transcription.py` (e2e with real model download)
  - `test_integration/test_cli_commands.py::test_process_command_accepts_valid_folder` (integration with mocked model)

**Recommendation**: 
- **DELETE** entire file - functionality is redundant

---

### 3. test_docker_container.py - LOW VALUE TESTS

**Issue**: Several tests that don't provide meaningful e2e value.

#### 3.1 TestDockerTestExecution class (lines 163-196)

**Tests**:
- `test_pytest_available` - Checks if pytest is installed
- `test_unit_tests_discoverable` - Checks if pytest can discover tests
- `test_sample_unit_test_runs` - Runs a single unit test

**Analysis**:
- These are testing the test infrastructure, not the application
- If pytest isn't available, unit tests would fail anyway
- Running unit tests inside Docker doesn't add e2e value
- This is more of a "can we run tests in Docker" check, which is low value

**Recommendation**: 
- **DELETE** entire `TestDockerTestExecution` class

#### 3.2 TestDockerPythonEnvironment::test_venv_in_path (lines 147-150)

**Analysis**:
- Checks if `/opt/venv/lib/python3.12/site-packages` is in sys.path
- This is a trivial configuration check, not e2e behavior
- If venv isn't in path, imports would fail anyway

**Recommendation**: 
- **DELETE** this test - if venv isn't configured, other tests would fail

#### 3.3 TestDockerBuild::test_build_succeeds (lines 126-131)

**Analysis**:
- Just checks if image name appears in `docker compose images` output
- The fixture already builds the image, so if build fails, fixture fails
- This test adds no value

**Recommendation**: 
- **DELETE** this test - fixture failure is sufficient

#### 3.4 TestDockerImageOptimization (lines 216-231)

**Tests**:
- `test_no_cache_in_apt_lists` - Checks apt cache cleanup
- `test_debian_snapshot_configured` - Checks Debian snapshot sources

**Analysis**:
- These are Dockerfile optimization checks, not e2e behavior
- Should be validated during Dockerfile review, not in e2e tests
- Low value for e2e test suite

**Recommendation**: 
- **DELETE** entire `TestDockerImageOptimization` class
- Move to Dockerfile linting/validation if needed

---

## 🟡 MEDIUM PRIORITY: Consider Rewriting or Moving

### 4. test_docker_container.py - OVERLAP WITH PRODUCTION TESTS

**Issue**: Some tests overlap with `test_production_container.py`.

#### 4.1 TestDockerRuntime::test_healthcheck_command (lines 137-141)

**Analysis**:
- Tests healthcheck command works
- Similar to `test_production_container.py::test_healthcheck_works`
- But tests dev container (docker-compose) vs production container

**Recommendation**: 
- **KEEP** but consider if we need both dev and production healthcheck tests
- If dev container healthcheck is same as production, consolidate

#### 4.2 TestDockerPythonEnvironment::test_non_root_user (lines 152-155)

**Analysis**:
- Tests container runs as non-root
- Similar to `test_production_container.py::test_runs_as_non_root` and `test_user_id_at_runtime`

**Recommendation**: 
- **KEEP** if dev container has different user setup than production
- Otherwise, consider consolidating

---

### 5. test_gpu_diagnostics.py - CONDITIONAL VALUE

**Issue**: Tests are valuable only if GPU is available.

**Analysis**:
- All tests are marked `@pytest.mark.gpu` and will skip without GPU
- Tests are well-structured and test actual GPU functionality
- However, if project doesn't commonly use GPU, these tests may rarely run

**Recommendation**: 
- **KEEP** but document that these are optional
- Consider if GPU tests should be in separate test suite
- The tests themselves are well-written and valuable when GPU is available

---

## 🟢 KEEP: High Value Tests

### 6. test_production_container.py - MOSTLY GOOD

**Keep all tests** - they provide valuable production deployment validation:
- Build verification
- Security checks (non-root, no sudo, minimal packages)
- Cloud-native characteristics (healthcheck, labels, volumes)
- Actual transcription in production container
- These are true e2e tests that validate production readiness

**Minor suggestion**: 
- `TestProductionVolumes` class is empty (lines 359-361) - **DELETE** empty class

---

### 7. test_real_transcription.py - KEEP (after cleanup)

**Keep** `test_transcribe_real_mp3_file` after removing redundant test:
- Validates end-to-end transcription with real model download
- Tests actual transcription quality (Estonian language detection)
- Validates database integration
- This is a true e2e test

---

### 8. test_docker_container.py - KEEP (selective)

**Keep**:
- `TestDockerRuntime::test_healthcheck_command` - validates dev container healthcheck
- `TestDockerPythonEnvironment::test_non_root_user` - if different from production
- `TestDockerPythonEnvironment::test_hf_cache_directory` - validates volume mount
- `TestDockerVolumes::test_logs_volume_mount` - validates volume functionality

---

## Recommended Actions

### ✅ COMPLETED: Immediate Deletions

1. ✅ **DELETED** `tests/e2e/test_cli_commands_real_model.py` (entire file)
2. ✅ **DELETED** `test_real_transcription.py::test_transcribe_accepts_preset_argument`
3. ✅ **DELETED** `test_docker_container.py::TestDockerTestExecution` (entire class)
4. ✅ **DELETED** `test_docker_container.py::TestDockerImageOptimization` (entire class)
5. ✅ **DELETED** `test_docker_container.py::TestDockerBuild` (entire class, including test_build_succeeds)
6. ✅ **DELETED** `test_docker_container.py::TestDockerPythonEnvironment::test_venv_in_path`
7. ✅ **DELETED** `test_production_container.py::TestProductionVolumes` (empty class)

### ✅ COMPLETED: Simplifications

1. ✅ **SIMPLIFIED** `test_real_transcription.py` - removed redundant test, kept only `test_transcribe_real_mp3_file`

### Consider Moving to Integration

1. **CONSIDER** moving some `test_docker_container.py` tests to integration if they don't require full Docker Compose stack

---

## Impact Summary

**Files deleted**: 1 entire file (`test_cli_commands_real_model.py`)
**Tests deleted**: 8 individual tests/classes
**Tests remaining**: 24 high-value e2e tests
**Lines removed**: ~250 lines of low-value test code

### Test Count Breakdown (After Cleanup)
- `test_production_container.py`: 6 test classes, ~15 tests
- `test_docker_container.py`: 3 test classes, 4 tests (down from 6 classes, 9 tests)
- `test_real_transcription.py`: 1 test (down from 2 tests)
- `test_gpu_diagnostics.py`: 4 tests (unchanged)

---

## Test Coverage After Cleanup

After cleanup, e2e tests will focus on:
1. **Production container** - Full production deployment validation
2. **Real transcription** - Actual model download and transcription (1 test, not 3)
3. **GPU diagnostics** - GPU environment validation (when GPU available)
4. **Dev container basics** - Essential dev container functionality

This maintains comprehensive e2e coverage while removing redundancy and low-value tests.

