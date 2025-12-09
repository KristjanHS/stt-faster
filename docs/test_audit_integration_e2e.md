# Integration & E2E Test Audit: Low-Value Tests

This document identifies low-value tests in integration and e2e test suites that should be considered for removal or rewriting.

## Integration Tests Audit

### Trivial Help Text Tests
These test help text without validating behavior:

- [x] 1. **`tests/integration/test_cli_commands.py::TestCLIHelp::test_no_command_shows_help`**
   - ~~Just checks if "usage:" or "help" appears in output~~
   - **Status**: ✅ Removed

- [x] 2. **`tests/integration/test_cli_commands.py::TestCLIHelp::test_process_help_shows_options`**
   - ~~Just checks if "input_folder" and "--preset" appear in help~~
   - **Status**: ✅ Removed

- [x] 3. **`tests/integration/test_cli_commands.py::TestCLIHelp::test_status_help_shows_options`**
   - ~~Just checks if "--verbose" appears in help~~
   - **Status**: ✅ Removed

### Trivial Code Inspection Tests
These check if strings exist in source code:

- [x] 4. **`tests/integration/test_variant_systems_comparison.py::TestVariantSystemIsolation::test_new_system_does_not_modify_legacy_code`**
   - ~~Checks if "def run_variant(" and "--use-new-variants" exist in script file~~
   - **Status**: ✅ Removed

- [x] 5. **`tests/integration/test_variant_systems_comparison.py::TestVariantSystemIsolation::test_both_systems_can_coexist`**
   - ~~Just checks if `len(variants) == 15`~~
   - **Status**: ✅ Removed

## E2E Tests Audit

### Trivial Help/Version Checks
These check output strings without behavior validation:

- [x] 6. **`tests/e2e/test_production_container.py::TestProductionRuntime::test_python_version`**
   - ~~Just checks if "Python 3.12" appears in output~~
   - **Status**: ✅ Removed during alignment check

- [x] 7. **`tests/e2e/test_real_transcription.py::TestRealTranscription::test_transcribe_accepts_preset_argument`**
   - ~~Just checks that no error messages appear (negative assertion)~~
   - **Status**: ✅ Enhanced - Now verifies preset behavior (transcription succeeds, output is valid, language matches preset)

### Trivial Folder Existence Checks
These check folder creation without validating behavior:

- [x] 8. **`tests/e2e/test_real_transcription.py::TestRealTranscription::test_transcribe_creates_processed_folder`**
   - ~~Just checks if folders exist after running command~~
   - **Status**: ✅ Removed - Already covered by `test_transcribe_real_mp3_file`

### Trivial Default Value Checks
These check default values without testing behavior:

- [x] 9. **`tests/e2e/test_real_transcription.py::TestRealTranscription::test_transcribe_real_mp3_file`** (partial)
   - ~~Contains check: `assert config.rnnoise_model == "models/sh.rnnn"`~~
   - **Status**: ✅ Removed - Default value assertion removed from E2E test

## Summary

- **Integration Tests**: 5 low-value tests identified (5 completed, 0 pending)
- **E2E Tests**: 4 low-value tests/assertions identified (4 completed, 0 pending)
- **Total**: 9 candidates for removal/rewriting (9 completed, 0 pending)

## Test Coverage Status

**Current Coverage** (unit + integration tests): **77%**
- Total statements: 1715
- Missing: 389
- Covered: 1326

**Coverage Analysis**:
- Integration tests provide good coverage of component interactions
- E2E tests validate end-to-end workflows but don't contribute to coverage metrics (real external dependencies)
- Coverage is healthy after removing ~45 low-value unit tests

## Recommendations

1. **Remove all help text tests** - Help text is documentation, not critical functionality
2. **Remove code inspection tests** - These test source code structure, not runtime behavior
3. **Consolidate folder existence checks** - Already covered by actual transcription tests
4. **Remove default value checks from E2E tests** - Defaults should be tested in unit tests
5. **Enhance negative assertion tests** - Test actual behavior, not just absence of errors

