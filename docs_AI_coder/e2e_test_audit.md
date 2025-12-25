# E2E Test Audit

This document lists E2E tests that should be rewritten or removed, with the rationale.

## Rewrite or remove

- `tests/e2e/test_docker_container.py::TestDockerPythonEnvironment::test_hf_cache_directory`
  - File-existence check; low behavioral value. Remove or replace with a functional cache-use assertion.
- `tests/e2e/test_docker_container.py::TestDockerPythonEnvironment::test_non_root_user`
  - Duplicates production container checks; either remove or move to a dedicated security/policy test.
- `tests/e2e/test_production_container.py::TestProductionRuntime::test_help_flag_works`
  - Help-text validation is explicitly low value; remove or replace with a real command (e.g., `status`).
- `tests/e2e/test_production_container.py::TestProductionCloudNative::test_venv_in_path`
  - Implementation detail (PATH order); low behavior value; remove.
- `tests/e2e/test_production_container.py::TestProductionCloudNative::test_has_healthcheck`
  - Metadata-only; low value. Remove or replace with actual healthcheck execution.
- `tests/e2e/test_production_container.py::TestProductionCloudNative::test_healthcheck_works`
  - Does not execute the configured healthcheck; rewrite to run the image’s Healthcheck CMD or the real `--healthcheck` path.
- `tests/e2e/test_production_container.py::TestProductionLabels::test_has_oci_labels`
  - Metadata-only; better suited to Dockerfile linting/CI policy checks; remove or move.
- `tests/e2e/test_production_container.py::TestProductionSecurity::test_minimal_packages`
  - Brittle against base image changes; if required, move to a dedicated Docker policy test; otherwise remove.
- `tests/e2e/test_production_container.py::TestProductionImageBuild::test_image_size_reasonable`
  - Non-behavioral and size drift prone; move to CI guardrails if needed, otherwise remove.
- `tests/e2e/test_real_transcription.py::TestRealTranscription::test_transcribe_real_mp3_file`
  - Valuable but brittle (hardcoded words and exact language). Rewrite to validate schema, segments, non-empty transcript,
    and a tolerant language check (e.g., `et` or `est`), avoid exact keyword matching.
