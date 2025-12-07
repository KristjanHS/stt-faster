#!/usr/bin/env python3
"""Root-level pytest configuration and fixtures."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

REPORTS_DIR = Path("reports")
LOGS_DIR = REPORTS_DIR / "logs"

_JOBLIB_TEMP_DIR = Path(tempfile.mkdtemp(prefix="joblib_tmp_"))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_JOBLIB_TEMP_DIR))
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

# Ensure the temp dir is cleaned up after the test session finishes
atexit.register(shutil.rmtree, _JOBLIB_TEMP_DIR, ignore_errors=True)

# Hugging Face deprecated HF_HUB_ENABLE_HF_TRANSFER; normalize early to avoid warnings
_legacy_hf_transfer = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
if _legacy_hf_transfer and "HF_XET_HIGH_PERFORMANCE" not in os.environ:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = _legacy_hf_transfer


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: D401
    """Ensure report directories exist; preserve service URLs in local runs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Guardrail: ensure tests run with the project interpreter to avoid
    # environment/plugin drift when a global `pytest` is used.
    expected = Path.cwd() / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
    # If a project venv exists but we're not inside any venv, fail fast with guidance.
    in_venv = getattr(sys, "base_prefix", sys.prefix) != sys.prefix
    if expected.exists() and not in_venv:
        pytest.exit(
            "Detected non-venv Python interpreter.\nPlease run tests via '.venv/bin/python -m pytest' or 'make unit'.",
            returncode=3,
        )


# Set up a logger for this module
logger = logging.getLogger(__name__)
console = Console()


@pytest.fixture(scope="session", autouse=True)
def _migrate_hf_transfer_env() -> None:
    """Prefer new Hugging Face transfer env var and drop deprecated one.

    Removes HF_HUB_ENABLE_HF_TRANSFER to avoid deprecation warnings and maps its
    value to HF_XET_HIGH_PERFORMANCE if the latter is not already set.
    """
    legacy = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if legacy and "HF_XET_HIGH_PERFORMANCE" not in os.environ:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = legacy


@pytest.fixture(scope="session", autouse=True)
def _force_joblib_thread_backend() -> None:
    """Prefer thread-based parallelism to avoid SemLock in WSL."""
    try:
        import joblib.parallel as joblib_parallel
    except ImportError:
        return

    joblib_parallel.DEFAULT_BACKEND = "threading"
    joblib_parallel.DEFAULT_THREAD_BACKEND = "threading"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Provides the absolute path to the project root directory.

    Moved from e2e/conftest.py to be available across all test types.
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def tiny_whisper_model() -> "WhisperModel":
    """Cached tiny Whisper model for integration/E2E tests when enabled.

    Controlled by USE_CACHED_MODEL (defaults to true). Explicitly fails when the
    cache is disabled or if loading the model errors so dependent tests cannot
    silently skip.
    """
    use_cached = os.getenv("USE_CACHED_MODEL", "true").lower() in ("true", "1", "yes")

    if not use_cached:
        pytest.fail("USE_CACHED_MODEL=false disables the tiny Whisper cache required for integration/E2E tests.")

    try:
        from faster_whisper import WhisperModel

        logger.info("Loading tiny Whisper model for tests (one-time download if not cached)...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Tiny Whisper model loaded successfully (cached for future test runs)")
        return model
    except Exception as exc:  # pragma: no cover - defensive guard
        pytest.fail(f"Failed to load tiny Whisper model for tests: {exc}", pytrace=True)
