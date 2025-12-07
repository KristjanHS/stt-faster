#!/usr/bin/env python3
"""Root-level pytest configuration and fixtures."""

from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from rich.console import Console
import pytest

REPORTS_DIR = Path("reports")
LOGS_DIR = REPORTS_DIR / "logs"

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


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Provides the absolute path to the project root directory.

    Moved from e2e/conftest.py to be available across all test types.
    """
    return Path(__file__).parent.parent
