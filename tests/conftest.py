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
