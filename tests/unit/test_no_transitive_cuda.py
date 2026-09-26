"""Guard against the CPU extras silently regressing to GPU torch.

Background: with `--extra cpu`, `uv sync` resolves `torch==X.Y.Z+cpu` and no
nvidia-* / cuda-* / triton wheels; a cu130 torch in a CPU venv would carry
~3 GB of CUDA libraries it never uses.

This test fails the moment `nvidia-cublas` shows up on a CPU install — which is
the canary the §5.8 of docs/plans/archived/2026-05-21-cuda-deps-cpu-gpu-extras-design.md
explicitly calls out. Cheap and runs in unit-test time (no torch import).
"""

from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest


def _gpu_variant_selected() -> bool:
    # Ask the same selector run_uv.sh uses, so a linked worktree that inherits
    # the main checkout's `.stt-variant.local` resolves identically here.
    selector = Path(__file__).resolve().parents[2] / "scripts" / "select_variant.sh"
    result = subprocess.run([str(selector)], capture_output=True, text=True, check=True)  # noqa: S603
    return result.stdout.strip() == "cu130"


def test_cpu_install_has_no_transitive_cuda() -> None:
    if _gpu_variant_selected():
        pytest.skip(".stt-variant.local opts into cu130 — nvidia wheels expected")

    try:
        dist = distribution("nvidia-cublas")
    except PackageNotFoundError:
        return

    pytest.fail(
        f"nvidia-cublas=={dist.version} is installed under the CPU extra. "
        "The CPU/GPU uv-extras split has regressed: CI or the local checkout "
        "is pulling GPU wheels into a CPU venv. Verify that every `uv sync` "
        "call in .github/workflows/*.yml passes `--extra cpu`, and that "
        "`.stt-variant.local` (if present) matches the intended variant."
    )
