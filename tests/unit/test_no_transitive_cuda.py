"""Guard against the CPU extras silently regressing to GPU torch.

Background: with `--extra cpu`, `uv sync` resolves `torch==X.Y.Z+cpu` and no
nvidia-* / cuda-* / triton wheels. With no extra (PyPI-default torch via
pyannote.audio's transitive resolution) the CPU runner ends up with ~3 GB of
CUDA libraries it never uses.

This test fails the moment `nvidia-cublas` shows up on a CPU install — which is
the canary the §5.8 of docs/plans/2026-05-21-cuda-deps-cpu-gpu-extras-design.md
explicitly calls out. Cheap and runs in unit-test time (no torch import).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest


def _gpu_variant_selected() -> bool:
    variant_file = Path(__file__).resolve().parents[2] / ".stt-variant.local"
    if not variant_file.exists():
        return False
    return variant_file.read_text().strip() == "cu130"


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
