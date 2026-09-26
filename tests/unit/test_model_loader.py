"""DeviceSelector: the CPU torch variant must never select cuda from a preset (SIGABRT at first encode)."""

from __future__ import annotations

import os

import pytest

from backend.model_config import ModelConfig
from backend.model_loader import DeviceSelector, TorchBuild

pytestmark = pytest.mark.skipif(bool(os.getenv("STT_DEVICE")), reason="STT_DEVICE overrides preset selection")


def _selector(build: TorchBuild) -> DeviceSelector:
    return DeviceSelector(torch_build=lambda: build)


def test_cpu_torch_variant_downgrades_cuda_preset_to_cpu() -> None:
    config = ModelConfig(model_id="m", device="cuda", compute_type="float16")

    assert _selector("cpu").select(config) == ("cpu", "int8")


@pytest.mark.parametrize("build", ["cuda", "absent"])
def test_cuda_or_lean_install_keeps_cuda_preset(build: TorchBuild) -> None:
    config = ModelConfig(model_id="m", device="cuda", compute_type="float16")

    assert _selector(build).select(config) == ("cuda", "float16")


def test_cpu_preset_is_untouched_by_cpu_torch_variant() -> None:
    config = ModelConfig(model_id="m", device="cpu", compute_type="int8")

    assert _selector("cpu").select(config) == ("cpu", "int8")
