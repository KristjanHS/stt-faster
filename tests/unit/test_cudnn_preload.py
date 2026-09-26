from __future__ import annotations

import os
from pathlib import Path

from backend._cudnn_preload import register_nvidia_dll_dirs


def test_register_nvidia_dll_dirs_adds_each_bin_and_prepends_path(tmp_path: Path) -> None:
    for pkg in ("cublas", "cudnn"):
        (tmp_path / "nvidia" / pkg / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cuda_nvrtc" / "lib").mkdir(parents=True)  # no bin → skipped
    added: list[str] = []
    environ = {"PATH": "system-dirs"}

    dirs = register_nvidia_dll_dirs([str(tmp_path / "nvidia")], add_dll_directory=added.append, environ=environ)

    expected = [str(tmp_path / "nvidia" / pkg / "bin") for pkg in ("cublas", "cudnn")]
    assert dirs == added == expected
    assert environ["PATH"].split(os.pathsep) == [*expected, "system-dirs"]


def test_register_nvidia_dll_dirs_without_gpu_extra_leaves_path(tmp_path: Path) -> None:
    environ = {"PATH": "system-dirs"}
    assert register_nvidia_dll_dirs([], add_dll_directory=lambda _d: None, environ=environ) == []
    assert register_nvidia_dll_dirs([str(tmp_path)], add_dll_directory=lambda _d: None, environ=environ) == []
    assert environ == {"PATH": "system-dirs"}
