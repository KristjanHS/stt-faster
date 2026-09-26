"""Pre-load PyTorch's bundled cuDNN before ctranslate2 dlopens the system one.

CTranslate2 4.x calls ``dlopen("libcudnn.so.9")`` at first CUDA inference
without a venv-aware RPATH, so on hosts that also have system cuDNN installed
(Ubuntu's ``cudnn9-cuda-12`` package puts 9.16 under ``/lib/x86_64-linux-gnu``)
the loader resolves to the system copy. Once that version is resident,
PyTorch's later ``torch.backends.cudnn._init()`` aborts with::

    cuDNN version incompatibility: PyTorch was compiled against (9, 20, 0)
    but found runtime version (9, 16, 0).

PyTorch 2.12 ships its bundled cuDNN under
``site-packages/nvidia/cudnn/lib/``. Loading the dispatcher (``libcudnn.so.9``)
with RTLD_GLOBAL before CT2 imports means CT2's later dlopen finds the bundled
9.20 already cached and reuses it. The dispatcher's RUNPATH is ``$ORIGIN``, so
its lazy-loaded sub-libs (``libcudnn_ops.so.9`` etc.) resolve from the same
bundled directory without needing to be preloaded individually.

Imported from ``backend/__init__.py`` so any backend submodule that pulls in
``faster_whisper`` benefits without per-call boilerplate. No-ops when running
on the CPU variant (no ``nvidia/cudnn/`` dir present).

Windows (``gpu-win`` extra): ctranslate2 LoadLibrary's ``cublas64_12.dll`` and cuDNN
sub-libs by name, so each ``site-packages/nvidia/*/bin`` goes on the DLL search path.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
from collections.abc import Callable, Iterable, MutableMapping
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def register_nvidia_dll_dirs(
    roots: Iterable[str],
    *,
    add_dll_directory: Callable[[str], object] | None = None,
    environ: MutableMapping[str, str] = os.environ,
) -> list[str]:
    add = add_dll_directory or getattr(os, "add_dll_directory")  # noqa: B009 - Windows-only attribute
    dirs = sorted(str(bin_dir) for root in roots for bin_dir in Path(root).glob("*/bin") if bin_dir.is_dir())
    for directory in dirs:
        add(directory)
    if dirs:  # native LoadLibrary calls search PATH, not only add_dll_directory dirs
        environ["PATH"] = os.pathsep.join([*dirs, environ.get("PATH", "")])
    return dirs


def preload_bundled_cudnn() -> None:
    if sys.platform == "win32":
        spec = importlib.util.find_spec("nvidia")  # namespace package from the nvidia-* wheels (gpu-win)
        register_nvidia_dll_dirs((spec.submodule_search_locations or []) if spec else [])
        return

    try:
        import torch
    except ImportError:
        return

    torch_file = getattr(torch, "__file__", None)
    if not torch_file:
        return

    cudnn_dir = os.path.join(os.path.dirname(torch_file), "..", "nvidia", "cudnn", "lib")
    dispatcher = os.path.abspath(os.path.join(cudnn_dir, "libcudnn.so.9"))
    if not os.path.isfile(dispatcher):
        return

    try:
        ctypes.CDLL(dispatcher, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        LOGGER.warning("cuDNN preload failed for %s: %s", dispatcher, exc)
