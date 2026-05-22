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
"""

from __future__ import annotations

import ctypes
import logging
import os

LOGGER = logging.getLogger(__name__)


def preload_bundled_cudnn() -> None:
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
