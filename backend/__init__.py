"""Backend package for the speech-to-text pipeline."""

# Must run before any submodule pulls in faster_whisper / ctranslate2.
# See backend/_cudnn_preload.py for the conflict this resolves.
from backend._cudnn_preload import preload_bundled_cudnn

preload_bundled_cudnn()
